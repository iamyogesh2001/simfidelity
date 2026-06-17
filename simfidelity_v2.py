"""
SimFidelity v2: Distributed Simulation Framework for Fidelity-Aware
Synchronization Scheduling in Large-Scale Digital Twin Systems

DS-RT 2026 — Production Simulation Script
=========================================
Designed to be FAIR, not rigged:
  - Bursty, unpredictable state changes (not smooth decay)
  - Variable network jitter on sync duration
  - Node failure and recovery cycles
  - 10 independent random seeds → confidence intervals on all results
  - Scale to N=2000 nodes
  - Channel deliberately overloaded in stress tests
  - Adaptive must EARN its wins

Run time estimate on M4 Mac mini: 15-40 minutes
Dependencies: pip install simpy numpy matplotlib scipy

Outputs: ./figures_v2/  (PDF + PNG, publication-ready)
"""

import simpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy import stats
import os, time, warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')

os.makedirs("figures_v2", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# GLOBAL CONFIG  —  deliberately NOT tuned to favor any policy
# ═══════════════════════════════════════════════════════════════════
SIM_DURATION     = 30_000    # ms  (30 seconds simulated)
FIDELITY_SLA     = 0.70      # minimum acceptable fidelity
SYNC_BASE        = 5.0       # ms base sync duration
SYNC_JITTER_STD  = 3.0       # ms std dev of sync jitter (realistic network)
SYNC_JITTER_MAX  = 15.0      # ms cap on jitter

# Node decay — two realistic populations
DECAY_SENSOR     = 0.00025   # fidelity/ms  fast (IoT sensors)
DECAY_ACTUATOR   = 0.00008   # fidelity/ms  slow (actuators/controllers)

# Burst parameters — nodes randomly spike in decay rate
BURST_PROB       = 0.0008    # probability per ms of a burst event starting
BURST_DURATION   = (200, 800) # ms range of burst length
BURST_MULTIPLIER = (3.0, 8.0) # how much faster decay during burst

# Node failure
FAILURE_PROB     = 0.00003   # probability per ms of node going offline
FAILURE_DURATION = (500, 2000) # ms offline

# Experiment seeds — 10 independent runs for confidence intervals
N_SEEDS = 10
SEEDS   = [42, 137, 256, 512, 1024, 2048, 3141, 9999, 7777, 1111]

# Visual style
COLORS = {
    'static_fast':  '#e74c3c',
    'static_slow':  '#f39c12',
    'threshold':    '#3498db',
    'adaptive':     '#2ecc71',
    'priority':     '#9b59b6',
}
LABELS = {
    'static_fast':  'Static (T=100ms)',
    'static_slow':  'Static (T=500ms)',
    'threshold':    'Threshold-based',
    'adaptive':     'Adaptive Load-Aware',
    'priority':     'Priority-Weighted',
}

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi':     150,
})

# ═══════════════════════════════════════════════════════════════════
# NODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class NodeConfig:
    node_id:    int
    base_decay: float
    priority:   float = 1.0
    node_type:  str   = "sensor"

@dataclass
class MetricsSnapshot:
    time:           float
    avg_fidelity:   float
    min_fidelity:   float
    sla_violations: float   # fraction 0-1
    queue_depth:    int
    syncs_count:    int
    nodes_in_burst: int
    nodes_offline:  int

# ═══════════════════════════════════════════════════════════════════
# CORE SIMULATOR  — fair, stress-tested
# ═══════════════════════════════════════════════════════════════════
class SimFidelity:
    def __init__(self, env, nodes: List[NodeConfig],
                 channel_capacity: int,
                 policy: str, policy_params: dict,
                 sample_interval: float = 100.0,
                 seed: int = 42):

        self.env      = env
        self.nodes    = nodes
        self.n        = len(nodes)
        self.policy   = policy
        self.params   = policy_params
        self.rng      = np.random.RandomState(seed)

        self.channel  = simpy.PriorityResource(env, capacity=channel_capacity)

        # per-node state
        self.fidelity      = {nd.node_id: 1.0  for nd in nodes}
        self.last_update   = {nd.node_id: 0.0  for nd in nodes}
        self.last_sync_time= {nd.node_id: 0.0  for nd in nodes}
        self.burst_end     = {nd.node_id: -1.0 for nd in nodes}
        self.burst_mult    = {nd.node_id: 1.0  for nd in nodes}
        self.offline_until = {nd.node_id: -1.0 for nd in nodes}
        self.in_sync       = {nd.node_id: False for nd in nodes}

        self.metrics: List[MetricsSnapshot] = []
        self.sync_log: List[dict] = []
        self._syncs_window = 0

        for nd in nodes:
            env.process(self._node_process(nd))
        env.process(self._monitor())

    # ── Realistic decay with bursts ──────────────────────────────
    def _effective_decay(self, nd: NodeConfig) -> float:
        now = self.env.now
        nid = nd.node_id

        # trigger new burst?
        if now > self.burst_end[nid]:
            if self.rng.random() < BURST_PROB * self.params.get('poll_interval', 30):
                dur  = self.rng.uniform(*BURST_DURATION)
                mult = self.rng.uniform(*BURST_MULTIPLIER)
                self.burst_end[nid]  = now + dur
                self.burst_mult[nid] = mult
            else:
                self.burst_mult[nid] = 1.0

        mult = self.burst_mult[nid] if now < self.burst_end[nid] else 1.0

        # add small Gaussian noise to base decay (sensor noise)
        noise = self.rng.normal(0, nd.base_decay * 0.15)
        return max(0.0, nd.base_decay * mult + noise)

    def _update_fidelity(self, nd: NodeConfig):
        now     = self.env.now
        elapsed = now - self.last_update[nd.node_id]
        decay   = self._effective_decay(nd)
        self.fidelity[nd.node_id] = max(
            0.0, self.fidelity[nd.node_id] - decay * elapsed)
        self.last_update[nd.node_id] = now

    def _is_offline(self, nd: NodeConfig) -> bool:
        return self.env.now < self.offline_until[nd.node_id]

    # ── Policies ─────────────────────────────────────────────────
    def _should_sync(self, nd: NodeConfig) -> bool:
        if self._is_offline(nd) or self.in_sync[nd.node_id]:
            return False

        f   = self.fidelity[nd.node_id]
        nid = nd.node_id
        now = self.env.now

        if self.policy == 'static_fast':
            return (now - self.last_sync_time[nid]) >= self.params['interval']

        if self.policy == 'static_slow':
            return (now - self.last_sync_time[nid]) >= self.params['interval']

        if self.policy == 'threshold':
            return f < self.params['threshold']

        if self.policy == 'adaptive':
            # load = fraction of nodes currently below SLA
            n_stale = sum(1 for fv in self.fidelity.values()
                          if fv < FIDELITY_SLA)
            load = n_stale / self.n
            # under high load: raise threshold slightly (back off non-critical)
            # under low load:  lower threshold (be more aggressive)
            adj = self.params['threshold'] - 0.06 * load + 0.04 * (1 - load)
            adj = np.clip(adj, 0.55, 0.88)
            return f < adj

        if self.policy == 'priority':
            # high priority nodes sync earlier (lower threshold)
            adj = self.params['threshold'] - 0.12 * nd.priority \
                  + 0.08 * (1 - nd.priority)
            adj = np.clip(adj, 0.50, 0.88)
            return f < adj

        return False

    # ── Per-node process ─────────────────────────────────────────
    def _node_process(self, nd: NodeConfig):
        # stagger starts to avoid thundering herd at t=0
        yield self.env.timeout(self.rng.uniform(0, 500))

        while True:
            poll = self.params.get('poll_interval', 30.0)
            yield self.env.timeout(poll)

            # random failure?
            if not self._is_offline(nd) and \
               self.rng.random() < FAILURE_PROB * poll:
                dur = self.rng.uniform(*FAILURE_DURATION)
                self.offline_until[nd.node_id] = self.env.now + dur
                self.fidelity[nd.node_id] = 0.0  # goes dark

            if self._is_offline(nd):
                continue

            self._update_fidelity(nd)

            if self._should_sync(nd):
                self.in_sync[nd.node_id] = True
                f_before   = self.fidelity[nd.node_id]
                t_request  = self.env.now
                priority   = int(10 - nd.priority * 9)

                with self.channel.request(priority=priority) as req:
                    yield req
                    wait = self.env.now - t_request

                    # variable sync duration — network jitter
                    jitter = min(SYNC_JITTER_MAX,
                                 abs(self.rng.normal(0, SYNC_JITTER_STD)))
                    yield self.env.timeout(SYNC_BASE + jitter)

                    self.fidelity[nd.node_id]     = 1.0
                    self.last_update[nd.node_id]  = self.env.now
                    self.last_sync_time[nd.node_id] = self.env.now
                    self.burst_mult[nd.node_id]   = 1.0  # burst resets on sync
                    self.in_sync[nd.node_id]      = False
                    self._syncs_window += 1

                    self.sync_log.append({
                        'node': nd.node_id,
                        'time': self.env.now,
                        'wait': wait,
                        'f_before': f_before,
                        'jitter': jitter,
                    })

    # ── Monitor ──────────────────────────────────────────────────
    def _monitor(self):
        while True:
            yield self.env.timeout(100.0)  # sample every 100ms

            fvals   = list(self.fidelity.values())
            viols   = sum(1 for f in fvals if f < FIDELITY_SLA) / self.n
            bursting = sum(1 for nd in self.nodes
                           if self.env.now < self.burst_end[nd.node_id])
            offline  = sum(1 for nd in self.nodes
                           if self._is_offline(nd))

            self.metrics.append(MetricsSnapshot(
                time           = self.env.now,
                avg_fidelity   = float(np.mean(fvals)),
                min_fidelity   = float(np.min(fvals)),
                sla_violations = viols,
                queue_depth    = len(self.channel.queue),
                syncs_count    = self._syncs_window,
                nodes_in_burst = bursting,
                nodes_offline  = offline,
            ))
            self._syncs_window = 0

    # ── Summary ──────────────────────────────────────────────────
    def summary(self) -> dict:
        if not self.metrics:
            return {}
        af   = [m.avg_fidelity   for m in self.metrics]
        sv   = [m.sla_violations for m in self.metrics]
        qd   = [m.queue_depth    for m in self.metrics]
        waits = [r['wait'] for r in self.sync_log] or [0]
        return {
            'avg_fidelity':     float(np.mean(af)),
            'std_fidelity':     float(np.std(af)),
            'avg_sla_viol':     float(np.mean(sv) * 100),
            'max_sla_viol':     float(np.max(sv)  * 100),
            'total_syncs':      len(self.sync_log),
            'syncs_per_node':   len(self.sync_log) / self.n,
            'avg_wait_ms':      float(np.mean(waits)),
            'max_wait_ms':      float(np.max(waits)),
            'avg_queue':        float(np.mean(qd)),
        }


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def make_nodes(n, heterogeneous=False, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    nodes = []
    for i in range(n):
        if heterogeneous:
            if i < int(n * 0.4):   # 40% fast sensors
                decay = rng.uniform(DECAY_SENSOR * 0.7, DECAY_SENSOR * 1.4)
                prio  = rng.uniform(0.6, 1.0)
                ntype = "sensor"
            else:                   # 60% slower actuators
                decay = rng.uniform(DECAY_ACTUATOR * 0.7, DECAY_ACTUATOR * 1.4)
                prio  = rng.uniform(0.1, 0.6)
                ntype = "actuator"
        else:
            decay = DECAY_SENSOR
            prio  = 1.0
            ntype = "sensor"
        nodes.append(NodeConfig(i, decay, prio, ntype))
    return nodes


def run_once(n_nodes, policy, params, heterogeneous=False,
             channel_cap=10, duration=SIM_DURATION, seed=42):
    rng = np.random.RandomState(seed)
    env = simpy.Environment()
    nodes = make_nodes(n_nodes, heterogeneous, rng)
    sim = SimFidelity(env, nodes, channel_cap, policy, params,
                      seed=seed)
    env.run(until=duration)
    return sim


def run_multi_seed(n_nodes, policy, params, heterogeneous=False,
                   channel_cap=10, duration=SIM_DURATION,
                   seeds=SEEDS):
    """Run across multiple seeds, return mean±ci for key metrics."""
    summaries = []
    for s in seeds:
        sim = run_once(n_nodes, policy, params, heterogeneous,
                       channel_cap, duration, seed=s)
        summaries.append(sim.summary())

    keys = summaries[0].keys()
    out  = {}
    for k in keys:
        vals = [s[k] for s in summaries]
        out[k + '_mean'] = float(np.mean(vals))
        out[k + '_std']  = float(np.std(vals))
        # 95% CI half-width
        out[k + '_ci']   = float(stats.sem(vals) * 1.96)
    return out


def get_traces(sim):
    t  = np.array([m.time          for m in sim.metrics])
    af = np.array([m.avg_fidelity  for m in sim.metrics])
    mf = np.array([m.min_fidelity  for m in sim.metrics])
    sv = np.array([m.sla_violations * 100 for m in sim.metrics])
    qd = np.array([m.queue_depth   for m in sim.metrics])
    bu = np.array([m.nodes_in_burst for m in sim.metrics])
    of = np.array([m.nodes_offline  for m in sim.metrics])
    return t, af, mf, sv, qd, bu, of


def smooth(arr, w=12):
    if len(arr) < w:
        return arr
    return uniform_filter1d(arr.astype(float), size=w)


def ci_band(ax, x, mean, ci, color, alpha=0.18):
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=alpha)


# ═══════════════════════════════════════════════════════════════════
# POLICIES CONFIG
# ═══════════════════════════════════════════════════════════════════
BASE_PARAMS = {
    'static_fast':  {'interval': 100,  'poll_interval': 30},
    'static_slow':  {'interval': 500,  'poll_interval': 30},
    'threshold':    {'threshold': FIDELITY_SLA, 'poll_interval': 30},
    'adaptive':     {'threshold': FIDELITY_SLA, 'poll_interval': 30},
    'priority':     {'threshold': FIDELITY_SLA, 'poll_interval': 30},
}

ALL_POLICIES  = list(BASE_PARAMS.keys())
CORE_POLICIES = ['static_fast', 'threshold', 'adaptive']


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Multi-seed policy comparison, N=100
#   Homogeneous nodes, moderate channel (cap=10)
#   Result: mean ± 95% CI bars + time-series from one representative seed
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("EXP 1: Policy comparison N=100, 10 seeds, homogeneous")
print("=" * 65)

exp1_multi = {}
exp1_single = {}   # one representative run for time-series
for pol in ALL_POLICIES:
    t0 = time.time()
    exp1_multi[pol]  = run_multi_seed(100, pol, BASE_PARAMS[pol],
                                       heterogeneous=False,
                                       channel_cap=10,
                                       duration=SIM_DURATION,
                                       seeds=SEEDS)
    exp1_single[pol] = run_once(100, pol, BASE_PARAMS[pol],
                                 heterogeneous=False,
                                 channel_cap=10,
                                 duration=SIM_DURATION, seed=42)
    m = exp1_multi[pol]
    print(f"  {LABELS[pol]:28s}  "
          f"AvgF={m['avg_fidelity_mean']:.3f}±{m['avg_fidelity_ci']:.3f}  "
          f"Viol={m['avg_sla_viol_mean']:.2f}%  "
          f"Syncs/N={m['syncs_per_node_mean']:.1f}  "
          f"({time.time()-t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Scalability under bandwidth pressure
#   N = 10 → 2000, channel capacity intentionally LIMITED
#   This is where static fails and adaptive earns it
# ═══════════════════════════════════════════════════════════════════
print("\nEXP 2: Scalability N=10..2000, bandwidth-constrained, 10 seeds")

node_counts = [10, 50, 100, 200, 500, 1000, 2000]
scale = {p: {'avg_f':[], 'avg_f_ci':[], 'viol':[], 'viol_ci':[],
             'wait':[], 'wait_ci':[]}
         for p in CORE_POLICIES}

for n in node_counts:
    # Cap channel at sqrt(n) — intentional bottleneck that grows slowly
    # This creates real contention pressure at high N
    cap = max(3, int(np.sqrt(n) * 0.8))
    print(f"  N={n:5d}  channel_cap={cap}", end="  ")
    for pol in CORE_POLICIES:
        r = run_multi_seed(n, pol, BASE_PARAMS[pol],
                           heterogeneous=True,
                           channel_cap=cap,
                           duration=15_000,   # 15s for speed
                           seeds=SEEDS[:5])   # 5 seeds for scale exp
        scale[pol]['avg_f'].append(r['avg_fidelity_mean'])
        scale[pol]['avg_f_ci'].append(r['avg_fidelity_ci'])
        scale[pol]['viol'].append(r['avg_sla_viol_mean'])
        scale[pol]['viol_ci'].append(r['avg_sla_viol_ci'])
        scale[pol]['wait'].append(r['avg_wait_ms_mean'])
        scale[pol]['wait_ci'].append(r['avg_wait_ms_ci'])
    print(f"adaptive: f={scale['adaptive']['avg_f'][-1]:.3f}  "
          f"viol={scale['adaptive']['viol'][-1]:.2f}%  "
          f"wait={scale['adaptive']['wait'][-1]:.1f}ms")


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Stress test: burst events + node failures
#   N=200 heterogeneous, moderate channel
#   Show how policies handle bursty, unpredictable decay
# ═══════════════════════════════════════════════════════════════════
print("\nEXP 3: Stress test — bursts + failures, N=200, 10 seeds")

exp3 = {}
for pol in CORE_POLICIES + ['priority']:
    exp3[pol] = run_multi_seed(200, pol, BASE_PARAMS[pol],
                                heterogeneous=True,
                                channel_cap=15,
                                duration=SIM_DURATION,
                                seeds=SEEDS)
    m = exp3[pol]
    print(f"  {LABELS[pol]:28s}  "
          f"AvgF={m['avg_fidelity_mean']:.3f}±{m['avg_fidelity_ci']:.3f}  "
          f"Viol={m['avg_sla_viol_mean']:.2f}%  "
          f"MaxViol={m['max_sla_viol_mean']:.2f}%  "
          f"AvgWait={m['avg_wait_ms_mean']:.1f}ms")

# single representative run for time-series
exp3_single = {}
for pol in CORE_POLICIES + ['priority']:
    exp3_single[pol] = run_once(200, pol, BASE_PARAMS[pol],
                                 heterogeneous=True,
                                 channel_cap=15,
                                 duration=SIM_DURATION, seed=42)


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Bandwidth contention: capacity sweep N=300
#   Vary channel capacity from severely constrained to abundant
#   Shows where each policy breaks down
# ═══════════════════════════════════════════════════════════════════
print("\nEXP 4: Bandwidth contention sweep, N=300, 5 seeds")

capacities = [2, 4, 6, 8, 12, 16, 24, 32]
cap_res = {p: {'avg_f':[], 'avg_f_ci':[], 'viol':[], 'viol_ci':[],
               'wait':[], 'wait_ci':[]}
           for p in CORE_POLICIES}

for cap in capacities:
    print(f"  cap={cap:3d}", end="  ")
    for pol in CORE_POLICIES:
        r = run_multi_seed(300, pol, BASE_PARAMS[pol],
                           heterogeneous=True,
                           channel_cap=cap,
                           duration=15_000,
                           seeds=SEEDS[:5])
        cap_res[pol]['avg_f'].append(r['avg_fidelity_mean'])
        cap_res[pol]['avg_f_ci'].append(r['avg_fidelity_ci'])
        cap_res[pol]['viol'].append(r['avg_sla_viol_mean'])
        cap_res[pol]['viol_ci'].append(r['avg_sla_viol_ci'])
        cap_res[pol]['wait'].append(r['avg_wait_ms_mean'])
        cap_res[pol]['wait_ci'].append(r['avg_wait_ms_ci'])
    print(f"adaptive: f={cap_res['adaptive']['avg_f'][-1]:.3f}  "
          f"wait={cap_res['adaptive']['wait'][-1]:.1f}ms")


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Pareto frontier: threshold sweep, multi-seed
# ═══════════════════════════════════════════════════════════════════
print("\nEXP 5: Pareto frontier, N=200, 10 seeds")

thresholds  = np.linspace(0.40, 0.92, 16)
pareto_f    = []
pareto_f_ci = []
pareto_cost = []
pareto_viol = []
pareto_viol_ci = []

for thr in thresholds:
    r = run_multi_seed(200, 'threshold',
                       {'threshold': thr, 'poll_interval': 30},
                       heterogeneous=True,
                       channel_cap=12,
                       duration=15_000,
                       seeds=SEEDS)
    pareto_f.append(r['avg_fidelity_mean'])
    pareto_f_ci.append(r['avg_fidelity_ci'])
    pareto_cost.append(r['syncs_per_node_mean'])
    pareto_viol.append(r['avg_sla_viol_mean'])
    pareto_viol_ci.append(r['avg_sla_viol_ci'])
    print(f"  thr={thr:.2f}  f={r['avg_fidelity_mean']:.3f}  "
          f"syncs/N={r['syncs_per_node_mean']:.1f}  "
          f"viol={r['avg_sla_viol_mean']:.2f}%")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Policy comparison (time-series + summary bars)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

fig = plt.figure(figsize=(14, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
fig.suptitle(
    'SimFidelity — Policy Comparison\n'
    'N=100 Homogeneous Nodes, 10 Independent Seeds, 30s Simulation',
    fontsize=13, fontweight='bold')

# (a) Avg fidelity time-series, representative seed
ax = fig.add_subplot(gs[0, :2])
for pol in ALL_POLICIES:
    t, af, mf, sv, qd, bu, of = get_traces(exp1_single[pol])
    lw = 2.5 if pol in ('adaptive', 'priority') else 1.5
    ax.plot(t, smooth(af), color=COLORS[pol], label=LABELS[pol], lw=lw)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.3, label='SLA Floor (0.70)')
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('Avg Digital Twin Fidelity')
ax.set_title('(a) Fidelity Over Time — Representative Run (seed=42)')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.22)
ax.set_ylim(0.35, 1.05)

# (b) SLA violation time-series
ax = fig.add_subplot(gs[0, 2])
for pol in ALL_POLICIES:
    t, af, mf, sv, qd, bu, of = get_traces(exp1_single[pol])
    lw = 2.5 if pol in ('adaptive',) else 1.4
    ax.plot(t, smooth(sv, 18), color=COLORS[pol], label=LABELS[pol], lw=lw)
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('SLA Violation Rate (%)')
ax.set_title('(b) SLA Violations Over Time')
ax.legend(fontsize=7)
ax.grid(alpha=0.22)

# (c) Mean fidelity with 95% CI error bars
ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(ALL_POLICIES))
means = [exp1_multi[p]['avg_fidelity_mean'] for p in ALL_POLICIES]
cis   = [exp1_multi[p]['avg_fidelity_ci']   for p in ALL_POLICIES]
bars  = ax.bar(x, means, color=[COLORS[p] for p in ALL_POLICIES],
               alpha=0.82, edgecolor='black', lw=0.7)
ax.errorbar(x, means, yerr=cis, fmt='none', color='black',
            capsize=4, lw=1.5)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast','Static\nSlow','Thresh',
                    'Adaptive','Priority'], fontsize=8)
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(c) Fidelity — 10-Seed Summary')
ax.set_ylim(0.60, 1.02)
ax.grid(alpha=0.22, axis='y')

# (d) Sync cost (syncs/node) vs SLA violation — the tradeoff bar
ax  = fig.add_subplot(gs[1, 1])
ax2 = ax.twinx()
syncs = [exp1_multi[p]['syncs_per_node_mean'] for p in ALL_POLICIES]
viols = [exp1_multi[p]['avg_sla_viol_mean']   for p in ALL_POLICIES]
s_ci  = [exp1_multi[p]['syncs_per_node_ci']   for p in ALL_POLICIES]
v_ci  = [exp1_multi[p]['avg_sla_viol_ci']     for p in ALL_POLICIES]
ax.bar(x, syncs, color=[COLORS[p] for p in ALL_POLICIES],
       alpha=0.82, edgecolor='black', lw=0.7)
ax.errorbar(x, syncs, yerr=s_ci, fmt='none', color='black',
            capsize=4, lw=1.5)
ax2.errorbar(x, viols, yerr=v_ci, fmt='ko--',
             ms=6, lw=1.8, capsize=4, label='SLA Viol %')
ax2.set_ylabel('Avg SLA Violation (%)')
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast','Static\nSlow','Thresh',
                    'Adaptive','Priority'], fontsize=8)
ax.set_ylabel('Syncs per Node (Bandwidth Cost)')
ax.set_title('(d) Bandwidth Cost vs SLA — 10 Seeds')
ax2.legend(fontsize=8)
ax.grid(alpha=0.22, axis='y')

# (e) Queue depth (contention) time-series
ax = fig.add_subplot(gs[1, 2])
for pol in ['threshold', 'adaptive', 'priority']:
    t, af, mf, sv, qd, bu, of = get_traces(exp1_single[pol])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(t, smooth(qd, 20), color=COLORS[pol], label=LABELS[pol], lw=lw)
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('Channel Queue Depth')
ax.set_title('(e) Sync Channel Contention')
ax.legend(fontsize=8)
ax.grid(alpha=0.22)

plt.savefig('figures_v2/fig1_policy_comparison.pdf', bbox_inches='tight')
plt.savefig('figures_v2/fig1_policy_comparison.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig1")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Scalability with confidence intervals
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'SimFidelity — Scalability Under Bandwidth Pressure\n'
    'N = 10 to 2000 Nodes, Heterogeneous, Channel Cap = ⌊0.8√N⌋, 5 Seeds',
    fontsize=12, fontweight='bold')

nc = np.array(node_counts)

for i, (key, ci_key, ylabel, title, marker) in enumerate([
    ('avg_f',  'avg_f_ci',  'Avg Fidelity',          '(a) Fidelity vs Scale',      'o'),
    ('viol',   'viol_ci',   'Avg SLA Violation (%)',  '(b) SLA Violations vs Scale','s'),
    ('wait',   'wait_ci',   'Avg Sync Wait (ms)',     '(c) Queue Wait vs Scale',    '^'),
]):
    ax = axes[i]
    for pol in CORE_POLICIES:
        m  = np.array(scale[pol][key])
        ci = np.array(scale[pol][ci_key])
        lw = 2.5 if pol == 'adaptive' else 1.5
        ax.plot(nc, m, color=COLORS[pol], label=LABELS[pol],
                lw=lw, marker=marker, ms=5)
        ci_band(ax, nc, m, ci, COLORS[pol])
    if key == 'avg_f':
        ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2,
                   label='SLA Floor')
    ax.set_xlabel('Number of DT Nodes')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22, which='both')

plt.tight_layout()
plt.savefig('figures_v2/fig2_scalability.pdf', bbox_inches='tight')
plt.savefig('figures_v2/fig2_scalability.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig2")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Stress test: bursts + failures
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
fig.suptitle(
    'SimFidelity — Stress Test: Bursty Decay + Node Failures\n'
    'N=200 Heterogeneous Nodes, 10 Seeds, 30s Simulation',
    fontsize=12, fontweight='bold')

pols_stress = CORE_POLICIES + ['priority']

# (a) Fidelity time-series with burst events overlay
ax = fig.add_subplot(gs[0, :2])
for pol in pols_stress:
    t, af, mf, sv, qd, bu, of = get_traces(exp3_single[pol])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(t, smooth(af, 15), color=COLORS[pol], label=LABELS[pol], lw=lw)
# overlay burst activity from adaptive run
t, af, mf, sv, qd, bu, of = get_traces(exp3_single['adaptive'])
ax2 = ax.twinx()
ax2.fill_between(t, 0, bu / 200 * 100, alpha=0.12,
                 color='#f39c12', label='Nodes in Burst (%)')
ax2.set_ylabel('Nodes in Burst Event (%)', color='#b7770d')
ax2.tick_params(axis='y', colors='#b7770d')
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2, label='SLA Floor')
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('Avg Fidelity')
ax.set_title('(a) Fidelity Under Burst Events — Representative Run')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.22)
ax.set_ylim(0.30, 1.05)

# (b) SLA violation time-series
ax = fig.add_subplot(gs[0, 2])
for pol in pols_stress:
    t, af, mf, sv, qd, bu, of = get_traces(exp3_single[pol])
    lw = 2.5 if pol == 'adaptive' else 1.4
    ax.plot(t, smooth(sv, 18), color=COLORS[pol], label=LABELS[pol], lw=lw)
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('SLA Violation Rate (%)')
ax.set_title('(b) SLA Violations Under Stress')
ax.legend(fontsize=7)
ax.grid(alpha=0.22)

# (c) Offline node count
ax = fig.add_subplot(gs[1, 0])
for pol in ['adaptive', 'threshold']:
    t, af, mf, sv, qd, bu, of = get_traces(exp3_single[pol])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(t, smooth(of, 20), color=COLORS[pol], label=LABELS[pol], lw=lw)
ax.set_xlabel('Simulation Time (ms)')
ax.set_ylabel('Nodes Offline')
ax.set_title('(c) Node Failure Events Over Time')
ax.legend(fontsize=8)
ax.grid(alpha=0.22)

# (d) Mean fidelity + CI under stress — summary bars
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(pols_stress))
means = [exp3[p]['avg_fidelity_mean'] for p in pols_stress]
cis   = [exp3[p]['avg_fidelity_ci']   for p in pols_stress]
ax.bar(x, means, color=[COLORS[p] for p in pols_stress],
       alpha=0.82, edgecolor='black', lw=0.7)
ax.errorbar(x, means, yerr=cis, fmt='none',
            color='black', capsize=4, lw=1.5)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast','Thresh','Adaptive','Priority'], fontsize=8)
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(d) Stress Test — 10-Seed Summary')
ax.set_ylim(0.55, 1.02)
ax.grid(alpha=0.22, axis='y')

# (e) Max SLA violation under stress
ax = fig.add_subplot(gs[1, 2])
max_viols = [exp3[p]['max_sla_viol_mean'] for p in pols_stress]
avg_viols = [exp3[p]['avg_sla_viol_mean'] for p in pols_stress]
width = 0.35
ax.bar(x - width/2, max_viols, width, label='Max Viol %',
       color=[COLORS[p] for p in pols_stress], alpha=0.5,
       edgecolor='black', lw=0.7, hatch='//')
ax.bar(x + width/2, avg_viols, width, label='Avg Viol %',
       color=[COLORS[p] for p in pols_stress], alpha=0.85,
       edgecolor='black', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast','Thresh','Adaptive','Priority'], fontsize=8)
ax.set_ylabel('SLA Violation Rate (%)')
ax.set_title('(e) Max vs Avg Violation Under Stress')
ax.legend(fontsize=8)
ax.grid(alpha=0.22, axis='y')

plt.savefig('figures_v2/fig3_stress_test.pdf', bbox_inches='tight')
plt.savefig('figures_v2/fig3_stress_test.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig3")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — Bandwidth contention sweep
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'SimFidelity — Bandwidth Contention Sweep\n'
    'N=300 Heterogeneous Nodes, Channel Capacity 2→32, 5 Seeds',
    fontsize=12, fontweight='bold')

caps = np.array(capacities)

for i, (key, ci_key, ylabel, title, marker) in enumerate([
    ('avg_f', 'avg_f_ci', 'Avg Fidelity',         '(a) Fidelity vs Channel Cap',   'o'),
    ('viol',  'viol_ci',  'Avg SLA Violation (%)', '(b) Violations vs Channel Cap', 's'),
    ('wait',  'wait_ci',  'Avg Sync Wait (ms)',    '(c) Wait Time vs Channel Cap',  '^'),
]):
    ax = axes[i]
    for pol in CORE_POLICIES:
        m  = np.array(cap_res[pol][key])
        ci = np.array(cap_res[pol][ci_key])
        lw = 2.5 if pol == 'adaptive' else 1.5
        ax.plot(caps, m, color=COLORS[pol], label=LABELS[pol],
                lw=lw, marker=marker, ms=5)
        ci_band(ax, caps, m, ci, COLORS[pol])
    if key == 'avg_f':
        ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2,
                   label='SLA Floor')
    ax.set_xlabel('Channel Capacity (concurrent syncs)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22)

plt.tight_layout()
plt.savefig('figures_v2/fig4_bandwidth_contention.pdf', bbox_inches='tight')
plt.savefig('figures_v2/fig4_bandwidth_contention.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig4")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Pareto frontier with confidence intervals
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'SimFidelity — Fidelity–Cost Pareto Frontier\n'
    'N=200 Heterogeneous Nodes, Threshold Sweep, 10 Seeds (mean ± 95% CI)',
    fontsize=12, fontweight='bold')

pf  = np.array(pareto_f)
pfc = np.array(pareto_f_ci)
pc  = np.array(pareto_cost)
pv  = np.array(pareto_viol)
pvc = np.array(pareto_viol_ci)

ax = axes[0]
sc = ax.scatter(pc, pf, c=thresholds, cmap='RdYlGn',
                s=100, edgecolors='black', lw=0.7, zorder=4)
ax.errorbar(pc, pf, yerr=pfc, fmt='none',
            color='#333', lw=1.2, capsize=3, zorder=3)
ax.plot(pc, pf, 'k--', lw=0.8, alpha=0.35)

# mark operating point (last threshold with zero violations)
zero_viol_idx = [i for i, v in enumerate(pv) if v < 0.01]
if zero_viol_idx:
    op = zero_viol_idx[0]
    ax.annotate('Optimal\nOperating Point',
                xy=(pc[op], pf[op]),
                xytext=(pc[op] + 0.5, pf[op] - 0.04),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                fontsize=8)

ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2, label='SLA Floor')
plt.colorbar(sc, ax=ax, label='Sync Threshold')
ax.set_xlabel('Syncs per Node (Bandwidth Cost)')
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(a) Fidelity–Cost Pareto Curve')
ax.legend(fontsize=8)
ax.grid(alpha=0.22)

ax = axes[1]
sc2 = ax.scatter(pc, pv, c=thresholds, cmap='RdYlGn',
                 s=100, edgecolors='black', lw=0.7, zorder=4)
ax.errorbar(pc, pv, yerr=pvc, fmt='none',
            color='#333', lw=1.2, capsize=3, zorder=3)
ax.plot(pc, pv, 'k--', lw=0.8, alpha=0.35)
ax.axhline(0, color='#555', ls='--', lw=1.0)
plt.colorbar(sc2, ax=ax, label='Sync Threshold')
ax.set_xlabel('Syncs per Node (Bandwidth Cost)')
ax.set_ylabel('Avg SLA Violation % (mean ± 95% CI)')
ax.set_title('(b) Violation Rate vs Bandwidth Cost')
ax.grid(alpha=0.22)

plt.tight_layout()
plt.savefig('figures_v2/fig5_pareto_frontier.pdf', bbox_inches='tight')
plt.savefig('figures_v2/fig5_pareto_frontier.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig5")


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("FINAL RESULTS TABLE — EXP 1 (N=100 Homogeneous, 10 Seeds)")
print("=" * 75)
print(f"{'Policy':28s} {'AvgF':>10} {'±CI':>8} {'Viol%':>8} "
      f"{'Syncs/N':>10} {'AvgWait':>10}")
print("-" * 75)
for pol in ALL_POLICIES:
    m = exp1_multi[pol]
    print(f"{LABELS[pol]:28s} "
          f"{m['avg_fidelity_mean']:>10.3f} "
          f"{m['avg_fidelity_ci']:>8.3f} "
          f"{m['avg_sla_viol_mean']:>8.2f} "
          f"{m['syncs_per_node_mean']:>10.1f} "
          f"{m['avg_wait_ms_mean']:>10.2f}")

print("\n" + "=" * 75)
print("STRESS TEST — EXP 3 (N=200 Heterogeneous+Bursts, 10 Seeds)")
print("=" * 75)
print(f"{'Policy':28s} {'AvgF':>10} {'±CI':>8} {'Viol%':>8} "
      f"{'MaxViol%':>10} {'AvgWait':>10}")
print("-" * 75)
for pol in pols_stress:
    m = exp3[pol]
    print(f"{LABELS[pol]:28s} "
          f"{m['avg_fidelity_mean']:>10.3f} "
          f"{m['avg_fidelity_ci']:>8.3f} "
          f"{m['avg_sla_viol_mean']:>8.2f} "
          f"{m['max_sla_viol_mean']:>10.2f} "
          f"{m['avg_wait_ms_mean']:>10.2f}")

print(f"\nAll figures saved to ./figures_v2/")
print("Done.")
