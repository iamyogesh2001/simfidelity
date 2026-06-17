"""
SimFidelity — Figure Regeneration Script
Produces exactly 3 publication-ready PNG figures from saved simulation data.
Run this on your Mac mini in the same folder as simfidelity_v2.py

If you don't have the saved simulation data, this script re-runs the
required experiments inline (takes ~15 min on M4).

Output: fig_A.png, fig_B.png, fig_C.png  (PNG only, no PDF)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import stats
import warnings, time, sys, os
warnings.filterwarnings('ignore')

# ── Try to import SimFidelity. If not found, tell user. ──────────────
try:
    from simfidelity_v2 import (run_once, run_multi_seed, BASE_PARAMS,
                                 SEEDS, FIDELITY_SLA, get_traces)
except ImportError:
    print("ERROR: simfidelity_v2.py not found in this folder.")
    print("Place this script in the same folder as simfidelity_v2.py")
    sys.exit(1)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.titlesize':  10,
    'axes.labelsize':  9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid':       True,
    'grid.alpha':      0.25,
    'lines.linewidth': 1.8,
})

COLORS = {
    'static_fast': '#e74c3c',
    'static_slow': '#f39c12',
    'threshold':   '#3498db',
    'adaptive':    '#2ecc71',
    'priority':    '#9b59b6',
}
LABELS = {
    'static_fast': 'Static (T=100ms)',
    'static_slow': 'Static (T=500ms)',
    'threshold':   'Threshold',
    'adaptive':    'Adaptive (Ours)',
    'priority':    'Priority-Weighted',
}

def smooth(arr, w=12):
    return uniform_filter1d(arr.astype(float), size=w)

def ci_band(ax, x, mean, ci, color, alpha=0.15):
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=alpha)

print("=" * 60)
print("Running simulations for 3-figure set...")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# DATA: E1 — Policy comparison N=100 homogeneous, 10 seeds
# ═══════════════════════════════════════════════════════════════
print("\n[1/4] Policy comparison (N=100, 10 seeds)...")
ALL_POLS = ['static_fast', 'static_slow', 'threshold', 'adaptive', 'priority']
exp1_multi  = {}
exp1_single = {}
for pol in ALL_POLS:
    exp1_multi[pol]  = run_multi_seed(100, pol, BASE_PARAMS[pol],
                                      heterogeneous=False,
                                      channel_cap=10,
                                      duration=30_000,
                                      seeds=SEEDS)
    exp1_single[pol] = run_once(100, pol, BASE_PARAMS[pol],
                                heterogeneous=False,
                                channel_cap=10,
                                duration=30_000, seed=42)
    print(f"  {LABELS[pol]:28s}  "
          f"AvgF={exp1_multi[pol]['avg_fidelity_mean']:.3f}  "
          f"Viol={exp1_multi[pol]['avg_sla_viol_mean']:.1f}%  "
          f"Syncs/N={exp1_multi[pol]['syncs_per_node_mean']:.1f}")

# ═══════════════════════════════════════════════════════════════
# DATA: E2 — Scalability N=10..2000
# ═══════════════════════════════════════════════════════════════
print("\n[2/4] Scalability sweep (N=10..2000)...")
node_counts = [10, 50, 100, 200, 500, 1000, 2000]
scale = {p: {'avg_f': [], 'avg_f_ci': [], 'wait': [], 'wait_ci': []}
         for p in ['static_fast', 'threshold', 'adaptive']}
for n in node_counts:
    cap = max(3, int(np.sqrt(n) * 0.8))
    for pol in ['static_fast', 'threshold', 'adaptive']:
        r = run_multi_seed(n, pol, BASE_PARAMS[pol],
                           heterogeneous=True,
                           channel_cap=cap,
                           duration=15_000,
                           seeds=SEEDS[:5])
        scale[pol]['avg_f'].append(r['avg_fidelity_mean'])
        scale[pol]['avg_f_ci'].append(r['avg_fidelity_ci'])
        scale[pol]['wait'].append(r['avg_wait_ms_mean'])
        scale[pol]['wait_ci'].append(r['avg_wait_ms_ci'])
    print(f"  N={n:5d}  adaptive wait={scale['adaptive']['wait'][-1]:.1f}ms  "
          f"static wait={scale['static_fast']['wait'][-1]:.1f}ms")

# ═══════════════════════════════════════════════════════════════
# DATA: E3 — Stress test N=200 heterogeneous
# ═══════════════════════════════════════════════════════════════
print("\n[3/4] Stress test (N=200, bursts+failures, 10 seeds)...")
STRESS_POLS = ['static_fast', 'threshold', 'adaptive', 'priority']
exp3_multi  = {}
exp3_single = {}
for pol in STRESS_POLS:
    exp3_multi[pol]  = run_multi_seed(200, pol, BASE_PARAMS[pol],
                                      heterogeneous=True,
                                      channel_cap=15,
                                      duration=30_000,
                                      seeds=SEEDS)
    exp3_single[pol] = run_once(200, pol, BASE_PARAMS[pol],
                                heterogeneous=True,
                                channel_cap=15,
                                duration=30_000, seed=42)
    print(f"  {LABELS[pol]:28s}  "
          f"AvgF={exp3_multi[pol]['avg_fidelity_mean']:.3f}  "
          f"MaxViol={exp3_multi[pol]['max_sla_viol_mean']:.1f}%")

# ═══════════════════════════════════════════════════════════════
# DATA: E4 — Channel capacity sweep + E5 Pareto
# ═══════════════════════════════════════════════════════════════
print("\n[4/4] Bandwidth sweep + Pareto frontier...")
capacities = [2, 4, 6, 8, 12, 16, 24, 32]
cap_res = {p: {'viol': [], 'viol_ci': [], 'wait': [], 'wait_ci': []}
           for p in ['static_fast', 'threshold', 'adaptive']}
for cap in capacities:
    for pol in ['static_fast', 'threshold', 'adaptive']:
        r = run_multi_seed(300, pol, BASE_PARAMS[pol],
                           heterogeneous=True, channel_cap=cap,
                           duration=15_000, seeds=SEEDS[:5])
        cap_res[pol]['viol'].append(r['avg_sla_viol_mean'])
        cap_res[pol]['viol_ci'].append(r['avg_sla_viol_ci'])
        cap_res[pol]['wait'].append(r['avg_wait_ms_mean'])
        cap_res[pol]['wait_ci'].append(r['avg_wait_ms_ci'])

thresholds  = np.linspace(0.40, 0.92, 16)
pareto_f, pareto_f_ci, pareto_cost, pareto_viol, pareto_viol_ci = [], [], [], [], []
for thr in thresholds:
    r = run_multi_seed(200, 'threshold',
                       {'threshold': thr, 'poll_interval': 30},
                       heterogeneous=True, channel_cap=12,
                       duration=15_000, seeds=SEEDS)
    pareto_f.append(r['avg_fidelity_mean'])
    pareto_f_ci.append(r['avg_fidelity_ci'])
    pareto_cost.append(r['syncs_per_node_mean'])
    pareto_viol.append(r['avg_sla_viol_mean'])
    pareto_viol_ci.append(r['avg_sla_viol_ci'])

print("  Pareto done.")

# ═══════════════════════════════════════════════════════════════
# FIGURE A — Policy Comparison: 3 subplots
# Kept: (a) fidelity over time, (b) 10-seed fidelity bars, (c) cost vs SLA
# Cut:  SLA time-series (noisy/redundant), queue depth (flat zero)
# ═══════════════════════════════════════════════════════════════
print("\nGenerating Figure A...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle('SimFidelity — Policy Comparison  (N=100, Homogeneous, 10 Seeds)',
             fontsize=11, fontweight='bold', y=1.01)

# (a) Fidelity over time
ax = axes[0]
for pol in ALL_POLS:
    t, af, mf, sv, qd, bu, of_ = get_traces(exp1_single[pol])
    lw = 2.5 if pol == 'adaptive' else 1.4
    ax.plot(t / 1000, smooth(af), color=COLORS[pol],
            label=LABELS[pol], lw=lw)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2, label='SLA Floor')
ax.set_xlabel('Simulation Time (s)')
ax.set_ylabel('Avg DT Fidelity')
ax.set_title('(a) Fidelity Over Time')
ax.legend(fontsize=7, loc='lower right')
ax.set_ylim(0.35, 1.05)

# (b) 10-seed fidelity bar chart with CI
ax = axes[1]
x = np.arange(len(ALL_POLS))
means = [exp1_multi[p]['avg_fidelity_mean'] for p in ALL_POLS]
cis   = [exp1_multi[p]['avg_fidelity_ci']   for p in ALL_POLS]
ax.bar(x, means, color=[COLORS[p] for p in ALL_POLS],
       alpha=0.85, edgecolor='black', lw=0.7)
ax.errorbar(x, means, yerr=cis, fmt='none',
            color='black', capsize=4, lw=1.5)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast', 'Static\nSlow', 'Thresh',
                    'Adaptive', 'Priority'], fontsize=8)
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(b) Fidelity — 10-Seed Summary')
ax.set_ylim(0.60, 1.02)

# (c) Bandwidth cost vs SLA violation
ax   = axes[2]
ax2  = ax.twinx()
syncs = [exp1_multi[p]['syncs_per_node_mean'] for p in ALL_POLS]
viols = [exp1_multi[p]['avg_sla_viol_mean']   for p in ALL_POLS]
s_ci  = [exp1_multi[p]['syncs_per_node_ci']   for p in ALL_POLS]
v_ci  = [exp1_multi[p]['avg_sla_viol_ci']     for p in ALL_POLS]
ax.bar(x, syncs, color=[COLORS[p] for p in ALL_POLS],
       alpha=0.85, edgecolor='black', lw=0.7)
ax.errorbar(x, syncs, yerr=s_ci, fmt='none',
            color='black', capsize=4, lw=1.5)
ax2.errorbar(x, viols, yerr=v_ci, fmt='ko--',
             ms=6, lw=1.8, capsize=4, label='SLA Viol %')
ax2.set_ylabel('Avg SLA Violation (%)')
ax.set_xticks(x)
ax.set_xticklabels(['Static\nFast', 'Static\nSlow', 'Thresh',
                    'Adaptive', 'Priority'], fontsize=8)
ax.set_ylabel('Syncs per Node (Bandwidth)')
ax.set_title('(c) Bandwidth Cost vs SLA Violation')
ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('fig_A.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig_A.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE B — Scalability + Stress: 4 subplots in 2x2
# Kept: scale fidelity, scale wait time, stress 10-seed bars, stress max/avg viol
# Cut:  scale SLA violations (overlaps), stress fidelity time (overlaps fig A),
#       stress node failure time (not policy-relevant)
# ═══════════════════════════════════════════════════════════════
print("Generating Figure B...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('SimFidelity — Scalability & Stress Test Results',
             fontsize=11, fontweight='bold')

nc = np.array(node_counts)

# (a) Fidelity vs scale
ax = axes[0, 0]
for pol in ['static_fast', 'threshold', 'adaptive']:
    m  = np.array(scale[pol]['avg_f'])
    ci = np.array(scale[pol]['avg_f_ci'])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(nc, m, color=COLORS[pol], label=LABELS[pol],
            lw=lw, marker='o', ms=4)
    ci_band(ax, nc, m, ci, COLORS[pol])
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2, label='SLA Floor')
ax.set_xscale('log')
ax.set_xlabel('Number of DT Nodes')
ax.set_ylabel('Avg Fidelity')
ax.set_title('(a) Fidelity vs Scale (E2)')
ax.legend(fontsize=7)

# (b) Wait time vs scale — the key scalability story
ax = axes[0, 1]
for pol in ['static_fast', 'threshold', 'adaptive']:
    m  = np.array(scale[pol]['wait'])
    ci = np.array(scale[pol]['wait_ci'])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(nc, m, color=COLORS[pol], label=LABELS[pol],
            lw=lw, marker='^', ms=4)
    ci_band(ax, nc, m, ci, COLORS[pol])
ax.set_xscale('log')
ax.set_xlabel('Number of DT Nodes')
ax.set_ylabel('Avg Sync Wait Time (ms)')
ax.set_title('(b) Queue Wait vs Scale (E2) — Key Result')
ax.legend(fontsize=7)

# (c) Stress: 10-seed fidelity summary bars
ax = axes[1, 0]
x4 = np.arange(len(STRESS_POLS))
means4 = [exp3_multi[p]['avg_fidelity_mean'] for p in STRESS_POLS]
cis4   = [exp3_multi[p]['avg_fidelity_ci']   for p in STRESS_POLS]
ax.bar(x4, means4, color=[COLORS[p] for p in STRESS_POLS],
       alpha=0.85, edgecolor='black', lw=0.7)
ax.errorbar(x4, means4, yerr=cis4, fmt='none',
            color='black', capsize=4, lw=1.5)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2)
ax.set_xticks(x4)
ax.set_xticklabels(['Static\nFast', 'Thresh', 'Adaptive', 'Priority'],
                   fontsize=8)
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(c) Stress Test: Fidelity Summary (E3)')
ax.set_ylim(0.60, 1.02)

# (d) Stress: max vs avg SLA violation
ax = axes[1, 1]
width = 0.35
max_v = [exp3_multi[p]['max_sla_viol_mean'] for p in STRESS_POLS]
avg_v = [exp3_multi[p]['avg_sla_viol_mean'] for p in STRESS_POLS]
ax.bar(x4 - width/2, max_v, width,
       color=[COLORS[p] for p in STRESS_POLS],
       alpha=0.45, edgecolor='black', lw=0.7, hatch='//', label='Max Viol %')
ax.bar(x4 + width/2, avg_v, width,
       color=[COLORS[p] for p in STRESS_POLS],
       alpha=0.85, edgecolor='black', lw=0.7, label='Avg Viol %')
ax.set_xticks(x4)
ax.set_xticklabels(['Static\nFast', 'Thresh', 'Adaptive', 'Priority'],
                   fontsize=8)
ax.set_ylabel('SLA Violation Rate (%)')
ax.set_title('(d) Stress Test: Peak vs Avg Violations (E3)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig_B.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig_B.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE C — Bandwidth + Pareto: 4 subplots in 2x2
# Kept: violations vs cap, wait vs cap, pareto fidelity, pareto violations
# Cut:  fidelity vs cap (flat lines, not interesting)
# ═══════════════════════════════════════════════════════════════
print("Generating Figure C...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('SimFidelity — Bandwidth Contention & Fidelity-Cost Trade-off',
             fontsize=11, fontweight='bold')

caps = np.array(capacities)
pf   = np.array(pareto_f)
pfc  = np.array(pareto_f_ci)
pc   = np.array(pareto_cost)
pv   = np.array(pareto_viol)
pvc  = np.array(pareto_viol_ci)

# (a) SLA violations vs channel capacity
ax = axes[0, 0]
for pol in ['static_fast', 'threshold', 'adaptive']:
    m  = np.array(cap_res[pol]['viol'])
    ci = np.array(cap_res[pol]['viol_ci'])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(caps, m, color=COLORS[pol], label=LABELS[pol],
            lw=lw, marker='s', ms=4)
    ci_band(ax, caps, m, ci, COLORS[pol])
ax.set_xlabel('Channel Capacity (concurrent syncs)')
ax.set_ylabel('Avg SLA Violation (%)')
ax.set_title('(a) Violations vs Channel Capacity (E4)')
ax.legend(fontsize=7)

# (b) Wait time vs channel capacity — the key bandwidth story
ax = axes[0, 1]
for pol in ['static_fast', 'threshold', 'adaptive']:
    m  = np.array(cap_res[pol]['wait'])
    ci = np.array(cap_res[pol]['wait_ci'])
    lw = 2.5 if pol == 'adaptive' else 1.5
    ax.plot(caps, m, color=COLORS[pol], label=LABELS[pol],
            lw=lw, marker='^', ms=4)
    ci_band(ax, caps, m, ci, COLORS[pol])
ax.set_xlabel('Channel Capacity (concurrent syncs)')
ax.set_ylabel('Avg Sync Wait Time (ms)')
ax.set_title('(b) Wait Time vs Channel Capacity (E4)')
ax.legend(fontsize=7)

# (c) Pareto: fidelity vs bandwidth cost
ax = axes[1, 0]
sc = ax.scatter(pc, pf, c=thresholds, cmap='RdYlGn',
                s=80, edgecolors='black', lw=0.6, zorder=4)
ax.errorbar(pc, pf, yerr=pfc, fmt='none',
            color='#333', lw=1.0, capsize=2.5, zorder=3)
ax.plot(pc, pf, 'k--', lw=0.7, alpha=0.35)
# mark the knee
knee_idx = np.argmin(np.abs(np.array(pareto_viol) - 5.0))
ax.annotate('Operating\nKnee', xy=(pc[knee_idx], pf[knee_idx]),
            xytext=(pc[knee_idx] + 3, pf[knee_idx] - 0.04),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            fontsize=7)
ax.axhline(FIDELITY_SLA, color='#555', ls='--', lw=1.2, label='SLA Floor')
plt.colorbar(sc, ax=ax, label='Sync Threshold', shrink=0.85)
ax.set_xlabel('Syncs per Node (Bandwidth Cost)')
ax.set_ylabel('Avg Fidelity (mean ± 95% CI)')
ax.set_title('(c) Fidelity-Cost Pareto Curve (E5)')
ax.legend(fontsize=7)

# (d) Pareto: violations vs bandwidth cost
ax = axes[1, 1]
sc2 = ax.scatter(pc, pv, c=thresholds, cmap='RdYlGn',
                 s=80, edgecolors='black', lw=0.6, zorder=4)
ax.errorbar(pc, pv, yerr=pvc, fmt='none',
            color='#333', lw=1.0, capsize=2.5, zorder=3)
ax.plot(pc, pv, 'k--', lw=0.7, alpha=0.35)
ax.axhline(0, color='#555', ls='--', lw=1.0)
plt.colorbar(sc2, ax=ax, label='Sync Threshold', shrink=0.85)
ax.set_xlabel('Syncs per Node (Bandwidth Cost)')
ax.set_ylabel('Avg SLA Violation % (mean ± 95% CI)')
ax.set_title('(d) Violation Rate vs Bandwidth Cost (E5)')

plt.tight_layout()
plt.savefig('fig_C.png', dpi=180, bbox_inches='tight')
plt.close()
print("  Saved fig_C.png")

print("\n" + "=" * 60)
print("DONE. Three figures saved:")
print("  fig_A.png — Policy comparison (3 subplots)")
print("  fig_B.png — Scalability + Stress test (4 subplots)")
print("  fig_C.png — Bandwidth contention + Pareto (4 subplots)")
print("\nUpload these to Overleaf, replacing fig1-5.")
print("Update the paper's \\includegraphics to use fig_A, fig_B, fig_C.")
print("=" * 60)
