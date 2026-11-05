"""Collect results/lipschitz/*/results.json into tables and figures.

    uv run --group lipschitz demo/additional_experiments/lipschitz/report.py [--out DIR]

Writes ``summary.json``, ``report_tables.md`` and PNG figures. Every number
comes from a saved run; nothing is plotted for runs that do not exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'lipschitz'

# Categorical slots in fixed order (reference palette, light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STATUS_COLORS = {'rejected': '#1baf7a', 'retained': '#d9d8d3', 'unresolved': '#e34948'}


def load_runs(root=RESULTS):
    runs = {}
    for p in sorted(root.glob('*/results.json')):
        runs[p.parent.name] = json.loads(p.read_text())
    return runs


def fnum(x, d=3):
    if isinstance(x, str):
        return x
    if x is None:
        return '-'
    if x == 0:
        return '0'
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f'{x:.{d - 1}e}'
    return f'{x:.{d}g}'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def tables(runs):
    out = []
    out.append('### Runs\n')
    out.append('| run | patches | fit s | centers/patch (median) | sum abs alpha (median) | max abs m0 | interp. err max |')
    out.append('|---|---|---|---|---|---|---|')
    for k, r in runs.items():
        p = r['patches']
        out.append(f"| {k} | {r['manifest']['num_patches']} | {fnum(r['manifest']['fit_seconds'])} | "
                   f"{fnum(p['n_centers']['median'])} | {fnum(p['A_sum_abs_alpha']['median'])} | "
                   f"{fnum(p['m0']['max'])} | {fnum(r['interpolation']['abs_error_at_samples']['max'])} |")
    out.append('\n### Reference vs native evaluator (max abs differences, normalized frame)\n')
    out.append('| run | value (all strata) | gradient (all but support boundary) | gradient support boundary | FD rel. err h=1e-5 |')
    out.append('|---|---|---|---|---|')
    for k, r in runs.items():
        v = r.get('validation')
        if not v:
            continue
        strata = [s for s in v if isinstance(v[s], dict) and 'n' in v[s] and v[s]['n']]
        vmax = max(v[s]['value_absdiff_max'] for s in strata)
        gmax = max(v[s]['grad_absdiff_max'] for s in strata if s != 'support_boundary')
        gsb = v.get('support_boundary', {}).get('grad_absdiff_max')
        out.append(f"| {k} | {fnum(vmax)} | {fnum(gmax)} | {fnum(gsb)} | "
                   f"{fnum(v['central_difference_relerr_median']['1e-05'])} |")
    out.append('\n### Local-fit bounds: certified bound / observed max |grad f_j| (median [p90])\n')
    for k, r in runs.items():
        lb = r.get('local_bounds')
        if not lb:
            continue
        out.append(f'\n**{k}**\n')
        out.append('| cells/axis | rho | B0 | B1 ideal | B2 | B3 | min(B0,B2,B3) | B1 violations |')
        out.append('|---|---|---|---|---|---|---|---|')
        for g, v in lb.items():
            rr = v['ratio']
            cell = lambda q: f"{fnum(q['median'])} [{fnum(q['p90'])}]"
            out.append(f"| {g} | {fnum(v['rho'])} | {cell(rr['B0'])} | {cell(rr['B1'])} | {cell(rr['B2'])} | "
                       f"{cell(rr['B3'])} | {cell(v['ratio_min_cert'])} | {v['B1_violations']} |")
    out.append('\n### Full-field certificates on regular cell grids\n')
    out.append('| run | grid | cells | certified | fallback fixed | unresolved | L_best/obs med [p90, p99, max] | L_B0/obs med | local-only/obs med (incomplete) | blend share med | violations | certify s |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, r in runs.items():
        for g, v in r.get('grids', {}).items():
            sc = v['status_counts']; n = v['cells']
            b = v['ratio']['L_best']
            out.append(f"| {k} | {g} | {n} | {fnum(sc['certified'] / n)} | {fnum(sc['fallback_fixed'] / n)} | "
                       f"{fnum(sc['unresolved'] / n)} | {fnum(b['median'])} [{fnum(b['p90'])}, {fnum(b['p99'])}, {fnum(b['max'])}] | "
                       f"{fnum(v['ratio']['L_B0']['median'])} | {fnum(v['ratio']['L_local_only']['median'])} | "
                       f"{fnum(v['blend_share_best']['median'])} | {sum(v['violations'].values())} | "
                       f"{fnum(v['certify_seconds'])} |")
    out.append('\n### Transitions\n')
    out.append('| run | fallback volume fraction | witnesses (kind: count, max limiting jump) |')
    out.append('|---|---|---|')
    for k, r in runs.items():
        t = r.get('transitions')
        if not t:
            continue
        kinds = {}
        for w in t['witnesses']:
            c, m = kinds.get(w['kind'], (0, 0.0))
            kinds[w['kind']] = (c + 1, max(m, w['limiting_jump_estimate']))
        desc = '; '.join(f'{kk}: {c}, {fnum(m)}' for kk, (c, m) in kinds.items()) or 'none found'
        out.append(f"| {k} | {fnum(t['fallback_volume_fraction'])} | {desc} |")
    out.append('\n### Conservative cell rejection (same octree, same leaf resolution)\n')
    out.append('| run | leaves | method | rejected | unresolved | field evals saved | bound kernel terms | certify s | eval s | total s | unpruned s | violations (edges / probes) |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, r in runs.items():
        p = r.get('prune')
        if not p:
            continue
        for m in ('B0', 'best', 'hull', 'best_hull'):
            v = p[m]
            out.append(f"| {k} | {p['leaf_n'][0]}^3 | {m} | {fnum(v['rejected_leaf_fraction'])} | "
                       f"{fnum(v['unresolved_leaf_fraction'])} | {fnum(v['evaluations_saved_fraction'])} | "
                       f"{fnum(v['bound_kernel_terms'])} | {fnum(v['certify_seconds'])} | {fnum(v['eval_seconds'])} | "
                       f"{fnum(v['total_seconds'])} | {fnum(p['baseline']['seconds'])} | "
                       f"{v['violations_sign_edges']} / {v['violations_interior_probes']} |")
    return '\n'.join(out) + '\n'


def fig_local(runs, out):
    import matplotlib.pyplot as plt
    names = [k for k in runs if 'local_bounds' in runs[k]]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], names):
        lb = runs[k]['local_bounds']
        rho = np.array([v['rho'] for v in lb.values()])
        for i, m in enumerate(('B0', 'B2', 'B3')):
            med = [v['ratio'][m]['median'] for v in lb.values()]
            p90 = [v['ratio'][m]['p90'] for v in lb.values()]
            ax.plot(rho, med, color=SERIES[i], lw=2, marker='o', ms=4, label=m)
            ax.fill_between(rho, med, p90, color=SERIES[i], alpha=0.12, lw=0)
            ax.annotate(m, (rho[-1], med[-1]), color=INK2, fontsize=8, xytext=(-14, 4),
                        textcoords='offset points')
        ax.axhline(1, color=INK2, lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(k, fontsize=9, color=INK)
        ax.set_xlabel('cell half-diagonal ρ (normalized units)', fontsize=8, color=INK2)
        style(ax)
    axes[0][0].set_ylabel('certified bound / observed max |∇f_j|', fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle('Local-fit bounds versus cell size (median, band to p90)', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'local_bounds_vs_cell_size.png', dpi=150)
    plt.close(fig)


def fig_gap_cdf(runs, out, grid='64'):
    import matplotlib.pyplot as plt
    names = [k for k in runs if grid in runs[k].get('grids', {})]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], names):
        z = np.load(RESULTS / k / f'grid_{grid}.npz')
        ok = (z['status'] != 2) & (z['obs'] > 0)
        for i, (key, lab) in enumerate((('L_best', 'best valid bound'), ('L_simple', 'simple disagreement'),
                                        ('L_B0', 'B0 through blend'))):
            r = np.sort(z[key][ok] / z['obs'][ok])
            ax.plot(r, np.linspace(0, 1, len(r)), color=SERIES[i], lw=2, label=lab)
        r = np.sort(z['L_local_only'][ok] / z['obs'][ok])
        ax.plot(r, np.linspace(0, 1, len(r)), color=INK2, lw=1, ls='--', label='local only (incomplete)')
        ax.axvline(1, color=INK2, lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_title(f'{k}, {grid}³ cells', fontsize=9, color=INK)
        ax.set_xlabel('upper bound / observed lower bound', fontsize=8, color=INK2)
        style(ax)
    axes[0][0].set_ylabel('fraction of resolved cells', fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.suptitle('Certificate gap distribution (resolved cells)', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / f'gap_cdf_{grid}.png', dpi=150)
    plt.close(fig)


def fig_blend(runs, out, grid='32'):
    import matplotlib.pyplot as plt
    names = [k for k in runs if grid in runs[k].get('grids', {})]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], names):
        z = np.load(RESULTS / k / f'grid_{grid}.npz')
        ok = z['status'] == 0
        ax.scatter(z['W_min'][ok], z['blend_best'][ok], s=3, color=SERIES[0], alpha=0.35, lw=0,
                   label='blend term')
        ax.scatter(z['W_min'][ok], z['L_local_only'][ok], s=3, color=SERIES[1], alpha=0.35, lw=0,
                   label='max local bound')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{k}, {grid}³ cells', fontsize=9, color=INK)
        ax.set_xlabel('certified W_min on cell', fontsize=8, color=INK2)
        style(ax)
    axes[0][0].set_ylabel('contribution to L_best', fontsize=8, color=INK2)
    axes[0][0].legend(fontsize=7, frameon=False, markerscale=4)
    fig.suptitle('Blend contribution versus certified denominator', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / f'blend_vs_denominator_{grid}.png', dpi=150)
    plt.close(fig)


def fig_prune(runs, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    names = [k for k in runs if 'prune' in runs[k]]
    methods = ('B0', 'best', 'hull')
    if not names:
        return
    cmap = ListedColormap([STATUS_COLORS['retained'], STATUS_COLORS['rejected'], STATUS_COLORS['unresolved']])
    fig, axes = plt.subplots(len(names), len(methods), figsize=(2.6 * len(methods), 2.6 * len(names)),
                             squeeze=False)
    for row, k in zip(axes, names):
        for ax, m in zip(row, methods):
            p = RESULTS / k / f'prune_labels_{m}.npy'
            if not p.exists():
                ax.axis('off'); continue
            lab = np.load(p)
            ax.imshow(lab[:, :, lab.shape[2] // 2].T, origin='lower', cmap=cmap, vmin=0, vmax=2,
                      interpolation='nearest')
            frac = runs[k]['prune'][m]['rejected_leaf_fraction']
            ax.set_title(f'{k}: {m} ({frac:.0%} rejected)', fontsize=8, color=INK)
            ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=STATUS_COLORS[s], label=s) for s in ('rejected', 'retained', 'unresolved')],
               loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Mid-z slice of leaf cells', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'prune_slices.png', dpi=150)
    plt.close(fig)


def fig_witness(runs, out):
    import matplotlib.pyplot as plt
    ws = [(k, w) for k, r in runs.items() for w in r.get('transitions', {}).get('witnesses', [])]
    if not ws:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    kinds = sorted({w['kind'] for _, w in ws})
    for _, w in ws:
        i = kinds.index(w['kind'])
        ax.plot(w['separation'], np.maximum(w['value_jump'], 1e-17), color=SERIES[i], lw=1, alpha=0.6,
                marker='o', ms=2)
    s = np.logspace(-9, -2, 10)
    ax.plot(s, s * 1.0, color=INK2, lw=0.8, ls=':', label='slope 1 (Lipschitz with L=1)')
    for i, kd in enumerate(kinds):
        ax.plot([], [], color=SERIES[i], lw=2, label=kd.replace('_', ' '))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('probe pair separation', fontsize=8, color=INK2)
    ax.set_ylabel('|F(a) − F(b)|', fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('Two-sided probes at evaluator transitions', fontsize=9, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / 'transition_probes.png', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=RESULTS / 'report')
    ap.add_argument('--runs', nargs='*', default=None)
    a = ap.parse_args()
    runs = load_runs()
    if a.runs:
        runs = {k: v for k, v in runs.items() if k in a.runs}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'report_tables.md').write_text(tables(runs))
    fig_local(runs, a.out)
    for g in ('32', '64'):
        fig_gap_cdf(runs, a.out, g)
    fig_blend(runs, a.out)
    fig_prune(runs, a.out)
    fig_witness(runs, a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
