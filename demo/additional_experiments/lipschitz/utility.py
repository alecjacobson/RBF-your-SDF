"""Runtime utility of conservative pruning with the C++ certifier, across leaf resolutions.

    uv run --group lipschitz demo/additional_experiments/lipschitz/utility.py sphere 24 --depths 4 5 6

For one frozen reconstruction, compares the unpruned query (native field
evaluation at every leaf-lattice corner) with octree pruning by each rule in
``certify.METHODS`` using ``PUInterpolator.certify_boxes``. Reports wall
times (median of repeats after warmup), kernel-term counts, rejection rates and
validation against every sign-changing lattice edge. Run it on an otherwise
idle machine: unlike run.py it is meant for timing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'lipschitz'

from lipschitz import run as R                                     # noqa: E402
from lipschitz import pu_bounds as pub                             # noqa: E402
from lipschitz.certify import (adaptive_prune, corner_points,      # noqa: E402
                               sign_change_violations, METHODS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('grid', type=int)
    ap.add_argument('--depths', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--root', type=int, default=4)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--reg', type=float, default=None)
    ap.add_argument('--dedup-tol', type=float, default=None)
    args = ap.parse_args(argv)
    kw = {}
    if args.reg is not None:
        kw['reg'] = args.reg
    if args.dedup_tol is not None:
        kw['pu_dedup_tol'] = args.dedup_tol
    tag = f'{args.input}_{args.grid}' + ''.join(f'_{k}{v:g}' for k, v in kw.items())
    outdir = R.RESULTS / 'utility'
    outdir.mkdir(parents=True, exist_ok=True)
    snap, native, dom, _, t_fit = R.fit(args.input, args.grid, 10, kw)
    lo, hi = dom
    cert = pub.NativeCertifier(native)
    t = time.perf_counter()
    cert.certify(lo[None], hi[None])          # builds the per-patch constants and pair table
    t_pre = time.perf_counter() - t
    res = {'input': tag, 'snapshot_sha256': snap.digest(), 'fit_seconds': t_fit,
           'preprocess_seconds': t_pre, 'threads': os.environ.get('OMP_NUM_THREADS', 'default'),
           'cpu_count': os.cpu_count(), 'levels': {}}
    rng = np.random.default_rng(0)
    for depth in args.depths:
        leaf_n = np.array([args.root * 2 ** depth] * 3)
        allpts, _ = corner_points(lo, hi, leaf_n)
        t_full = R.timed(lambda: native.predict(allpts), reps=args.reps)
        ref = native.predict(allpts)
        lev = {'leaf_n': int(leaf_n[0]), 'unpruned_seconds': t_full,
               'unpruned_evaluations': int(len(allpts)), 'unpruned_kernel_terms': R.kernel_terms(snap, allpts)}
        for method in METHODS:
            def go():
                pr = adaptive_prune(cert, lo, hi, args.root, depth, method)
                pts, _ = corner_points(lo, hi, leaf_n, pr.leaf_label != 1)
                native.predict(pts)
                return pr, pts
            go()                                                    # warmup
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter(); pr, pts = go(); ts.append(time.perf_counter() - t)
            rejected = pr.leaf_label == 1
            viol, n_sc = sign_change_violations(ref, leaf_n, rejected)
            lev[method] = {'total_seconds': float(np.median(ts)), 'certify_seconds_last': pr.time_certify,
                           'speedup': t_full / float(np.median(ts)),
                           'rejected_leaf_fraction': float(rejected.mean()),
                           'unresolved_leaf_fraction': float((pr.leaf_label == 2).mean()),
                           'field_evaluations': int(len(pts)), 'boxes_certified': pr.n_boxes_certified,
                           'bound_kernel_terms': int(pr.cost_bound_terms),
                           'eval_kernel_terms': R.kernel_terms(snap, pts),
                           'sign_change_edges': n_sc, 'violations_sign_edges': viol}
            print(f'[{tag}] {leaf_n[0]}^3 {method}: {lev[method]["speedup"]:.2f}x, '
                  f'rejected {lev[method]["rejected_leaf_fraction"]:.3f}, violations {viol}', flush=True)
        res['levels'][str(int(leaf_n[0]))] = lev
        (outdir / f'{tag}.json').write_text(json.dumps(R.jsonable(res), indent=1))
    return res


if __name__ == '__main__':
    main()
