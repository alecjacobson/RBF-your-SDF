"""Full-pipeline comparison of tangent-direction optimizers, judged by hull-test pruning (issue #11).

    uv run --group lipschitz demo/additional_experiments/tangent_opt/pipeline_study.py

For each input and optimizer setting, runs main_algorithm end to end and
reports fit time, coefficient magnitude, reconstruction error and hull-test
pruning at 64^3-256^3 leaves (same metrics and code as the dedup study).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lipschitz.dedup_study import one, jsonable               # noqa: E402
from lipschitz.run import RESULTS                             # noqa: E402

CONFIGS = {f'{opt} x{steps}': dict(grad_optimizer=opt, optim_steps=steps)
           for opt in ('bfgs', 'newton') for steps in (5, 20)}
CONFIGS['ascent x5'] = dict(grad_optimizer='ascent', optim_steps=5)
INPUTS = [('bunny', 30, False), ('fandisk', 30, False), ('bunny', 30, True), ('box', 24, False)]


def main():
    out_path = RESULTS / 'optimizer_pipeline_study.json'
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name, n, scatter in INPUTS:
        key = f'{name}_{n}' + ('_scatter' if scatter else '')
        for cname, kw in CONFIGS.items():
            if cname in res.get(key, {}):
                continue
            r = one(name, n, scatter, kw)
            res.setdefault(key, {})[cname] = r
            out_path.write_text(json.dumps(jsonable(res), indent=1))
            h = ' '.join(f"{k}^3:rej={v['rejected']:.2f},x{v['speedup']:.2f}" for k, v in r['hull'].items())
            print(f"{key:16s} {cname:12s} fit={r['fit_seconds']:.0f}s A med={r['sum_abs_alpha_median']:.2g} "
                  f"surf med/p99={r['abs_F_on_true_surface']['median']:.1e}/{r['abs_F_on_true_surface']['p99']:.1e} "
                  f"| {h}", flush=True)


if __name__ == '__main__':
    main()
