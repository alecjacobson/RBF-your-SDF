"""Experiment inputs: exact analytic SDFs and the repository's mesh sampling driver.

Analytic inputs are sampled on an ``n^3`` grid spanning ``[-0.5, 0.5]^3``
(endpoints included). Mesh inputs follow ``demo/SDF_to_surface_3D.py``:
normalize the mesh to the unit cube and sample its exact unsigned distance
with a sign on a grid over the bounding box padded by 0.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[2]
REPO = DEMO.parent


def sdf_sphere(x, r=0.3):
    return np.linalg.norm(x, axis=1) - r


def sdf_box(x, b=(0.3, 0.22, 0.15)):
    q = np.abs(x) - np.asarray(b)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


def sdf_torus(x, R=0.28, r=0.1):
    q = np.stack([np.linalg.norm(x[:, [0, 2]], axis=1) - R, x[:, 1]], 1)
    return np.linalg.norm(q, axis=1) - r


def sdf_shell(x, r=0.3, t=0.02):
    """Thin spherical shell of half-thickness t (a thin-feature case)."""
    return np.abs(np.linalg.norm(x, axis=1) - r) - t


ANALYTIC = {'sphere': sdf_sphere, 'box': sdf_box, 'torus': sdf_torus, 'shell': sdf_shell}


def grid_points(n, lo=-0.5, hi=0.5):
    a = np.linspace(lo, hi, n)
    X, Y, Z = np.meshgrid(a, a, a, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def make_input(name, n, seed=1):
    """Return (points, values, description dict) for an input name and grid size."""
    if name in ANALYTIC:
        pts = grid_points(n)
        vals = ANALYTIC[name](pts)
        keep = np.abs(vals) > 1e-8          # same rule as the demo driver
        return pts[keep], vals[keep], {'kind': 'analytic', 'name': name, 'grid': n,
                                       'domain': [-0.5, 0.5], 'exact_sdf': True}
    mesh_path = REPO / 'examples' / f'{name}.obj'
    if not mesh_path.exists():
        raise ValueError(f"unknown input {name!r}")
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    import SDF_to_surface_3D as sdf3d
    sdf3d.seed = seed
    _, pts, vals, _ = sdf3d.generate_test_mesh_data(str(mesh_path), name, grid_len=n)
    return pts, vals, {'kind': 'mesh', 'name': name, 'grid': n, 'path': str(mesh_path.relative_to(REPO)),
                       'exact_sdf': True, 'sampling': 'demo/SDF_to_surface_3D.generate_test_mesh_data'}
