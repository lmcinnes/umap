"""Which parameter settings make up the map.

The design has three parts:

  1. every combination of the ``GRID`` parameters, everything else default;
  2. from every grid point, each ``VARIED`` parameter set to each of its
     other levels, on its own;
  3. random draws over all the parameters, in which each varied parameter
     stays at its default with probability ``DEFAULT_PROBABILITY`` (or as
     given in ``DEFAULT_PROBABILITIES``), so that most settings differ from
     the defaults in only a few ways.

Part 1 comes first, and parts 2 and 3 follow in one fixed shuffle, so the
first N settings of the design are a representative sample of the whole:
a smaller map uses a prefix of the design that a larger map extends.

Changing any of the levels or rules below changes the design, and so which
runs are needed; runs already made are reused wherever settings coincide.
"""

import hashlib
import itertools
import json

import numpy as np

GRID = {
    "n_neighbors": [5, 15, 45],
    "min_dist": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
}
# name: (default, levels). init_scale_factor is used with the default
# init_rescale="isotropic".
VARIED = {
    "spread": (1.0, [0.5, 0.75, 1.0, 2.0, 4.0]),
    "repulsion_strength": (1.0, [0.1, 0.33, 1.0, 3.0, 10.0, 30.0]),
    "init_scale_factor": (1.0, [0.1, 0.5, 1.0, 1.5, 2.0]),
    "negative_selection_range": (200_000, [3_000, 10_000, 200_000]),
    "set_op_mix_ratio": (1.0, [0.01, 0.1, 1.0]),
    "densmap": (False, [False, True]),
}
# The size of the whole design, which every map's settings are a prefix of.
FULL_SIZE = 16384
DEFAULT_PROBABILITY = 0.5
# densMAP is a side option: about a tenth of the map uses it.
DEFAULT_PROBABILITIES = {"densmap": 0.9}
# Relative frequency of each non-default level in the random draws (equal
# if not given). The lowest set_op_mix_ratio often gives a halo of stray
# points, so it is drawn a third as often as the middle value.
WEIGHTS = {"set_op_mix_ratio": {0.01: 1, 0.1: 3}}
# The largest share of the design that may use a value. Draws must be
# distinct, so in a large design the commonest combinations run out and
# rarer values would otherwise take more than their intended share.
CAPS = {("densmap", True): 0.10, ("set_op_mix_ratio", 0.01): 0.125}
DESIGN_SEED = 0

# Every run uses the newer layout path and one fixed seed.
BASE_PARAMS = {"compatibility_layout": False, "n_components": 2}
RUN_SEED = 42
# negative_selection_range is given for Fashion-MNIST's 70,000 points; other
# datasets use the same fraction of their points. Values at or above this
# size mean "no limit" and are left alone.
REFERENCE_SIZE = 70_000


def valid(params):
    """Whether a setting is included in the design.

    min_dist may not exceed spread. A set_op_mix_ratio of 0.01 keeps too few
    edges when n_neighbors is 5, and densMAP mostly fails unless
    set_op_mix_ratio is at its default: either way the plot collapses to a
    small core inside a wide halo of disconnected points.
    """
    if params["min_dist"] > params.get("spread", 1.0):
        return False
    mix = params.get("set_op_mix_ratio", 1.0)
    if mix == 0.01 and params["n_neighbors"] == 5:
        return False
    return not (params.get("densmap", False) and mix != 1.0)


def levels():
    """Every parameter's levels, grid parameters first."""
    return {**GRID, **{name: levels for name, (_, levels) in VARIED.items()}}


def defaults():
    return {name: default for name, (default, _) in VARIED.items()}


def full_params(params):
    """A setting with every varied parameter it leaves unset at its default."""
    return {**defaults(), **params}


def _canonical(params):
    """Drop varied parameters at their default, so equal settings share a key."""
    return {
        name: value for name, value in params.items() if name in GRID or value != VARIED[name][0]
    }


def settings(size=FULL_SIZE):
    """The first ``size`` settings of the design, as parameter dicts that
    leave varied parameters at their defaults unset."""
    caps = {key: share * FULL_SIZE for key, share in CAPS.items()}
    used = dict.fromkeys(caps, 0)
    grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    one_at_a_time = [
        {**point, name: value}
        for point in grid
        for name, (default, levels) in VARIED.items()
        for value in levels
        if value != default
    ]
    rng = np.random.default_rng(DESIGN_SEED)
    seen = {tuple(sorted(p.items())) for p in grid + one_at_a_time}
    one_at_a_time = [p for p in one_at_a_time if valid(p)]
    draws = []
    while len(draws) < FULL_SIZE - len(grid) - len(one_at_a_time):
        params = {name: levels[rng.integers(len(levels))] for name, levels in GRID.items()}
        for name, (default, levels) in VARIED.items():
            if rng.random() >= DEFAULT_PROBABILITIES.get(name, DEFAULT_PROBABILITY):
                others = [value for value in levels if value != default]
                if name in WEIGHTS:
                    p = np.array([WEIGHTS[name][value] for value in others], dtype=float)
                    params[name] = others[rng.choice(len(others), p=p / p.sum())]
                else:
                    params[name] = others[rng.integers(len(others))]
        key = tuple(sorted(params.items()))
        capped = [k for k in caps if params.get(k[0]) == k[1]]
        if valid(params) and key not in seen and all(used[k] < caps[k] for k in capped):
            seen.add(key)
            draws.append(params)
            for k in capped:
                used[k] += 1
    rest = one_at_a_time + draws
    rest = [rest[i] for i in rng.permutation(len(rest))]
    return [_canonical(p) for p in (grid + rest)[:size]]


def run_key(params, seed=RUN_SEED):
    """A stable identifier for a run: a setting and a seed."""
    canonical = json.dumps({"params": params, "seed": seed}, sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def as_used(name, value, n_points):
    """A design value as passed to UMAP on a dataset of ``n_points`` points."""
    if name == "negative_selection_range" and value < REFERENCE_SIZE:
        return round(value * n_points / REFERENCE_SIZE)
    return value
