# -*- coding: utf-8 -*-
"""
Design-space coverage metrics for wa-hls4ml FPGA generation batches.

Two complementary uses:

1. `python compute_coverage_metrics.py --manifest <prefix>_lhs_manifest.json`
   Computes fill/separation distance directly from a gen_models.py --lhs run's
   manifest (see gen_models_documentation.md) -- the primary way to check whether
   sampling N designs via --lhs actually covers that config's design space better
   than N independent random draws would, and whether it's worth extending further.

2. `python compute_coverage_metrics.py` (no arguments)
   Reports fixed axis coverage for the corpora already checked into this repo.
   Full (size + bitwidth + RF) coverage is only reported for dense_latency_fast and
   dense_latency_fast_small, whose generation config is checked in
   (config_dense_latency_fast.json / config_dense_latency_fast_small.json). The other
   three checked-in corpora (dense_latency, dense_resource, conv1d, conv2d) only get
   RF-axis coverage, verified directly from their *_slurm.sh --rf_lower/--rf_upper/
   --rf_step values -- their original generation-time size/bitwidth config isn't
   checked into this repo, and gen_models.py's get_default_config() is missing keys
   (min_layer_count/max_layer_count/max_bit_width_po2) the CLI generation path
   requires, so it almost certainly isn't what was actually used. Don't guess at
   those axes; fill VERIFIED_GRIDS in below once the real config is confirmed.

Core metrics (fill distance / separation distance, linear and log2, plus a normalized
multi-dimensional L-inf fill distance) are ported unchanged from the wa-hls4ml
Catapult-ASIC branch's coverage tool -- pure numpy, no backend-specific assumptions.
"""

import argparse
import json

import numpy as np

# ── Core metrics (generic, backend/architecture-agnostic) ───────────────────────────


def fill_distance_1d(samples, lo, hi, n_eval=10_000):
    """Max distance from any point in [lo,hi] to its nearest sample."""
    samples = np.asarray(samples, dtype=float)
    xs = np.linspace(lo, hi, n_eval)
    dists = np.min(np.abs(xs[:, None] - samples[None, :]), axis=1)
    return float(dists.max())


def sep_distance_1d(samples):
    """Min distance between any two distinct sample points."""
    s = np.sort(np.asarray(samples, dtype=float))
    if len(s) < 2:
        return float("nan")
    return float(np.min(np.diff(s)))


def metrics_1d(samples, lo, hi):
    """Return (fill_lin, sep_lin, fill_log2, sep_log2) for one axis."""
    samples = np.asarray(samples, dtype=float)
    fill_lin = fill_distance_1d(samples, lo, hi)
    sep_lin = sep_distance_1d(samples)
    if lo > 0:
        slog = np.log2(samples)
        fill_log2 = fill_distance_1d(slog, np.log2(lo), np.log2(hi))
        sep_log2 = sep_distance_1d(slog)
    else:
        fill_log2 = sep_log2 = float("nan")
    return fill_lin, sep_lin, fill_log2, sep_log2


def normalize(samples, lo, hi, log2=False):
    samples = np.asarray(samples, dtype=float)
    if hi <= lo:
        return np.zeros_like(samples)
    if log2:
        return (np.log2(samples) - np.log2(lo)) / (np.log2(hi) - np.log2(lo))
    return (samples - lo) / (hi - lo)


def multidim_fill_distance(grids_norm, n_eval=100):
    """L-inf fill distance: for a grid of eval points in the unit hypercube, the max
    over eval points of the min distance to the nearest actual sample point.

    grids_norm: dict axis -> normalized 1D array, one value per generated design (not
    per discrete choice) -- i.e. all axes must have the same length M.
    Returns (fill_dist, bottleneck_axis).
    """
    axes = list(grids_norm.keys())
    samples = np.stack([np.asarray(grids_norm[a], dtype=float) for a in axes], axis=1)  # (M, D)

    pts = np.meshgrid(*[np.linspace(0, 1, n_eval) for _ in axes], indexing="ij")
    eval_pts = np.stack([p.ravel() for p in pts], axis=1)  # (N, D)

    diffs = np.abs(eval_pts[:, None, :] - samples[None, :, :])  # (N, M, D)
    linf_to_samples = diffs.max(axis=2)
    min_dist_per_eval = linf_to_samples.min(axis=1)

    worst_idx = int(np.argmax(min_dist_per_eval))
    worst_pt = eval_pts[worst_idx]
    diffs_worst = np.abs(worst_pt[None, :] - samples)
    nearest_idx = np.argmin(diffs_worst.max(axis=1))
    per_axis_gap = np.abs(worst_pt - samples[nearest_idx])
    bottleneck = axes[int(np.argmax(per_axis_gap))]

    return float(min_dist_per_eval.max()), bottleneck


# ── Mode 1: report against a gen_models.py --lhs manifest ───────────────────────────


def report_from_manifest(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("mode") != "lhs":
        raise ValueError(
            f"{manifest_path} is not a gen_models.py --lhs manifest "
            f"(expected mode='lhs', got {manifest.get('mode')!r})"
        )

    print()
    print("=" * 70)
    print(f" Coverage report: {manifest_path}")
    print(f" seed={manifest['seed']}  n_samples={manifest['n_samples']}  "
          f"joint_cardinality={manifest['joint_cardinality']}")
    print("=" * 70)

    # The manifest records each axis's configured *discrete choice set*, not the
    # per-model realized values (gen_models.py doesn't persist those separately from
    # the generated batch JSON) -- this reports how evenly the choice grid itself is
    # spaced, which bounds how good coverage of that axis could possibly be. For an
    # exact per-realized-model report, pull the actual sampled sizes/bitwidths out of
    # the batch JSON files this manifest sits alongside and feed them to metrics_1d /
    # multidim_fill_distance directly instead of the choice sets below.
    print()
    print("[ Per-axis choice-set spacing (linear and log2) ]")
    for axis_name, axis in manifest["axes"].items():
        choices = axis["choices"]
        lo, hi = min(choices), max(choices)
        fl, sl, flog, slog = metrics_1d(choices, lo, hi)
        print(f"  {axis_name:<16} choices={choices}")
        print(f"    fill(lin)={fl:.3f}  sep(lin)={sl:.3f}  "
              f"fill(log2)={flog:.3f}  sep(log2)={slog:.3f}")

    n_samples = manifest["n_samples"]
    joint_cardinality = manifest["joint_cardinality"]
    coverage_frac = n_samples / joint_cardinality if joint_cardinality else float("nan")
    print()
    print(f"  Sampling {n_samples} of {joint_cardinality} possible designs "
          f"({coverage_frac:.1%} of the joint discrete space).")
    if coverage_frac > 0.5:
        print("  >50% of the joint space -- --cartesian would enumerate it exhaustively "
              "for about the same generation cost.")
    print()


# ── Mode 2: fixed report for the checked-in corpora ──────────────────────────────────


def _dense_sizes(lb, ub):
    return [2**i for i in range(0, 20) if lb <= 2**i <= ub]


# Verified directly from the checked-in config files (config_dense_latency_fast.json,
# config_dense_latency_fast_small.json: dense_lb/dense_ub, max_bit_width_po2) and their
# matching *_slurm.sh RF sweeps (dense_latency_fast_slurm.sh: --rf_lower 1 --rf_upper 2
# --rf_step 1).
VERIFIED_GRIDS = {
    "dense_latency_fast": {
        "sizes": _dense_sizes(16, 256),
        "bitwidths": [2**i for i in range(2, 4 + 1)],  # max_bit_width_po2: 4
        "rf": [1],
    },
    "dense_latency_fast_small": {
        "sizes": _dense_sizes(16, 64),
        "bitwidths": [2**i for i in range(2, 4 + 1)],
        "rf": [1],
    },
}

# RF axis only, verified directly from each architecture's HPRC_scripts/*_slurm.sh
# --rf_lower/--rf_upper/--rf_step (reproducing iter_manager_v2.py's JSON-mode
# range(lower, upper, step) with rf=0 remapped to rf=1). Size/bitwidth axes are
# deliberately NOT included: see module docstring.
RF_ONLY_GRIDS = {
    "dense_latency":  {"rf": [1, 32, 64, 96]},              # dense_latency_slurm.sh
    "dense_resource": {"rf": [1, 512, 1024, 1536]},         # dense_resource_slurm.sh
    "conv1d":         {"rf": [8192, 16384, 24576, 32768]},  # conv1d_slurm.sh
    "conv2d":         {"rf": [1024, 2048, 3072, 4096]},     # conv2d_slurm.sh
}


def report_fixed_corpora():
    print()
    print("=" * 70)
    print(" Coverage report: checked-in wa-hls4ml FPGA corpora")
    print("=" * 70)

    print()
    print("[ Verified full axis coverage (from checked-in config files) ]")
    for arch, grids in VERIFIED_GRIDS.items():
        print(f"\n  {arch}:")
        for axis_key, axis_label, do_log in (
            ("sizes", "sizes", True), ("bitwidths", "bitwidths", False), ("rf", "RF", False),
        ):
            s = grids[axis_key]
            lo, hi = min(s), max(s)
            fl, sl, flog, slog = metrics_1d(s, lo, hi)
            log_str = f"  fill(log2)={flog:.3f} sep(log2)={slog:.3f}" if do_log else ""
            print(f"    {axis_label:<10} {{{','.join(str(v) for v in s)}}}  "
                  f"fill(lin)={fl:.2f} sep(lin)={sl:.2f}{log_str}")

    print()
    print("[ RF-axis-only coverage (size/bitwidth axes unverified -- see module docstring) ]")
    for arch, grids in RF_ONLY_GRIDS.items():
        s = grids["rf"]
        lo, hi = min(s), max(s)
        fl, sl, flog, slog = metrics_1d(s, lo, hi)
        print(f"  {arch:<16} RF={{{','.join(str(v) for v in s)}}}  "
              f"fill(lin)={fl:.1f} sep(lin)={sl:.1f} fill(log2)={flog:.3f} sep(log2)={slog:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to a gen_models.py --lhs run's <prefix>_lhs_manifest.json. If "
             "omitted, reports fixed coverage for the checked-in corpora instead.",
    )
    args = parser.parse_args()

    if args.manifest:
        report_from_manifest(args.manifest)
    else:
        report_fixed_corpora()


if __name__ == "__main__":
    main()
