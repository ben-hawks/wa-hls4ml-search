"""Argparse group for the SLURM job-array submission flags used by run_synthesis_array.py."""


# Confirmed on ACES 2026-08-26 via HPRC_scripts/pilot/ (see planning/dataset_gen_plan.md
# addendum for the full readout):
#   - MaxArraySize = 1001 (scontrol show config)
#   - SLURM array tasks DO count individually against the 40-concurrent-running-job cap:
#     a 60-task array at %60 was observed pinned at exactly 40 concurrently-RUNNING tasks
#     the entire time, stepping down through 20 as tasks finished -- not a guess anymore.
CONFIRMED_MAX_ARRAY_SIZE = 1001
CONFIRMED_RUNNING_JOB_CAP = 40


def add_slurm_args(parser):
    """Add --slurm-*/--array-*/--units-* flags to an argparse parser.

    Defaults match this repo's existing HPRC_scripts/*_slurm.sh settings where there's a
    direct equivalent (account, per-unit cpus/mem). --array-max-tasks and
    --array-concurrency are now set from the confirmed pilot numbers above, with margin.
    --minutes-per-unit's default is calibrated from a single pilot against
    dense_latency_fast_small -- the smallest/fastest-synthesizing corpus in this repo --
    and should not be assumed to hold for conv1d/conv2d/dense_latency/dense_resource
    without their own pilot first.
    """
    parser.add_argument('--slurm-account', type=str, default='157537460776',
                         help='ACES project account to bill (default: 157537460776, matching '
                              'the existing HPRC_scripts/*_slurm.sh scripts)')
    parser.add_argument('--slurm-time', type=str, default=None,
                         help="Walltime per array task, HH:MM:SS or D-HH:MM:SS. Default: "
                              "computed as ceil(K/P) * --minutes-per-unit, capped with a "
                              "warning at the cpu queue's 72h ceiling -- a flat default here "
                              "regardless of chunk size caused a real 60x walltime-vs-runtime "
                              "mismatch on a small pilot chunk (golden_rules.md #4 flags "
                              "exactly this as a reason ACES will kill a job). Set explicitly "
                              "to override the computed value.")
    parser.add_argument('--minutes-per-unit', type=float, default=60.0,
                         help='Assumed worst-case wall-clock minutes for one synthesis unit, '
                              'used only to compute --slurm-time when it is not given '
                              'explicitly. Default: 60 -- roughly 2x the worst per-wave time '
                              'observed in the 2026-08-26 dense_latency_fast_small pilot '
                              '(~30 min/wave at K=5,P=4). Recalibrate per-architecture before '
                              'a real run against conv1d/conv2d/dense_latency/dense_resource, '
                              'which are expected to synthesize slower than this pilot\'s corpus.')
    parser.add_argument('--slurm-partition', type=str, default=None,
                         help='Slurm partition/queue (default: unset, lands on ACES\' default '
                              '"cpu" queue, matching the existing scripts)')
    parser.add_argument('--array-max-tasks', type=int, default=800,
                         help='Upper bound on the number of array tasks to create -- chunk size '
                              f'K is computed as ceil(remaining_units / array_max_tasks). '
                              f'Default 800: confirmed real MaxArraySize is '
                              f'{CONFIRMED_MAX_ARRAY_SIZE}, this leaves ~20% margin.')
    parser.add_argument('--array-concurrency', type=int, default=30,
                         help='Max array tasks running at once (the --array=0-N%%CONCURRENCY '
                              f'throttle). CONFIRMED empirically that SLURM array tasks count '
                              f'individually against ACES\'s {CONFIRMED_RUNNING_JOB_CAP}-'
                              f'concurrent-running-job cap (not a guess -- see '
                              f'planning/dataset_gen_plan.md). Default 30 leaves headroom '
                              f'under that hard ceiling for other work under this account; '
                              f'run_synthesis_array.py warns (using a live squeue check) if '
                              f'this run\'s concurrency plus what\'s already running would '
                              f'exceed the cap.')
    parser.add_argument('--units-per-chunk', '-K', type=int, default=None,
                         help='Override the auto-computed chunk size (units per array task).')
    parser.add_argument('--units-parallel', '-P', type=int, default=4,
                         help='Synthesis units to run concurrently within one array task/chunk '
                              '(default: 4 -- the 2026-08-26 pilot ran this successfully '
                              'against dense_latency_fast_small at 34-56%% memory efficiency '
                              '(review headroom), but that\'s the smallest/fastest corpus in '
                              'this repo; don\'t raise this for other architectures without '
                              'piloting them specifically -- larger conv/dense models may have '
                              'a very different memory footprint per unit).')
    parser.add_argument('--cpus-per-unit', type=int, default=2,
                         help='CPUs reserved per synthesis unit (default: 2, matching the '
                              'existing HPRC_scripts/*_slurm.sh --cpus-per-task).')
    parser.add_argument('--mem-per-unit', type=str, default='32G',
                         help='Memory reserved per synthesis unit, e.g. "32G" (default: 32G, '
                              'matching the existing scripts\' --mem-per-cpu=16384 x 2 cpus).')
