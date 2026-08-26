"""Argparse group for the SLURM job-array submission flags used by run_synthesis_array.py."""


def add_slurm_args(parser):
    """Add --slurm-*/--array-*/--units-* flags to an argparse parser.

    Defaults match this repo's existing HPRC_scripts/*_slurm.sh settings where there's a
    direct equivalent (account, per-unit cpus/mem); the array-shape defaults (max-tasks,
    concurrency, units-parallel) are conservative placeholders pending the empirical
    pilot described in planning/dataset_gen_plan.md -- do not raise them for a real
    campaign without running that pilot first.
    """
    parser.add_argument('--slurm-account', type=str, default='157537460776',
                         help='ACES project account to bill (default: 157537460776, matching '
                              'the existing HPRC_scripts/*_slurm.sh scripts)')
    parser.add_argument('--slurm-time', type=str, default='60:00:00',
                         help="Walltime per array task, HH:MM:SS or D-HH:MM:SS (default: "
                              "60:00:00 -- intentionally under the cpu queue's 72h ceiling, "
                              "which planning/golden_rules.md #3 flags as 'zero margin' today)")
    parser.add_argument('--slurm-partition', type=str, default=None,
                         help='Slurm partition/queue (default: unset, lands on ACES\' default '
                              '"cpu" queue, matching the existing scripts)')
    parser.add_argument('--array-max-tasks', type=int, default=500,
                         help='Upper bound on the number of array tasks to create -- chunk size '
                              'K is computed as ceil(remaining_units / array_max_tasks). '
                              'Conservative default pending confirmation of ACES\'s real '
                              'MaxArraySize (`scontrol show config | grep -i MaxArraySize`).')
    parser.add_argument('--array-concurrency', type=int, default=8,
                         help='Max array tasks running at once (the --array=0-N%%CONCURRENCY '
                              'throttle). Conservative default pending the empirical check of '
                              'whether ACES\'s 40-concurrent-running-job cap applies per array '
                              'task (see planning/dataset_gen_plan.md) -- do not raise this for '
                              'a real campaign without running that check first.')
    parser.add_argument('--units-per-chunk', '-K', type=int, default=None,
                         help='Override the auto-computed chunk size (units per array task).')
    parser.add_argument('--units-parallel', '-P', type=int, default=4,
                         help='Synthesis units to run concurrently within one array task/chunk '
                              '(default: 4, a conservative pilot value -- planning docs suggest '
                              'scaling to 15 once a pilot confirms real per-unit memory usage, '
                              'matching the existing 32GB-per-unit / 488GB-per-node packing math).')
    parser.add_argument('--cpus-per-unit', type=int, default=2,
                         help='CPUs reserved per synthesis unit (default: 2, matching the '
                              'existing HPRC_scripts/*_slurm.sh --cpus-per-task).')
    parser.add_argument('--mem-per-unit', type=str, default='32G',
                         help='Memory reserved per synthesis unit, e.g. "32G" (default: 32G, '
                              'matching the existing scripts\' --mem-per-cpu=16384 x 2 cpus).')
