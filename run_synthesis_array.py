"""SLURM/local job-array orchestrator for fine-grained (model, RF) synthesis units.

Generalizes the per-architecture *_slurm.sh + *_runner.sh script pairs in HPRC_scripts/
into one parameterized tool. Those scripts submit one srun step per ~50-model batch
file, with each step running that whole file's models through its entire RF sweep
sequentially, in-process, and no isolation between failures (the first exception in a
200-unit sequential loop kills every remaining unit in that step). This orchestrator
instead expands every (batch_file, model, RF) triple into its own unit of work up
front, groups them into SLURM-array-sized chunks, and runs each chunk's units as
isolated subprocesses (see run_one_unit.py) so one bad unit can't take its neighbors
down with it.

Does NOT touch the on-disk dataset schema (_processed.json / raw_reports/ / projects/
*.tar.gz) that batch_compress_files.py / json_dataset_merge.py / util/fixes/* depend
on -- run_one_unit.py calls the exact same run_search_iteration.run_iter() the existing
*_slurm.sh scripts do. iter_manager_v2.py's existing direct-call contract is untouched;
this is a new, additive entry point.

Modes:
    --prepare      Expand an architecture's batch-file glob x RF sweep into a joblist,
                    dropping already-complete/excluded units, and write a new run
                    directory (manifest.json + joblist.txt).
    --submit RUN_DIR
                    Compute chunk size / array size from the prepared joblist, render
                    and submit the SLURM array (--dry-run to render without sbatch-ing).
    --run-chunk RUN_DIR TASK_INDEX
                    Internal: run one array task's slice of the joblist. Invoked by the
                    array script itself (see slurm/job_array.py), not by a human.
    --status RUN_DIR
                    Report done/pending counts for a run directory without polling SLURM
                    -- the recovery path if the submitting process died mid-poll.

IMPORTANT -- before a real campaign, run the empirical pilot steps in
planning/dataset_gen_plan.md (confirm ACES's real MaxArraySize, and whether array tasks
count individually against the 40-concurrent-running-job cap) before trusting the
--array-max-tasks/--array-concurrency defaults at scale.
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

from util.completion import is_unit_complete

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_exclude_set(path):
    """Load an optional exclusion file: lines of `model_name` (exclude that model at every
    RF) or `model_name<TAB>rf` (exclude just that one RF) -- e.g. an "already published"
    list per planning/dataset_gen_plan.md #2b's dedup plan. Returns (exclude_all, exclude_pairs).
    """
    exclude_all, exclude_pairs = set(), set()
    if not path:
        return exclude_all, exclude_pairs
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                exclude_pairs.add((parts[0], int(parts[1])))
            else:
                exclude_all.add(parts[0])
    return exclude_all, exclude_pairs


def build_rf_values(rf_lower, rf_upper, rf_step):
    """Same RF enumeration iter_manager_v2.py's JSON-input branch uses: range(lower, upper,
    step) with rf=0 remapped to rf=1 ("fix to let us start at 0 to get clean steps, but
    still do rf=1") -- keeps generated joblists consistent with what the existing
    *_slurm.sh scripts' --rf_lower/--rf_upper/--rf_step values would have produced.
    """
    return [1 if rf == 0 else rf for rf in range(rf_lower, rf_upper, rf_step)]


def build_joblist(batch_files, rf_values, output_dir, exclude_all=None, exclude_pairs=None):
    """Expand (batch_file x model_name x rf) into flat job tuples, dropping units that are
    already complete (per util.completion.is_unit_complete) or explicitly excluded.

    Returns (jobs, total_before_dedup) where jobs is a list of
    (abs_batch_file_path, model_name, rf) tuples.
    """
    exclude_all = exclude_all or set()
    exclude_pairs = exclude_pairs or set()
    jobs = []
    total_before_dedup = 0
    for batch_file in batch_files:
        abs_batch_file = os.path.abspath(batch_file)
        with open(batch_file, "r") as f:
            models = json.load(f)
        for model_name in models:
            if "_rf" in model_name:
                raise ValueError(
                    f"Model name {model_name!r} in {batch_file} contains '_rf', which breaks "
                    f"downstream filename parsing (util/json_dataset_processor.py recovers the "
                    f"base model name from result filenames by splitting on '_rf')."
                )
            for rf in rf_values:
                total_before_dedup += 1
                if model_name in exclude_all or (model_name, rf) in exclude_pairs:
                    continue
                if is_unit_complete(output_dir, model_name, rf):
                    continue
                jobs.append((abs_batch_file, model_name, rf))
    return jobs, total_before_dedup


def _manifest_path(run_dir):
    return os.path.join(run_dir, "manifest.json")


def _submission_path(run_dir):
    return os.path.join(run_dir, "submission.json")


def _joblist_path(run_dir):
    return os.path.join(run_dir, "joblist.txt")


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _read_joblist(run_dir):
    jobs = []
    with open(_joblist_path(run_dir)) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            batch_file, model_name, rf = line.split("\t")
            jobs.append((batch_file, model_name, int(rf)))
    return jobs


def cmd_prepare(args):
    batch_files = sorted(glob.glob(args.batch_glob))
    if not batch_files:
        raise SystemExit(f"No batch files matched {args.batch_glob!r}")

    rf_values = build_rf_values(args.rf_lower, args.rf_upper, args.rf_step)
    logger.info(f"{len(batch_files)} batch file(s), RF values: {rf_values}")

    exclude_all, exclude_pairs = _load_exclude_set(args.exclude_file)
    jobs, total_before_dedup = build_joblist(
        batch_files, rf_values, args.output, exclude_all, exclude_pairs
    )
    logger.info(f"{total_before_dedup} total (model, RF) units; {len(jobs)} remaining "
                f"after dropping already-complete/excluded units")

    # Shuffle once with a logged seed before chunking: if joblist order correlates at all
    # with per-unit cost (plausible for cartesian/LHS-generated batches, which can be
    # systematically ordered by size), contiguous chunking would otherwise create "hot"
    # and "cold" chunks with very different wall-clock times.
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else random.randint(0, 2**31 - 1)
    random.Random(shuffle_seed).shuffle(jobs)

    if args.limit is not None and len(jobs) > args.limit:
        logger.info(f"--limit {args.limit}: capping {len(jobs)} shuffled units down to {args.limit} "
                    f"(a random subset, since the shuffle above already ran)")
        jobs = jobs[:args.limit]

    run_dir_root = args.run_dir_root or os.path.join(args.output, "_runs")
    os.makedirs(run_dir_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_dir_root, f"run_{args.arch}_{ts}")
    os.makedirs(run_dir, exist_ok=False)

    with open(_joblist_path(run_dir), "w") as f:
        for batch_file, model_name, rf in jobs:
            f.write(f"{batch_file}\t{model_name}\t{rf}\n")

    manifest = {
        "arch": args.arch,
        "batch_glob": args.batch_glob,
        "rf_lower": args.rf_lower, "rf_upper": args.rf_upper, "rf_step": args.rf_step,
        "rf_values": rf_values,
        "strat": args.strat,
        "part": args.part,
        "conv": args.conv,
        "output": os.path.abspath(args.output),
        "hlsproj": os.path.abspath(args.hlsproj),
        "total_before_dedup": total_before_dedup,
        "remaining": len(jobs),
        "shuffle_seed": shuffle_seed,
        "created": ts,
    }
    with open(_manifest_path(run_dir), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Joblist: {_joblist_path(run_dir)} ({len(jobs)} units)")
    logger.info(f"Shuffle seed: {shuffle_seed}")
    if not jobs:
        logger.info("Nothing left to do -- every unit is already complete or excluded.")
    return run_dir


def _format_hms(total_minutes):
    """Round up to whole minutes, format as H:MM:SS (or D-HH:MM:SS past 24h)."""
    total_minutes = math.ceil(total_minutes)
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:00"
    return f"{hours}:{minutes:02d}:00"


def _current_running_job_count():
    """Best-effort live squeue check; returns None if squeue isn't available/fails
    (e.g. running this off-cluster) rather than raising -- this check is a courtesy
    warning, not a hard gate.
    """
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-h"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return len([line for line in result.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_submit(run_dir, args):
    from slurm.cli import CONFIRMED_RUNNING_JOB_CAP

    # Refuse to submit a second array for a run that's already been submitted -- without
    # this, re-running --submit after e.g. a killed tmux session (which only kills the
    # *polling* loop, not the SLURM array itself) would launch a duplicate array
    # re-doing the same units in parallel with whatever's still running, wasting real
    # SUs. Reattach to the existing job with --wait instead; if you genuinely want a
    # fresh attempt, --prepare a new run (already-complete units are deduped away).
    existing_job_id_path = os.path.join(run_dir, "slurm_job_id.txt")
    if os.path.exists(existing_job_id_path) and not args.dry_run:
        with open(existing_job_id_path) as f:
            existing_job_id = f.read().strip()
        raise SystemExit(
            f"{run_dir} was already submitted as job {existing_job_id}. Refusing to "
            f"submit a second array for the same run (this would duplicate work still "
            f"in flight). If your terminal/session died mid-poll, the array itself kept "
            f"running under SLURM independently -- reattach instead:\n"
            f"  python run_synthesis_array.py --wait {run_dir}"
        )

    manifest = _read_json(_manifest_path(run_dir))
    remaining = manifest["remaining"]
    if remaining == 0:
        logger.info("Nothing to submit -- this run was prepared with 0 remaining units.")
        return

    k = args.units_per_chunk or max(1, math.ceil(remaining / args.array_max_tasks))
    n_tasks = max(1, math.ceil(remaining / k))
    p = args.units_parallel

    # Confirmed empirically (2026-08-26 pilot, see planning/dataset_gen_plan.md): SLURM
    # array tasks count individually against ACES's 40-concurrent-running-job cap, not
    # just as one job for the whole array. This is a courtesy warning, not a hard block
    # -- SLURM itself will simply queue anything past the real cap rather than error, but
    # silently starving the user's *other* concurrent work is exactly the failure mode
    # worth flagging before submitting.
    current_running = _current_running_job_count()
    if current_running is not None and current_running + args.array_concurrency > CONFIRMED_RUNNING_JOB_CAP:
        logger.warning(
            f"You currently have {current_running} job(s) running, and this array requests "
            f"%{args.array_concurrency} concurrency -- combined ({current_running + args.array_concurrency}) "
            f"exceeds ACES's confirmed {CONFIRMED_RUNNING_JOB_CAP}-concurrent-running-job cap. "
            f"SLURM will simply queue the excess rather than error, but it will also queue "
            f"behind (or starve) whatever else you have running under this account. Consider "
            f"a lower --array-concurrency."
        )

    if args.slurm_time:
        slurm_time = args.slurm_time
    else:
        waves = math.ceil(k / p)
        computed_minutes = max(15, waves * args.minutes_per_unit)
        slurm_time = _format_hms(computed_minutes)
        logger.info(f"--slurm-time not given: computed {slurm_time} from ceil(K/P)={waves} "
                    f"wave(s) x --minutes-per-unit={args.minutes_per_unit}")
        if computed_minutes > 72 * 60:
            logger.warning(
                f"Computed --slurm-time ({slurm_time}) exceeds the cpu queue's 72h ceiling "
                f"(planning/golden_rules.md #3) -- this job would be rejected or truncated. "
                f"Reduce K (more, smaller chunks) or increase P (more parallelism per chunk), "
                f"or pass --slurm-time explicitly if you've confirmed a different queue/ceiling."
            )

    submission = {
        "k": k,
        "n_tasks": n_tasks,
        "units_parallel": p,
        "array_concurrency": args.array_concurrency,
        "cpus_per_unit": args.cpus_per_unit,
        "mem_per_unit": args.mem_per_unit,
        "array_max_tasks": args.array_max_tasks,
        "slurm_time": slurm_time,
        "submitted_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(_submission_path(run_dir), "w") as f:
        json.dump(submission, f, indent=2)

    logger.info(f"Chunking: K={k} units/task -> {n_tasks} array task(s), "
                f"P={args.units_parallel} units/task in parallel, "
                f"%{args.array_concurrency} concurrent tasks")

    from slurm import job_array
    job_array.submit(run_dir, manifest, submission, args, dry_run=args.dry_run)


def cmd_run_chunk(run_dir, task_index):
    """Run one array task's K-line slice of the joblist as up to P concurrent
    run_one_unit.py subprocesses.

    Each unit runs as its own subprocess (not an in-process function call) specifically
    so a native crash or OOM-kill in one unit can't silently take the rest of the chunk
    down with it -- something iter_manager_v2.py's existing in-process per-batch-file
    loop is exposed to at coarser granularity today. Exits nonzero if any assigned unit
    is still incomplete after the chunk runs, so sacct-level chunk state reflects real
    per-chunk health, not just "the process didn't crash."
    """
    manifest = _read_json(_manifest_path(run_dir))
    submission = _read_json(_submission_path(run_dir))
    jobs = _read_joblist(run_dir)

    k = submission["k"]
    p = submission["units_parallel"]
    start = task_index * k
    end = min(start + k, len(jobs))
    chunk = jobs[start:end]
    if not chunk:
        logger.warning(f"Task {task_index}: no jobs in range [{start}, {end}) -- nothing to do.")
        return 0

    logger.info(f"Task {task_index}: {len(chunk)} unit(s) assigned, up to {p} concurrent")

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    run_one_unit = os.path.join(repo_dir, "run_one_unit.py")

    def _launch(job):
        batch_file, model_name, rf = job
        # Re-check right before launching: a chunk re-invoked after a requeue/retry
        # shouldn't redo units another attempt already finished.
        if is_unit_complete(manifest["output"], model_name, rf):
            return None
        cmd = [
            sys.executable, run_one_unit,
            "--batch-file", batch_file,
            "--model-name", model_name,
            "--rf", str(rf),
            "-o", manifest["output"],
            "--hlsproj", manifest["hlsproj"],
            "--strat", manifest["strat"],
            "--part", manifest["part"],
            "--vsynth",
        ]
        if manifest["conv"]:
            cmd.append("--conv")
        return subprocess.Popen(cmd)

    pending = list(chunk)
    running = {}  # Popen -> job tuple
    failures = []

    while pending or running:
        while pending and len(running) < p:
            job = pending.pop(0)
            proc = _launch(job)
            if proc is not None:
                running[proc] = job
        if running:
            time.sleep(2)
        for proc in list(running):
            if proc.poll() is not None:
                job = running.pop(proc)
                if proc.returncode != 0:
                    logger.error(f"Unit failed (exit {proc.returncode}): {job}")
                    failures.append(job)

    still_incomplete = [
        job for job in chunk
        if not is_unit_complete(manifest["output"], job[1], job[2])
    ]
    if still_incomplete:
        logger.error(f"Task {task_index}: {len(still_incomplete)}/{len(chunk)} unit(s) still "
                     f"incomplete after this chunk ran ({len(failures)} raised an error).")
        return 1
    logger.info(f"Task {task_index}: all {len(chunk)} unit(s) complete.")
    return 0


def cmd_status(run_dir):
    manifest = _read_json(_manifest_path(run_dir))
    jobs = _read_joblist(run_dir)
    done = sum(1 for _, name, rf in jobs if is_unit_complete(manifest["output"], name, rf))
    logger.info(f"{run_dir}: {done}/{len(jobs)} unit(s) complete "
                f"({manifest['total_before_dedup']} total before dedup)")

    job_id_path = os.path.join(run_dir, "slurm_job_id.txt")
    if os.path.exists(job_id_path):
        with open(job_id_path) as f:
            job_id = f.read().strip()
        sq = subprocess.run(["squeue", "-j", job_id, "--noheader", "-o", "%T"],
                             capture_output=True, text=True)
        states = [s.strip() for s in sq.stdout.strip().splitlines() if s.strip()]
        if states:
            logger.info(f"Slurm job {job_id} still active: {dict(Counter(states))}")
        else:
            logger.info(f"Slurm job {job_id}: no tasks currently queued/running.")


def cmd_wait(run_dir):
    """Reattach to an already-submitted run's SLURM job and poll until done, WITHOUT
    submitting anything new.

    Recovery path for when the process that called --submit (and was blocking on its
    own poll loop) died -- e.g. a tmux session getting killed by the server -- while
    the SLURM array itself kept running independently under SLURM's control. Safe to
    run at any point after submission: if the array already finished, this returns
    after one short poll with the same done/failed summary --submit would have printed.
    """
    job_id_path = os.path.join(run_dir, "slurm_job_id.txt")
    if not os.path.exists(job_id_path):
        raise SystemExit(
            f"No slurm_job_id.txt in {run_dir} -- this run was never submitted "
            f"(or only --dry-run'd). Use --submit first."
        )
    with open(job_id_path) as f:
        job_id = f.read().strip()
    logger.info(f"Reattaching to job {job_id}...")
    from slurm.job_array import poll
    poll(job_id)
    cmd_status(run_dir)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true",
                       help="Build a new run directory + joblist from a batch-file glob x RF sweep.")
    mode.add_argument("--submit", metavar="RUN_DIR",
                       help="Compute chunk/array sizing and submit (or --dry-run render) the "
                            "SLURM array for a prepared RUN_DIR.")
    mode.add_argument("--run-chunk", nargs=2, metavar=("RUN_DIR", "TASK_INDEX"),
                       help="Internal: run one array task's chunk. Invoked by the array "
                            "script itself, not by a human.")
    mode.add_argument("--status", metavar="RUN_DIR",
                       help="Report done/pending unit counts for RUN_DIR without polling SLURM.")
    mode.add_argument("--wait", metavar="RUN_DIR",
                       help="Reattach to RUN_DIR's already-submitted SLURM job and poll until "
                            "done, without submitting anything new -- recovery path if the "
                            "process that called --submit died (e.g. a killed tmux session) "
                            "while its polling loop was running. The array itself keeps going "
                            "under SLURM regardless of what happens to the submitting process.")

    parser.add_argument("--dry-run", action="store_true",
                         help="With --submit: render job_array.sh and print it without calling sbatch.")

    prepare_group = parser.add_argument_group("--prepare options")
    prepare_group.add_argument("--arch", type=str, help="Architecture label, e.g. dense_latency "
                                                          "(used in the run directory name).")
    prepare_group.add_argument("--batch-glob", type=str, help="Glob for input batch JSON files.")
    prepare_group.add_argument("--rf-lower", type=int, default=0)
    prepare_group.add_argument("--rf-upper", type=int, default=1)
    prepare_group.add_argument("--rf-step", type=int, default=1)
    prepare_group.add_argument("--strat", type=str, default="Resource")
    prepare_group.add_argument("--part", type=str, default="xcu250-figd2104-2L-e")
    prepare_group.add_argument("--conv", action="store_true")
    prepare_group.add_argument("-o", "--output", type=str,
                                help="Output directory (same convention as iter_manager_v2.py -o).")
    prepare_group.add_argument("--hlsproj", type=str, help="HLS scratch project directory.")
    prepare_group.add_argument("--run-dir-root", type=str, default=None,
                                help="Where to create the new run_<arch>_<timestamp>/ bookkeeping "
                                     "directory (default: <output>/_runs).")
    prepare_group.add_argument("--exclude-file", type=str, default=None,
                                help="Optional file of already-published units to skip: lines of "
                                     "'model_name' or 'model_name<TAB>rf'.")
    prepare_group.add_argument("--shuffle-seed", type=int, default=None,
                                help="Seed for the joblist shuffle (default: generated and logged).")
    prepare_group.add_argument("--limit", type=int, default=None,
                                help="Cap the joblist to at most N units (a random subset, taken "
                                     "after shuffling) -- for a bounded-size pilot/smoke-test run "
                                     "regardless of how the underlying batch files are sized.")

    from slurm import cli as slurm_cli
    slurm_cli.add_slurm_args(parser)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.prepare:
        missing = [name for name in ("arch", "batch_glob", "output", "hlsproj") if getattr(args, name) is None]
        if missing:
            parser.error(f"--prepare requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        cmd_prepare(args)
    elif args.submit:
        cmd_submit(args.submit, args)
    elif args.run_chunk:
        run_dir, task_index = args.run_chunk
        sys.exit(cmd_run_chunk(run_dir, int(task_index)))
    elif args.status:
        cmd_status(args.status)
    elif args.wait:
        cmd_wait(args.wait)


if __name__ == "__main__":
    main()
