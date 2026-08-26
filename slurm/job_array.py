"""SLURM job-array rendering/submission for the fine-grained per-(model,RF) synthesis
orchestrator (run_synthesis_array.py).

Two entry points:
    write_array_script(...) -- render {run_dir}/job_array.sh from a prepared run's
                                manifest + resolved submission sizing.
    submit(...)             -- write the script, sbatch it (unless dry_run), persist the
                                job id, and poll squeue/sacct until done.

Each array task re-invokes run_synthesis_array.py --run-chunk <run_dir> <task_index>,
which reads manifest.json + submission.json from run_dir itself rather than being
passed sizing parameters on the array script's command line -- keeps the per-task
command trivial and avoids any risk of the array script's baked-in parameters drifting
from what's actually recorded for the run.
"""

import logging
import os
import subprocess
import sys
import time
from collections import Counter

logger = logging.getLogger(__name__)


def _split_mem(mem_str):
    """'32G' -> (32, 'G')."""
    digits = ''.join(c for c in mem_str if c.isdigit())
    unit = ''.join(c for c in mem_str if c.isalpha()) or 'G'
    return int(digits), unit


def write_array_script(run_dir, manifest, submission, args):
    """Write {run_dir}/job_array.sh. Returns its path.

    Args:
        run_dir: the prepared run directory (contains joblist.txt, manifest.json).
        manifest: dict from run_synthesis_array.cmd_prepare (arch, output, hlsproj, ...).
        submission: dict with the resolved array shape (k, n_tasks, units_parallel, ...).
        args: argparse Namespace with slurm_account/slurm_time/slurm_partition.
    """
    n_tasks = submission['n_tasks']
    p = submission['units_parallel']
    total_cpus = p * submission['cpus_per_unit']
    mem_val, mem_unit = _split_mem(submission['mem_per_unit'])
    total_mem = f"{mem_val * p}{mem_unit}"

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_synthesis_array = os.path.join(repo_dir, "run_synthesis_array.py")

    slurm_logs_dir = os.path.join(run_dir, "slurm_logs")
    os.makedirs(slurm_logs_dir, exist_ok=True)

    partition_line = f"#SBATCH --partition={args.slurm_partition}\n" if args.slurm_partition else ""

    script_content = f"""#!/bin/bash
#SBATCH --job-name=wa_hls4ml_{manifest['arch']}
#SBATCH --account={args.slurm_account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={total_cpus}
#SBATCH --mem={total_mem}
#SBATCH --time={args.slurm_time}
{partition_line}#SBATCH --array=0-{n_tasks - 1}%{submission['array_concurrency']}
#SBATCH --output={slurm_logs_dir}/task_%a.out
#SBATCH --error={slurm_logs_dir}/task_%a.err

set -uo pipefail

python "{run_synthesis_array}" --run-chunk "{run_dir}" "$SLURM_ARRAY_TASK_ID"
"""
    script_path = os.path.join(run_dir, "job_array.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    logger.info(f"Wrote SLURM array script: {script_path}")
    return script_path


def submit(run_dir, manifest, submission, args, dry_run=False):
    """Write job_array.sh; if not dry_run, sbatch it, persist the job id, and poll to
    completion. Returns the job id, or None if dry_run.
    """
    script_path = write_array_script(run_dir, manifest, submission, args)

    if dry_run:
        with open(script_path) as f:
            print(f.read())
        logger.info(
            f"[dry-run] Would submit {submission['n_tasks']} array task(s) "
            f"(K={submission['k']} units/task, %{submission['array_concurrency']} "
            f"concurrency). Not calling sbatch."
        )
        return None

    logger.info(f"Submitting SLURM array ({submission['n_tasks']} tasks, "
                f"max {submission['array_concurrency']} concurrent)...")
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"sbatch failed:\n{result.stderr}")
        sys.exit(result.returncode)

    job_id = result.stdout.strip().split()[-1]
    logger.info(f"Submitted SLURM array job: {job_id}")

    jobid_path = os.path.join(run_dir, "slurm_job_id.txt")
    with open(jobid_path, "w") as f:
        f.write(job_id + "\n")
    logger.info(f"Job ID saved: {jobid_path}")

    poll(job_id)
    return job_id


def poll(job_id, interval=30):
    """Poll squeue every `interval` seconds until every task in job_id has finished,
    then summarize any failures via a single sacct call.

    If this process dies mid-poll, the array tasks keep running independently under
    SLURM -- re-run with --status <run_dir> (or `squeue -j <job_id>` directly) to check
    on them later; nothing here is required for the array itself to finish.
    """
    logger.info(f"Polling squeue for job {job_id} every {interval}s until complete...")
    while True:
        time.sleep(interval)
        sq = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True, text=True,
        )
        states = [s.strip() for s in sq.stdout.strip().splitlines() if s.strip()]
        if not states:
            logger.info("All array tasks have finished.")
            break
        logger.info(f"  tasks: {dict(Counter(states))}")

    sacct = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "--noheader", "-P"],
        capture_output=True, text=True,
    )
    failed = []
    for line in sacct.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            task_id, state, exit_code = parts[0], parts[1], parts[2]
            if state == "FAILED" or (exit_code != "0:0" and "batch" not in task_id and "." not in task_id):
                failed.append((task_id, state, exit_code))

    if failed:
        logger.warning(f"{len(failed)} array task(s) failed:")
        for task_id, state, exit_code in failed:
            logger.warning(f"  {task_id}: {state} (exit {exit_code})")
    else:
        logger.info("All array tasks completed successfully.")
