"""SLURM job-array submission for run_synthesis_array.py.

Public API:
    slurm.cli.add_slurm_args(parser)   -- argparse group consumed by run_synthesis_array.py
    slurm.job_array.write_array_script(run_dir, manifest, submission, args) -- render job_array.sh
    slurm.job_array.submit(run_dir, manifest, submission, args, dry_run=False) -- submit + poll
    slurm.job_array.poll(job_id) -- squeue/sacct polling loop, reusable standalone

See run_synthesis_array.py's module docstring for how these fit into the overall
prepare -> submit -> run-chunk -> status flow.
"""
