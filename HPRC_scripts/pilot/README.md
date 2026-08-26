# Phase 0 pilot scripts

These answer the three open questions blocking real trust in `run_synthesis_array.py`'s
defaults (see `HPRC_scripts/README.md` and `planning/dataset_gen_plan.md` #6):

1. What is ACES's real SLURM `MaxArraySize`? (`--array-max-tasks` defaults to a
   conservative 500 pending this.)
2. Does a SLURM job array count each task individually against ACES's 40-concurrent-
   running-job cap, or does the whole array count as one job? (`--array-concurrency`
   defaults to a conservative 8 pending this -- if array tasks count individually, the
   real ceiling is much lower than you'd guess from `--array=...%N` alone.)
3. What's real per-unit synthesis wall-clock time and memory usage for this pipeline,
   so `-K`/`-P`/`--slurm-time` can be tuned from evidence instead of guesses?

## Run these in order, from the repo root on an ACES login node

```bash
cd $SCRATCH/wa-hls4ml-search
bash HPRC_scripts/pilot/00_gather_env.sh                    # instant, read-only, free
bash HPRC_scripts/pilot/01_array_concurrency_test.sh         # ~2-3 min, near-zero cost (sleep jobs only)
bash HPRC_scripts/pilot/02_synthesis_timing_pilot.sh --dry-run   # instant, review the rendered script, no cost
bash HPRC_scripts/pilot/02_synthesis_timing_pilot.sh          # real thing -- costs real SUs, see below
bash HPRC_scripts/pilot/03_collect_report.sh                 # bundles everything into one file
```

Then send back the file `03_collect_report.sh` writes (`pilot_full_report_<timestamp>.txt`)
-- paste its contents into chat, or transfer the file itself.

## What each one costs

- **00**: nothing. Pure read-only commands (`scontrol`, `sinfo`, `squeue`, `myproject`,
  `showquota`, `module spider`) -- safe to run directly on a login node.
- **01**: near-zero. Submits a 60-task SLURM array where each task just `sleep`s for 90
  seconds -- no Vivado, no hls4ml, nothing that costs meaningful SUs. Deliberately
  requests more concurrency (`%60`) than the suspected 40-job cap, specifically to see
  whether SLURM caps it anyway.
- **02**: real cost. Runs actual `--vsynth` synthesis on up to 50 models (one
  `dense_latency_fast_small` batch file) at RF=1 -- the smallest, fastest-synthesizing,
  zero-published-overlap corpus in this repo, but still real Vivado runs. Run the
  `--dry-run` pass first and read the rendered `job_array.sh` before running for real.
  If you want an even smaller/cheaper first look, edit `--units-per-chunk`/
  `--array-concurrency` in the script down further, or point `BATCH_FILE` in the script
  at a hand-trimmed JSON file with fewer than 50 entries.
- **03**: nothing. Just `cat`s together whatever 00/01/02 already produced.

## If something looks off

These scripts assume the same paths/module names/account already hardcoded into
`HPRC_scripts/*_slurm.sh` (account `157537460776`, Vivado at
`/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh`, conda env `wa-hls4ml`, `$PROJECT` at
`/scratch/group/p.cis250242.000/wa-hls4ml`). If any of those have changed, or `00`'s
output shows something missing, fix the relevant line before running `01`/`02` rather
than letting them fail partway through a real submission.
