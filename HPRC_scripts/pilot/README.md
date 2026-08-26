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

## Run these in order, from the repo root on an ACES login node, with `bash` -- not `source`

```bash
cd $SCRATCH/wa-hls4ml-search
bash HPRC_scripts/pilot/00_gather_env.sh                    # instant, read-only, free
bash HPRC_scripts/pilot/01_array_concurrency_test.sh         # ~2-3 min, near-zero cost (sleep jobs only)
bash HPRC_scripts/pilot/02_synthesis_timing_pilot.sh --dry-run   # instant, review the rendered script, no cost
bash HPRC_scripts/pilot/02_synthesis_timing_pilot.sh          # real thing -- costs real SUs, see below
bash HPRC_scripts/pilot/03_collect_report.sh                 # bundles everything into one file
```

**Use `bash script.sh`, never `source script.sh` / `. script.sh`.** All four scripts set
`set -uo pipefail` and some do `cd` -- sourced instead of executed, those apply to your
*interactive shell*, not a subprocess, so a `cd` inside the script permanently moves your
shell, and a script hitting `exit` (e.g. on a bad input) closes your shell/session
outright instead of just ending the script. This is also what caused the first run's
"Couldn't parse the run directory... while pilot_02_prepare.log doesn't exist" failure --
`02_synthesis_timing_pilot.sh` was run via `source`, and its `tee`-then-immediately-
`grep`-the-same-file pattern isn't robust to whatever cwd state a sourced script leaves
behind. `02` no longer depends on re-reading a file it just wrote (see below) and `01` was
hardened the same way, but the `bash`-not-`source` rule still applies to all four.

Then send back the file `03_collect_report.sh` writes (`pilot_full_report_<timestamp>.txt`)
-- paste its contents into chat, or transfer the file itself.

## What each one costs

- **00**: nothing. Pure read-only commands (`scontrol`, `sinfo`, `squeue`, `myproject`,
  `showquota`, `module spider`) -- safe to run directly on a login node.
- **01**: near-zero. Submits a 60-task SLURM array where each task just `sleep`s for 90
  seconds -- no Vivado, no hls4ml, nothing that costs meaningful SUs. Deliberately
  requests more concurrency (`%60`) than the suspected 40-job cap, specifically to see
  whether SLURM caps it anyway.
- **02**: real cost. Runs actual `--vsynth` synthesis on 20 models (`--limit 20`, a
  random subset of whichever `dense_latency_fast_small` batch file it finds first) at
  RF=1 -- the smallest, fastest-synthesizing, zero-published-overlap corpus in this
  repo, but still real Vivado runs. The first run of this script (before `--limit`
  existed) found that batch file actually contains 200 models, not the 50
  `repo_notes.md`'s table describes -- `--limit` keeps the pilot's cost bounded and
  predictable regardless of that. Run the `--dry-run` pass first and read the rendered
  `job_array.sh` before running for real. For an even smaller/cheaper first look, edit
  the `--limit`/`--units-per-chunk`/`--array-concurrency` values in the script down
  further.
- **03**: nothing. Just `cat`s together whatever 00/01/02 already produced.

## If something looks off

These scripts assume the same paths/module names/account already hardcoded into
`HPRC_scripts/*_slurm.sh` (account `157537460776`, Vivado at
`/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh`, conda env `wa-hls4ml`, `$PROJECT` at
`/scratch/group/p.cis250242.000/wa-hls4ml`). If any of those have changed, or `00`'s
output shows something missing, fix the relevant line before running `01`/`02` rather
than letting them fail partway through a real submission.
