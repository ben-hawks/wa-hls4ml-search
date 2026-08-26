# Phase 0 pilot scripts

`00`/`01` answered the two SLURM-behavior questions blocking real trust in
`run_synthesis_array.py`'s defaults -- **both now confirmed** (see
`planning/dataset_gen_plan.md` #6 for the full readout, and `slurm/cli.py` for where
the confirmed numbers now live as actual defaults): real `MaxArraySize` is 1001, and
SLURM array tasks *do* count individually against ACES's 40-concurrent-running-job cap.

`02` is an ongoing timing/reliability pilot, now parameterized over `-P`/`--units-parallel`
so it can be re-run at different concurrency levels for comparison. The first run
(`P=4`, 2026-08-26) found a 2/20 failure rate root-caused to Vivado's `synth_design`
internally multithreading up to 7 processes *per unit* (visible in
`slurm_logs/task_N.out` as `"Multithreading enabled for synth_design using a maximum
of 7 processes"`) -- meaning `P=4` oversubscribes an 8-core chunk by up to ~28 threads.
See "Comparing concurrency levels" below for the follow-up test this prompted.

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

## Comparing concurrency levels

`02_synthesis_timing_pilot.sh` now takes `-P`/`--units-parallel`, `-K`/`--units-per-chunk`,
`--limit`, and `--array-concurrency` overrides, and tags its output files with the
parallelism level (`pilot_02_p4_*`, `pilot_02_p2_*`, ...) so multiple runs don't
overwrite each other -- `03_collect_report.sh` picks up every tagged set it finds.

To test whether lowering parallelism reduces the failure rate observed at `P=4`:

```bash
bash HPRC_scripts/pilot/02_synthesis_timing_pilot.sh --units-parallel 2 --units-per-chunk 4
# K=4 at P=2 -> 2 waves/chunk, same wave count as the original P=4,K=5 run (~2 waves),
# so the two runs are comparable on chunk shape, not just parallelism.
bash HPRC_scripts/pilot/03_collect_report.sh
```

A fresh `--prepare` draws a new random subset of `dense_latency_fast_small` each time
(auto-excluding whatever's already complete from a prior run against the same output
dir), so the `P=2` run mostly hits different units than `P=4` did -- comparable, not
identical, samples. Compare the two tagged sections' `--status` line (succeeded/failed
counts) and `sacct`/`seff` timing once both have run.

## What each one costs

- **00**: nothing. Pure read-only commands (`scontrol`, `sinfo`, `squeue`, `myproject`,
  `showquota`, `module spider`) -- safe to run directly on a login node.
- **01**: near-zero. Submits a 60-task SLURM array where each task just `sleep`s for 90
  seconds -- no Vivado, no hls4ml, nothing that costs meaningful SUs. Requested more
  concurrency (`%60`) than the 40-job cap on purpose, to see whether SLURM would cap it
  anyway -- confirmed it does (pinned at exactly 40 the entire time).
- **02**: real cost, per run. Runs actual `--vsynth` synthesis on `--limit` models
  (default 20, a random subset of whichever `dense_latency_fast_small` batch file it
  finds first) at RF=1 -- the smallest, fastest-synthesizing, zero-published-overlap
  corpus in this repo, but still real Vivado runs. (The first run of this script,
  before `--limit` existed, found that batch file actually contains 200 models, not the
  50 `repo_notes.md`'s table describes.) Run the `--dry-run` pass first and read the
  rendered `job_array.sh` before running for real. `--limit`/`-K`/`-P`/
  `--array-concurrency` are all CLI flags now (see "Comparing concurrency levels") --
  no need to edit the script to change them.
- **03**: nothing. Just `cat`s together whatever 00/01/02 already produced.

## If something looks off

These scripts assume the same paths/module names/account already hardcoded into
`HPRC_scripts/*_slurm.sh` (account `157537460776`, Vivado at
`/sw/hprc/sw/amd/Vivado/2024.2/settings64.sh`, conda env `wa-hls4ml`, `$PROJECT` at
`/scratch/group/p.cis250242.000/wa-hls4ml`). If any of those have changed, or `00`'s
output shows something missing, fix the relevant line before running `01`/`02` rather
than letting them fail partway through a real submission.
