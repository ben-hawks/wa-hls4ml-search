# Installing on HPRC

```
# $SCRATCH is assumed to be your scratch directory on HPRC. The repo itself (code + config JSON) is
# fine in $SCRATCH; large generated *output* (synthesis results, project tarballs) should instead be
# written to your ACES project allocation ($PROJECT, i.e. /scratch/group/<project-id>) -- see the
# --account/--hlsproj/-o paths used by the *_slurm.sh scripts below.
cd $SCRATCH
git clone git@github.com:ben-hawks/wa-hls4ml-search.git
cd wa-hls4ml-search
mkdir hlsproj
mkdir hlsproj/output
cd HPRC_scripts
source modules.sh

# Create the conda environment used by every *_slurm.sh script in this directory (they all run
# `conda run --name wa-hls4ml ...` / `source activate wa-hls4ml`). Use the environment.yml at the repo
# root, not a venv + requirements.txt -- requirements.txt here is kept for reference but the scripts
# do not use it (the venv-activation lines in each *_slurm.sh are present but commented out).
conda env create -f ../environment.yml -n wa-hls4ml
```

Every `*_slurm.sh` script in this directory bills SUs to project account `157537460776` via `#SBATCH --account=157537460776`. Confirm this is still the correct account (via `myproject -l`) before submitting anything.

# runner.sh usage

`runner.sh` (and `2layer_slurm.sh`, which it submits) targets the **2-layer model grid-search generation phase**. That phase's source data (`pregen_2layer_models/*.csv` filelists) and the script it drives (`iter_manager.py` v1, now under `../deperecated/`) are not present in this checkout -- this pairing appears to be leftover from a generation phase that already ran and completed (the published `wa-hls4ml` dataset already includes a `2layer` split). Don't use `runner.sh`/`2layer_slurm.sh` as a template for new work; they'd fail immediately as-is (missing input files), and separately request `--partition=staff`, which is not a valid ACES partition. If you specifically want to resume/extend the 2-layer or 3-layer grid-search splits, you'll need to regenerate the filelists first -- that's a bigger task than fixing this script.

For everything else (dense-latency, dense-resource, dense-latency-fast, conv1d, conv2d), use the matching runner:

```
bash dense_latency_runner.sh <start_config_num> <end_config_num>
bash dense_resource_runner.sh <start_config_num> <end_config_num>
bash dense_latency_fast_runner.sh <start_config_num> <end_config_num>
bash runner_1d.sh <start_config_num> <end_config_num>          # conv1d
sbatch --ntasks=$((end_config_num - start_config_num)) conv2d_slurm.sh <start_config_num> <end_config_num>   # no dedicated runner_2d.sh exists yet
```

Each of these submits a single Slurm job whose allocation is subdivided into `(end_config_num - start_config_num)` concurrent `srun` steps -- one per batch-config file, each running all of that file's models across the architecture's full reuse-factor sweep. Each step requests 2 cores (`--cpus-per-task=2`) and 32 GB of memory (`--mem-per-cpu=16384`).

## Example
```
bash dense_latency_runner.sh 1 49
```
This will run configs 1-48 (inclusive) in parallel, each as its own `srun` step within one Slurm job.

See `../../planning/golden_rules.md` and `../../planning/dataset_gen_plan.md` (alongside this repo, under `agentic-dataset-gen/planning/`) for ACES-specific policy constraints and the current dataset-extension plan before submitting anything at scale.

# `run_synthesis_array.py` (new, finer-grained SLURM array alternative)

`../run_synthesis_array.py` (repo root) is a newer, additive alternative to the `*_slurm.sh`/`*_runner.sh` pattern above -- it is **not yet a replacement**; both exist side by side until the array approach is validated on a real ACES pilot (see below). Where each `*_slurm.sh` script submits one `srun` step per ~50-model batch file (that step then running every model in the file through the architecture's whole RF sweep sequentially, in-process, with no isolation between failures -- the first exception in a 200-unit loop kills every remaining unit in that step), `run_synthesis_array.py` expands every `(batch_file, model, RF)` triple into its own unit of work up front and runs each one as an isolated subprocess, chunked into a real SLURM job array.

```bash
# 1. Expand a batch-file glob x RF sweep into a joblist, dropping already-complete units
python run_synthesis_array.py --prepare \
    --arch dense_latency \
    --batch-glob "${REPO_DIR}/dense_latency_models/dense_latency_batch_*.json" \
    --output "${PROJECT_DIR}/output/dense_latency_run_vsynth_2024-2" \
    --hlsproj "${HLS_PROJ_OUT}" \
    --strat latency --rf-lower 0 --rf-upper 128 --rf-step 32
# -> logs a new run directory, e.g. .../output/dense_latency_run_vsynth_2024-2/_runs/run_dense_latency_<timestamp>/

# 2. Render the array script without submitting, to sanity check sizing first
python run_synthesis_array.py --submit <RUN_DIR> --dry-run

# 3. Actually submit (writes slurm_job_id.txt, then polls squeue/sacct until done)
python run_synthesis_array.py --submit <RUN_DIR>

# 4. If step 3's process dies mid-poll, the array keeps running under SLURM independently --
#    check on it later without re-submitting:
python run_synthesis_array.py --status <RUN_DIR>
```

**Do not point this at a full architecture's worth of work yet.** Chunk size (`-K`/`--units-per-chunk`), array size (`--array-max-tasks`, default 500), concurrency (`--array-concurrency`, default 8), and in-chunk parallelism (`-P`/`--units-parallel`, default 4) are all conservative placeholders pending the pilot steps in `planning/dataset_gen_plan.md`:

1. `scontrol show config | grep -i MaxArraySize` on a login node -- confirms the real ceiling `--array-max-tasks` should respect.
2. A small throwaway array (e.g. against `dense_latency_fast_small`, confirmed zero published-overlap and lowest-risk per `planning/dataset_gen_plan.md` #2b) while watching `squeue -u $USER -t RUNNING | wc -l`, to determine whether SLURM array tasks count individually against ACES's 40-concurrent-running-job cap (`planning/golden_rules.md` #3). If they do, `--array-concurrency` has a hard ceiling around ~30-35 *system-wide*, not just for this run.
3. Use that pilot's `seff` output + observed per-unit wall-clock to tune real `-K`/`-P`/`--slurm-time` defaults before a full-architecture run.

`run_one_unit.py` (repo root) is what each chunk actually shells out to per unit -- it calls the same `run_search_iteration.run_iter()` the existing scripts do, so the on-disk result schema (`_processed.json`, `raw_reports/`, `projects/*.tar.gz`) is unchanged; `util/batch_compress_files.py`, `util/json_dataset_merge.py`, and `util/fixes/*` all keep working against either script's output without modification.
