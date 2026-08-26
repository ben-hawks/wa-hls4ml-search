# This explains how to use gen_models.py

## gen_models.py
This is a tool that allows the user to randomly generate a set of models based on specified parameter ranges. 
It provides the main generator, **ModelGenerator**. The class contains multiple methods but there are only 3 main API calls.

### Command-line usage (the way this script is actually invoked in production)

The sections below describe the underlying `ModelGenerator` Python API, but in practice this script is run as a CLI tool, not imported and called directly:

```
python gen_models.py -c <config.json> -b <batch_range> -s <batch_size> -o <output_dir> -p <prefix>
python gen_models.py -c <config.json> -o <output_dir> -p <prefix> --cartesian
python gen_models.py -c <config.json> -o <output_dir> -p <prefix> --lhs <N> [--lhs-seed <seed>]
```

* `-c/--config`: path to a JSON config file of the `add_params` fields described below (see `config_dense_latency_fast.json` for a real example; `config_example.json` is an annotated reference only -- it contains comments and is not valid JSON to load directly). If omitted, falls back to the hardcoded `get_default_config()` in `gen_models.py`.
* `-b/--batch_range`: number of output batch files to generate (default 1). Ignored by `--cartesian`/`--lhs`, which instead chunk their (respectively exhaustive or fixed-`N`) output into as many `<prefix>_batch_<i>.json` files as needed to keep each file at or under 1000 models.
* `-s/--batch_size`: number of models per batch file (default 50). Ignored by `--cartesian`/`--lhs` (see above).
* `-o/--output_dir`: output directory for the batch JSON files (default `dense_resource_test`).
* `-p/--prefix`: prefix for generated model names (`<prefix>_<n>`) and output filenames (`<prefix>_batch_<i>.json`) (default `model`). Set this to match the corpus being generated.
* `--cartesian`: enumerate the full Cartesian product of the design space instead of random sampling -- see "Structured generation modes" below. Mutually exclusive with `--lhs`.
* `--lhs N`: draw `N` Latin Hypercube samples instead of enumerating (`--cartesian`) or independently random-sampling (default) the design space -- see "Structured generation modes" below. Mutually exclusive with `--cartesian`.
* `--lhs-seed SEED`: seed for a reproducible `--lhs` draw (default: a fresh seed is generated, logged, and recorded in `<prefix>_lhs_manifest.json`).

The config file also supports two fields not part of the `gen_network()` `add_params` dict below, read directly by the CLI entry point (`generate_model`/`threaded_exec`, not `gen_network` itself) when using the **default** (non-`--cartesian`, non-`--lhs`) random-sampling mode:
* `min_layer_count` / `max_layer_count`: the actual network depth used per model is `random.randint(min_layer_count, max_layer_count)`, not the `total_layers` parameter documented under `gen_network` below (that parameter is only relevant if you call `gen_network()` directly from Python).
* `max_bit_width_po2`: **each generated model's actual `weight_bit_width`/`activ_bit_width` is resampled** as `2 ** random.randint(2, max_bit_width_po2)` (both weight and activation set to the same sampled value, `weight_int_width`/`activ_int_width` forced to 1) -- this overwrites whatever `weight_bit_width`/`activ_bit_width` you set elsewhere in the config file when generating via the CLI. If you need fixed, non-resampled bit widths, call `ModelGenerator.gen_network()` directly from Python instead of going through the CLI/`threaded_exec` path.

Default-mode generation is parallelized across CPU cores via [Ray](https://www.ray.io/), initialized lazily (only when the default random-sampling path actually runs, not at module import) with `num_cpus` defaulting to `min(cpu_count, 4)` rather than every core on the machine -- override via the `RAY_NUM_CPUS` env var. `--cartesian`/`--lhs` don't use Ray at all (they build models directly in-process), so `ray` doesn't even need to be installed to use those two modes. Keep the parallelism default in mind if running the default mode as a step inside a larger Slurm job that also expects to control core allocation.

### Structured generation modes: `--cartesian` and `--lhs`

Both are alternatives to the default independent-random sampling above, and both are **dense-only** (`QDense` + `QActivation` per layer -- no conv/time layers, dropout, bias, pooling, or padding). They dispatch on which of two config shapes you provide:

* **General shape** (supports both `--cartesian` and `--lhs`): fixed network depth, one size range per layer.
  ```json
  {
    "input_lb": 4, "input_ub": 32,
    "layers": [
      {"size_lb": 4, "size_ub": 32},
      {"size_lb": 4, "size_ub": 32}
    ],
    "bitwidths": [4, 8, 12],
    "weight_int_width": 2, "activ_int_width": 2,
    "probs": {"activations": [1, 1, 1, 0]}
  }
  ```
  `bitwidths` may instead be given as `bitwidth_lb`/`bitwidth_ub` (an even-integer range, matching the legacy shape's convention). `probs.activations` follows the same `["relu", "tanh", "sigmoid", "softmax"]` order as everywhere else in this file -- a `0` weight excludes that activation.

* **Legacy shape** (`--cartesian` only): `dense_lb`/`dense_ub` + `min_layer_count`/`max_layer_count` + `max_bit_width_po2`, i.e. the same fields the default random-sampling mode reads (see `config_dense_latency_fast.json`). Depth varies per generated design across `[min_layer_count, max_layer_count]`; `--lhs` does not support this shape (see below).

**`--cartesian`** enumerates every combination in the chosen shape's design space exactly once -- useful for small, fully-covered sweeps (e.g. a 1-2 layer grid) where you want guaranteed, deterministic, gap-free coverage rather than a sample.

**`--lhs N`** (general shape only) draws `N` points via a scrambled Latin Hypercube (`scipy.stats.qmc.LatinHypercube`, `optimization="random-cd"`) over the joint space of input size + per-layer size (each stratified in log2-space, then snapped to the nearest valid power-of-2) + bitwidth (stratified by index over the configured choices) -- a stratified alternative to independent random sampling for a fixed generation budget, so a batch of `N` models covers the *joint* space more evenly than `N` independent random draws would (independent sampling's coverage gaps show up jointly across axes, not in any single axis's marginal frequency). Activation choice is deliberately **not** part of the LHS cube -- it's a low-cardinality categorical choice that independent weighted sampling already tracks well, so the LHS dimensionality budget is spent on the numeric axes instead. `--lhs` doesn't support the legacy shape: that path's variable per-slot layer-type choice makes it a variable-structure space, not the fixed-dimensional hypercube LHS needs.

Every `--lhs` run also writes `<prefix>_lhs_manifest.json` to the output directory: the seed used, sample count, per-axis bounds/discrete choice sets, and the design shape's total joint cardinality `M` (the size of its full discrete space). If `N` exceeds 10% of `M`, a warning is logged suggesting `--cartesian` instead, since at that point you're sampling most of an enumerable space rather than a small fraction of a large one.

**Extending an `--lhs` batch later**: re-run with a new (or omitted) `--lhs-seed` and treat it as an independent draw. Two independent scrambled draws will very rarely land on the exact same discrete tuple at realistic config sizes and sample counts (the manifest's `joint_cardinality` tells you exactly how unlikely) -- this is not worth building coordinated non-overlap tracking for; if the manifest's 10%-of-M warning starts firing routinely for a config you extend often, switch that config to `--cartesian` instead of continuing to sample it.

### gen_network
This is the main API call.

The function header is as follows.
```
def gen_network(self, total_layers: int = 3,
                add_params: dict = {}, callback=None,
                save_file: typing.IO = None)

total_layers -- (default: 3)    the number of specified layers to generate. (Conv, Dense, etc.)
save_file    -- (default: None) where the files generated saves to
add_params   -- (default: {})   custom fields to set for generation
callback     -- (default: None) a custom function. Upon return the generation is aborted and the value is returned.
```

By default this is the parameters list for generation. If these are not specified by the user the values used are as follows.
```
self.params = {
    'dense_lb': 32, 'dense_ub': 1024,                   -- dense layer size range (clipped to base2)
    'conv_init_size_lb': 32, 'conv_init_size_ub': 128,  -- conv2d layer input range (clipped to base2)
    'conv_filters_lb': 3, 'conv_filters_ub': 64,        -- conv2d layer filter range (clipped to base2)
    'conv_stride_lb': 1, 'conv_stride_ub': 3,           -- conv2d layer stride range
    'conv_kernel_lb': 1, 'conv_kernel_ub': 6,           -- conv2d layer kernel size range
    'time_lb': 30, 'time_ub': 150,                      -- conv1d time-step dimension range
    'conv_flatten_limit': 8,                            -- minimum output dimension size of a conv layer before it flattens
    'q_chance': .5,                                     -- probability we use qkeras vs keras
    'activ_bit_width': 8, 'activ_int_width': 4,         -- range for qkeras bitwidths
    'weight_bit_width': 6, 'weight_int_width': 3,       -- range for qkeras bitwidths
    'probs': {                                          -- hyperparameter generation chances. Default is a uniform distribution
        'activations': [.30,.30,.30,.10],
         # Activations: ["relu", "tanh", "sigmoid", "softmax"]
         # if a layer is quantized, softmax is removed, the last element of
         # the [activations][probs] entry is removed, and others are quantized
        # For layer probs, you must set probabilities for the layers in start_layers as well!
        # conv layers
        # q_chance = 0 [Conv2D]
        # q_chance < 1 [Conv2D, QConv2D, QSeparableConv2D, QDepthwiseConv2D]
        # q_chance = 1 [QConv2D, QSeparableConv2D, QDepthwiseConv2D]
        # Dense/Time are either qkeras or not in line with q_chance,
        # 1 element if 0/1, else 2 elements
         'dense_layers': [], 'conv_layers': [], 'time_layers': [],
         # start layers
         #q_chance = 0 [Conv1D, Conv2D, Dense]
         #q_chance < 1 [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]
         #q_chance = 1 [QConv1D, QConv2D, QDense, QSeparableConv2D, QDepthwiseConv2D]
         'start_layers': [],
        'padding': [0.5, 0.5],  # border, off
        'pooling': [0.5, 0.5]  # max, avg
    },
    'activation_rate': .5,                              -- probability we apply activation function per layer
    'dropout_chance': .5,                               -- probability dropout is on
    'dropout_rate': .4,                                 -- how much to dropout if dropout on
    'flatten_chance': .5,                               -- probability the conv layer flatten itself
    'pooling_chance': .5,                               -- probability we apply pooling
    'bias_rate': .5,                                    -- probability we apply bias to the layer
    'layers_blacklist': []}                             -- Class of layers we don't want to include
```

#### Setting probabilities:
This is intentionally left sensitive. To set likelihood of model types, the lists within *'probs'* can be set to define a custom distribution. 

For ex.
Disabling the no activation function, relu and softmax would be:
```
    self.activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]

    params['probs']['activations'] = [0, 0, 0.5, 0.5, 0]
```

The same logic applies to layers but be aware, based on qkeras and keras the layer widths are different and require alternate handling.

#### Setting callback function:
This is more of an advanced functionality giving the user full control of the generation pipeline. The callback function is triggered after the previous layer has been constructed. As parameters it must be able to expect self and layers. 

Ex. 
The code below would be to generate networks exactly 3 conv layers and 7 transformation layers.
```

# function changes probabilities during generation
def callback(mg: Model_Generator, layers: list):
    if mg.layer_depth > 2:
        mg.params['flatten_chance'] = 1

mg = Model_Generator()
params = {
    'dense_lb': 32, 'dense_ub': 64, 
    'conv_filters_ub': 16, 
    'q_chance': 1,                                          # forces gen to qkeras
    'probs': {'start_layers': [0, 0.33, 0, 0.33, 0.33]},    # forces probs to only qconv layers
    'flatten_chance': 0                                     # never allows transition to dense
    }
model = mg.gen_network(add_params=params, total_layers=7, callback=callback)
model.summary()
```

#### Setting blacklist:
To blacklist a layer type there are 2 options.
    1. 'layers_blacklist' can be set to [layer_type1, layer_type2, ...]
    2. The probability can be set to 0

Option 1 is probably the simplest.

Ex. Blacklisting all dense layer types

```
    params = {'layers_blacklist': [Dense, QDense]}
```

Note: as of this cleanup pass, `layers_blacklist` filtering is implemented in `ModelGenerator.filter_q()` (it previously had no effect -- the key was documented here but never read from `self.params` in the code). It uses the same membership-filtering approach as the q/non-q layer filtering it's applied alongside: if you've also explicitly set `probs` entries for a layer type you're blacklisting, you're responsible for keeping the `probs` list aligned to the resulting (post-filter) layer list, the same caveat that already applied to `q_chance` filtering.

### reset_layers
The network generation feature modifies the member variables during the generation process. This is left untouched so the user could monitor state. It is required to call **reset_layers()** before a subsequent generation.

### load_models
Parses and loads the previously generated modules. Supports **load_models(file_path)**

### threading
Parallel batch generation (the default, non-`--cartesian`/`--lhs` mode) is callable via **gen_models.threaded_exec(batch_range, batch_size, config_params, output_dir, prefix='model')**, which generates **batch_range** output files, each containing **batch_size** models, generated concurrently via Ray -- `num_cpus` defaults to `min(cpu_count, 4)`, overridable via `RAY_NUM_CPUS`. This is what the CLI entry point (see "Command-line usage" above) calls when neither `--cartesian` nor `--lhs` is given.