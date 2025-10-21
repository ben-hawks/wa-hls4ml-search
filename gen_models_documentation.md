# This explains how to use gen_models.py

## gen_models.py
This is a tool that allows the user to randomly generate a set of models based on specified parameter ranges. 
It provides the main generator, **ModelGenerator**. The class contains multiple methods but there are only 3 main API calls.

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


### reset_layers
The network generation feature modifies the member variables during the generation process. This is left untouched so the user could monitor state. It is required to call **reset_layers()** before a subsequent generation.

### load_models
Parses and loads the previously generated modules. Supports **load_models(file_path)**

### threading
There is a threading workflow which is callable under **gen_models.thread_exec(batch_range, batch_size)** which will generate **batch_range** threads targetting **batch_size** workloads each.