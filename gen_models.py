import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import yaml
import argparse
from qkeras import QDense, QConv2D, QConv1D, QAveragePooling2D, QActivation, quantized_bits, QDepthwiseConv2D, QSeparableConv2D, QSeparableConv1D, QLSTM
from keras.layers import Dense, Conv2D, Flatten, Activation, Conv1D, LSTM, Layer, Input
from keras.models import Model, model_from_json
from qkeras.utils import _add_supported_quantized_objects
import keras
import typing
import random
import time
import itertools
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import sys, os

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@contextmanager
def suppress_stdout():
    """
    Context manager to suppress stdout.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#clip_base_2 = lambda x: 2 ** round(np.log2(x))
clip_base_2 = lambda x: 2 ** max(1, round(np.log2(max(1, x))))

class ModelGenerator:
    """
    Class for generating neural network models with various configurations.
    """
    
    def __init__(self):
        """
        Initialize the ModelGenerator with default parameters and layer configurations.
        """
        self.reset_layers()
        self.failed_models = 0

    def config_layer(self, layer_type: Layer) -> dict:
        """
        Returns hyper parameters for layer initialization as a dict.
        
        Args:
            layer_type (Layer): The type of layer to configure
            
        Returns:
            dict: Configuration parameters for the layer
        """
        # Ensure weights match the population
        #if len(self.params['probs']['activations']) != len(self.activations):
        #    self.params['probs']['activations'] = [1 / len(self.activations)] * len(self.activations)
        activation = random.choices(self.activations, weights=self.params['probs']['activations'], k=1)[0]
        use_bias = random.random() < self.params['bias_rate']

        if layer_type in self.dense_layers:
            layer_size = clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub']))
            dropout = random.random() < self.params['dropout_chance']

            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': self.params['dropout_rate']}
        elif layer_type in self.conv_layers:
            out_filters = clip_base_2(random.randint(self.params['conv_filters_lb'], self.params['conv_filters_ub']))
            flatten = (random.random() < self.params['flatten_chance']) or \
                      (self.params['last_layer_shape'][0] < self.params['conv_flatten_limit'] or
                       self.params['last_layer_shape'][1] < self.params['conv_flatten_limit'])

            pooling = random.random() < self.params['pooling_chance']
            padding = random.choices(['same', 'valid'], weights=self.params['probs']['padding'], k=1)[0]
            kernel_size = min(random.randint(self.params['conv_kernel_lb'], self.params['conv_kernel_ub']),
                              *self.params['last_layer_shape'][:-1])

            stride = random.randint(self.params['conv_stride_lb'], self.params['conv_stride_ub'])
            row_dim_pred = (self.params['last_layer_shape'][0] - kernel_size + 2 * int(padding == 'valid')) / stride + 1
            col_dim_pred = (self.params['last_layer_shape'][1] - kernel_size + 2 * int(padding == 'valid')) / stride + 1

            if row_dim_pred <= 0 or col_dim_pred <= 0:
                kernel_size, stride, padding = 1, 1, 'same'

            hyper_params = {'out_filters': out_filters, 'kernel': (kernel_size, kernel_size),
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'pooling': pooling, 'padding': padding, 'stride': (stride, stride)}
        elif layer_type in self.time_layers:
            out_filters = clip_base_2(random.randint(self.params['conv_filters_lb'], self.params['conv_filters_ub']))
            kernel_size = random.randint(self.params['conv_kernel_lb'], self.params['conv_kernel_ub'])
            flatten = random.random() < self.params['flatten_chance']
            stride = random.randint(self.params['conv_stride_lb'], self.params['conv_stride_ub'])
            padding = random.choices(['same', 'valid'], weights=self.params['probs']['padding'], k=1)[0]
            pooling = random.random() < self.params['pooling_chance']
            hyper_params = {'out_filters': out_filters, 'kernel': kernel_size,
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'pooling': pooling, 'padding': padding, 'stride': stride}
        return hyper_params

    def next_layer(self, last_layer: Layer, input_layer: Layer = None, pre_config: dict = None) -> Layer:
        """
        Takes previous layer and configuration displays and returns back layer.
        
        Args:
            last_layer (Layer): Previous keras/qkeras layer
            input_layer (Layer, optional): Input layer
            pre_config (dict, optional): Predefined configuration
            
        Returns:
            list: List of layers
        """
                
        if 'dense' in self.name:
            layer_type = random.choices(self.dense_layers, weights=self.params['probs']['dense_layers'], k=1)[0] if input_layer is None else last_layer
            hyper_params = self.config_layer(layer_type) if not pre_config else pre_config
            
            last_layer = last_layer if input_layer is None else input_layer
            
            if self.q_on:
                layer_choice = [layer_type(hyper_params['size'],
                                           kernel_quantizer=quantized_bits(self.params['weight_bit_width'],
                                                                           self.params['weight_int_width']),
                                           use_bias=hyper_params['use_bias'])(last_layer)]
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'])(layer_choice[-1]))
            else:
                layer_choice = [layer_type(hyper_params['size'],
                                           use_bias=hyper_params['use_bias'])(last_layer)]
                if hyper_params['dropout']:
                    layer_choice.append(keras.layers.Dropout(hyper_params['dropout_rate'])(layer_choice[-1]))
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'])(layer_choice[-1]))

            self.name = 'dense'
        elif 'conv' in self.name:
            layer_type = random.choices(self.conv_layers, weights=self.params['probs']['conv_layers'], k=1)[0] if input_layer is None else last_layer
            if input_layer is None:
                self.params['last_layer_shape']
            
            hyper_params = self.config_layer(layer_type)
            
            last_layer = last_layer if input_layer is None else input_layer

            if self.q_on:
                if layer_type == QConv2D:
                    layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                               kernel_quantizer=quantized_bits(self.params['weight_bit_width'],
                                                                               self.params['weight_int_width']),
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                elif layer_type == QSeparableConv2D:
                    layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                elif layer_type == QDepthwiseConv2D:
                    layer_choice = [layer_type(hyper_params['kernel'], strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'])(layer_choice[-1]))

                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling2D, keras.layers.AveragePooling2D],
                                             weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling((2, 2))(layer_choice[-1]))
            else:
                layer_choice = [layer_type(hyper_params['out_filters'], hyper_params['kernel'], strides=hyper_params['stride'],
                                           use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'])(layer_choice[-1]))
                
                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling2D, keras.layers.AveragePooling2D],
                                             weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling((2, 2))(layer_choice[-1]))

            self.name = 'conv'

            if hyper_params['flatten'] and input_layer is None:
                layer_choice.append(Flatten()(last_layer))
                self.name = 'dense'
        elif 'time' in self.name:
            layer_type = random.choices(self.time_layers, weights=self.params['probs']['time_layers'], k=1)[0] if input_layer is None else last_layer
            if input_layer is None:
                self.params['last_layer_shape']
            hyper_params = self.config_layer(layer_type)

            last_layer = last_layer if input_layer is None else input_layer
            
            if self.q_on:
                if layer_type == QConv1D:
                    layer_choice = [layer_type(filters=hyper_params['out_filters'], kernel_size=hyper_params['kernel'],
                                               strides=hyper_params['stride'],
                                               kernel_quantizer=quantized_bits(self.params['weight_bit_width'],
                                                                               self.params['weight_int_width']),
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                elif layer_type == QSeparableConv1D:
                    layer_choice = [layer_type(filters=hyper_params['out_filters'], kernel_size=hyper_params['kernel'],
                                               strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                elif layer_type == QLSTM:
                    raise NotImplementedError("LSTM not implemented")
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'])(layer_choice[-1]))
                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling1D, keras.layers.AveragePooling1D],
                                             weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling(2)(layer_choice[-1]))
            else:
                if layer_type == LSTM:
                    raise NotImplementedError("LSTM not implemented")
                elif layer_type == Conv1D:
                    layer_choice = [layer_type(filters=hyper_params['out_filters'], kernel_size=hyper_params['kernel'],
                                               strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                    
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'])(layer_choice[-1]))
                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling1D, keras.layers.AveragePooling1D],
                                             weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling(2)(layer_choice[-1]))
            self.name = 'time'
            if hyper_params['flatten'] and input_layer is None:
                layer_choice.append(Flatten()(last_layer))
                self.name = 'dense'
        self.params['last_layer_shape'] = layer_choice[-1].shape[1:]
        self.layer_depth += 1
        return layer_choice

    def gen_network(self, total_layers: int = 3,
                    add_params: dict = {}, callback=None,
                    save_file: typing.IO = None) -> Model:

        """
        Generates interconnected network based on defaults or extra params, returns Model.
        
        Args:
            total_layers (int): Total active layers in a network (default: 3)
            add_params (dict): Parameters to specify besides defaults for model generation (default: {})
            callback: Callback function for model generation
            save_file: Open file descriptor for log file (default: None)
            
        Returns:
            Model: Generated Keras model
        """

        add_params = {k: add_params[k] for k in add_params}
        self.params = {
            'dense_lb': 32, 'dense_ub': 1024,
            'conv_init_size_lb': 32, 'conv_init_size_ub': 128,
            'conv_filters_lb': 3, 'conv_filters_ub': 64,
            'conv_stride_lb': 1, 'conv_stride_ub': 3,
            'conv_kernel_lb': 1, 'conv_kernel_ub': 6,
            'time_lb': 30, 'time_ub': 150,
            'conv_flatten_limit': 8,
            'q_chance': .5,
            'activ_bit_width': 8, 'activ_int_width': 4,
            'weight_bit_width': 6, 'weight_int_width': 3,
            'probs': {
                'activations': [],
                'dense_layers': [], 'conv_layers': [], 'start_layers': [], 'time_layers': [],
                'padding': [0.5, 0.5],
                'pooling': [0.5, 0.5]
            },
            'activation_rate': .5,
            'dropout_chance': 0,
            'dropout_rate': .4,
            'flatten_chance': .5,
            'pooling_chance': .5,
            'bias_rate': .5
        }

        self.params.update(add_params)
        self.filter_q(self.params['q_chance'], self.params)

        init_layer = random.choices(self.start_layers, weights=self.params['probs']['start_layers'], k=1)[0]

        # Generate input shape based on the initial layer type
        if init_layer in self.dense_layers:
            input_shape = (clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub'])),)
        elif init_layer in self.conv_layers:
            y_dim = random.randint(self.params['conv_init_size_lb'], self.params['conv_init_size_ub'])
            x_dim = random.randint(self.params['conv_init_size_lb'], self.params['conv_init_size_ub'])
            num_filters = clip_base_2(random.randint(self.params['conv_filters_lb'], self.params['conv_filters_ub']))
            input_shape = (y_dim, x_dim, num_filters)
        elif init_layer in self.time_layers:
            input_shape = (clip_base_2(random.randint(self.params['time_lb'], self.params['time_ub'])),
                           random.randint(self.params['dense_lb'], self.params['dense_ub']))
        else:
            raise ValueError("Invalid initial layer type")

        # Validate input_shape
        if not input_shape or not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise ValueError(f"Invalid input shape: {input_shape}")

        try:
            layers = [Input(shape=input_shape)]  # Ensure input_shape is valid
            self.params['last_layer_shape'] = layers[0].shape[1:]

            if init_layer in self.dense_layers:
                self.name = "dense"
            elif init_layer in self.conv_layers:
                self.name = "conv"
            elif init_layer in self.time_layers:
                self.name = "time"
            else:
                raise Exception("Layer not of a valid type")

            self.layer_depth += 1
            layers.extend(self.next_layer(init_layer, input_layer=layers[0]))
            layer_units = 0

            while layer_units < total_layers:
                if callback:
                    callback_output = callback(self, layers)
                    if callback_output:
                        return callback_output

                if layer_units == total_layers - 2 and self.name:
                    self.params['flatten_chance'] = 1
                if layer_units == total_layers - 1:
                    self.params['dropout_rate'] = 0

                layers.extend(self.next_layer(layers[-1]))
                layer_units += 1

            model = Model(inputs=layers[0], outputs=layers[-1])

            if save_file:
                save_file.write(model.to_json())
                save_file.write("--------------")
            return model

        except ValueError as e:
            self.failed_models += 1
            self.reset_layers()
            if self.failed_models > 10:  # Limit recursion depth
                raise RuntimeError("Exceeded maximum retries for generating network") from e
            logger.error(f"Error generating network: {e}")
            return self.gen_network(total_layers=total_layers,
                                    add_params=add_params, callback=callback,
                                    save_file=save_file)

    def reset_layers(self) -> None:
        """
        Used to return class to initial state. Useful if generating multiple networks.
        """
        self.dense_layers = [Dense, QDense]
        self.conv_layers = [QConv2D, Conv2D, QSeparableConv2D, QDepthwiseConv2D]
        self.time_layers = [Conv1D, QConv1D]
        self.start_layers = [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]

        self.activations = ["relu", "tanh", "sigmoid", "softmax"]
        # self.activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]
        
        self.layer_depth = 0

    def filter_q(self, q_chance: float, params: dict) -> None:
        """
        Filter layers based on quantization chance.
        
        Args:
            q_chance (float): Probability of using qkeras over keras
            params (dict): Parameters dictionary
        """
        blacklist = list(params.get('layers_blacklist', []))
        self.q_on = random.random() < q_chance

        # filter out the qkeras/non-qkeras layers
        for layer in set(self.start_layers + self.conv_layers + self.dense_layers):
            is_qkeras = layer.__module__[:6] == 'qkeras'
            if self.q_on ^ is_qkeras:
                blacklist.append(layer)
        # user-supplied 'layers_blacklist' (documented in gen_models_documentation.md but previously
        # unused by this method) is folded into the same filter pass as the q/non-q filtering above,
        # so both use identical semantics: membership-only filtering of the layer-type lists. As with
        # q-filtering, if 'probs' entries were explicitly set for layers removed here, the caller is
        # responsible for keeping probs aligned to the resulting (post-filter) layer list order -- the
        # 'defaults if the layer was not set' block below only auto-generates probs when none were given.
        self.start_layers = [layer for layer in self.start_layers if layer not in blacklist]
        self.dense_layers = [layer for layer in self.dense_layers if layer not in blacklist]
        self.conv_layers = [layer for layer in self.conv_layers if layer not in blacklist]
        self.time_layers = [layer for layer in self.time_layers if layer not in blacklist]

        # adjust activation layers based on quantization
        if self.q_on:
            if 'softmax' in self.activations:
                self.activations.remove('softmax')
                if len(self.activations) != len(self.params['probs']['activations']):
                    self.params['probs']['activations'] = self.params['probs']['activations'][:-1]

            self.activations = [f'quantized_{activ_func}({params["activ_bit_width"]},{params["activ_int_width"]})' for
                                activ_func in self.activations]
            
        # defaults if the layer was not set. Setting these is intentionally very delicate
        pairs = {'activations': self.activations, 'start_layers': self.start_layers, 'dense_layers': self.dense_layers,
                 'conv_layers': self.conv_layers, 'time_layers': self.time_layers}
        for param_type in pairs:
            if param_type not in self.params['probs']:
                self.params['probs'][param_type] = []

            if not self.params['probs'][param_type]:
                self.params['probs'][param_type] = [1 / len(pairs[param_type]) for _ in pairs[param_type]]

        for p_type in ['padding', 'pooling']:
            if p_type not in self.params['probs']:
                self.params['probs'][p_type] = [.5, .5]

    def load_models(self, save_file: str) -> list[Model]:
        """
        Parses and returns an iterable of generated models.
        
        Args:
            save_file (str): Path to batch of models
            
        Yields:
            Model: Generated models
        """

        with open(save_file, "r") as chunk_file:
            models = chunk_file.read().split("--------------")[:-1]
            for model_desc in models:
                co = {}
                _add_supported_quantized_objects(co)
                yield model_from_json(model_desc, custom_objects=co)

def load_config(config_file_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_file_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
        logger.info(f"Loaded configuration from {config_file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_file_path}: {e}")
        raise

def get_default_config():
    """
    Get the default configuration parameters.
    
    Returns:
        dict: Default configuration parameters
    """
    return {
        'dense_lb': 8, 'dense_ub': 128, 'conv_filters_ub': 16,
        'conv_init_size_lb': 8, 'conv_init_size_ub': 64,
        'q_chance': 1, 'flatten_chance': .1, 'pooling_chance': .3,
        'weight_bit_width': 2, 'weight_int_width': 1,
        'activ_bit_width': 2, 'activ_int_width': 1,
        'activation_rate': 1,
        'probs': {
            'activations': [.30,.30,.30,.10],
            'dense_layers': [1], 'conv_layers': [0, 0, 0], 'time_layers': [0],
            'start_layers': [0, 0, 1, 0, 0],
            'padding': [0.5, 0.5],
            'pooling': [0.2, 0.2]
        },
        'total_layers': 5
    }

# --- Structured dense-only generation modes: --cartesian (exhaustive enumeration)
# and --lhs (stratified sampling), as alternatives to the default independent-random
# sampling above. Both dispatch on the same two config shapes:
#   - general mode: 'layers' present -- independent input size + per-layer independent
#     sizes/activations, shared bitwidth axis ('bitwidths' list or 'bitwidth_lb'/'bitwidth_ub').
#   - legacy mode (cartesian only): 'dense_lb'/'dense_ub' + 'min_layer_count'/'max_layer_count'
#     + 'max_bit_width_po2', matching config_dense_latency_fast.json-style configs.
# Both modes are dense-only (QDense + QActivation per layer) -- they don't attempt to cover
# the full random architecture search (conv/time layers, per-layer type choice, dropout/bias/
# pooling/padding) that the default ModelGenerator.gen_network() path supports, since that
# space is a variable-structure tree rather than a fixed-dimensional space these techniques
# assume.

_ACTIVATION_NAMES = ["relu", "tanh", "sigmoid", "softmax"]


def _dense_sizes(lb, ub):
    """All powers of 2 in [lb, ub] -- the discrete grid dense layer widths actually land on
    (see clip_base_2 above)."""
    return [2**i for i in range(0, 20) if lb <= 2**i <= ub]


def _active_activations(probs, names):
    """Activation names with nonzero configured probability, in the same order as `probs`."""
    return [n for n, p in zip(names, probs) if p > 0]


def _build_dense_model(layer_configs, bitwidth, config_params, input_size=None):
    """Build a QKeras dense-only model from explicit per-layer (size, activation) pairs.

    input_size: override the input feature dimension (defaults to the first layer's size,
    for the legacy config shape where input and first-layer size are the same axis).
    """
    w_int = config_params.get('weight_int_width', 1)
    a_int = config_params.get('activ_int_width', 1)
    in_dim = input_size if input_size is not None else layer_configs[0][0]
    inputs = Input(shape=(in_dim,))
    x = inputs
    for size, activ in layer_configs:
        x = QDense(size, kernel_quantizer=quantized_bits(bitwidth, w_int))(x)
        x = QActivation(f'quantized_{activ}({bitwidth},{a_int})')(x)
    return Model(inputs=inputs, outputs=x)


def _write_batch(model_dict, output_dir, batch_i, prefix):
    path = os.path.join(output_dir, f"{prefix}_batch_{batch_i}.json")
    with open(path, "w") as f:
        json.dump(model_dict, f, indent=2)


def cartesian_exec(config_params, output_dir, prefix, chunk_size=1000):
    """Enumerate every unique dense-NN configuration and write batch JSON files.

    See the module-level note above for the two supported config shapes.
    """
    activs = _active_activations(config_params['probs']['activations'], _ACTIVATION_NAMES)

    if 'layers' in config_params:
        # General mode: independent input size, per-layer independent sizes
        input_sizes = _dense_sizes(config_params['input_lb'], config_params['input_ub'])
        bitwidths   = (config_params['bitwidths'] if 'bitwidths' in config_params
                       else list(range(config_params['bitwidth_lb'], config_params['bitwidth_ub'] + 1, 2)))
        n_layers    = len(config_params['layers'])

        layer_options = [
            list(itertools.product(_dense_sizes(l['size_lb'], l['size_ub']), activs))
            for l in config_params['layers']
        ]
        all_combos = list(itertools.product(input_sizes, *layer_options, bitwidths))
        logger.info(f"Cartesian ({n_layers}-layer): {len(all_combos)} designs "
                    f"({len(input_sizes)} in"
                    + "".join(f" x {len(lo)} layer{i+1}" for i, lo in enumerate(layer_options))
                    + f" x {len(bitwidths)} bw)")

        model_dict, batch_i, total = {}, 0, 0
        for idx, combo in enumerate(tqdm(all_combos, desc="Building models")):
            in_size       = combo[0]
            layer_configs = list(combo[1:-1])
            bitwidth      = combo[-1]
            model = _build_dense_model(layer_configs, bitwidth, config_params, input_size=in_size)
            model_dict[f"{prefix}_{idx}"] = model.to_json()
            total += 1
            if len(model_dict) >= chunk_size:
                _write_batch(model_dict, output_dir, batch_i, prefix)
                model_dict, batch_i = {}, batch_i + 1
    else:
        # Legacy multi-layer mode: independent per-layer (size, activation), power-of-2
        # bitwidths up to max_bit_width_po2, variable depth in [min_layer_count, max_layer_count]
        sizes     = _dense_sizes(config_params['dense_lb'], config_params['dense_ub'])
        bitwidths = [2**i for i in range(2, config_params['max_bit_width_po2'] + 1)]
        n_min     = config_params['min_layer_count']
        n_max     = config_params['max_layer_count']

        layer_choices = list(itertools.product(sizes, activs))
        all_combos = [
            (list(combo), bitwidth)
            for n_layers in range(n_min, n_max + 1)
            for combo in itertools.product(layer_choices, repeat=n_layers)
            for bitwidth in bitwidths
        ]
        logger.info(f"Cartesian (legacy multi-layer): {len(all_combos)} designs total")

        model_dict, batch_i, total = {}, 0, 0
        for idx, (layer_config, bitwidth) in enumerate(tqdm(all_combos, desc="Building models")):
            model = _build_dense_model(layer_config, bitwidth, config_params)
            model_dict[f"{prefix}_{idx}"] = model.to_json()
            total += 1
            if len(model_dict) >= chunk_size:
                _write_batch(model_dict, output_dir, batch_i, prefix)
                model_dict, batch_i = {}, batch_i + 1

    if model_dict:
        _write_batch(model_dict, output_dir, batch_i, prefix)
    logger.info(f"Cartesian: wrote {total} designs in {batch_i + 1} file(s)")


def lhs_exec(config_params, output_dir, n_samples, prefix, seed=None, chunk_size=1000):
    """Draw a Latin Hypercube sample of dense-NN configurations, as a stratified
    alternative to independent-random sampling for a fixed generation budget.

    Requires the general 'layers' config shape (see cartesian_exec) -- LHS needs a
    fixed-dimensional space to stratify jointly, which this config shape provides
    (one axis per layer size, plus input size, plus bitwidth); the legacy
    full-architecture-search path (gen_network()'s random conv/time/dense branching,
    variable per-slot type choice) is a variable-structure space, not a fixed
    hypercube, so LHS doesn't apply there without a much bigger redesign -- out of
    scope here, same as where cartesian_exec's own legacy-mode support stops short
    of covering it.

    Activation choice is deliberately NOT part of the LHS cube: it's a low-cardinality
    (3-4 way) categorical choice that independent weighted random sampling already
    tracks closely at realistic batch sizes, so the LHS dimensionality budget is
    better spent on the numeric axes, where independent-random sampling's *joint*
    coverage gaps (not marginal-frequency gaps) actually show up.
    """
    from scipy.stats import qmc

    if 'layers' not in config_params:
        raise ValueError(
            "--lhs requires the general 'layers' config shape (input_lb/input_ub + "
            "layers: [{size_lb, size_ub}, ...]), not the legacy dense_lb/dense_ub + "
            "min_layer_count/max_layer_count shape. See gen_models_documentation.md."
        )

    activ_probs = config_params['probs']['activations']
    activs = _active_activations(activ_probs, _ACTIVATION_NAMES)
    activ_weights = [p for p in activ_probs if p > 0]

    input_sizes = _dense_sizes(config_params['input_lb'], config_params['input_ub'])
    layer_size_sets = [_dense_sizes(l['size_lb'], l['size_ub']) for l in config_params['layers']]
    n_layers = len(config_params['layers'])

    if 'bitwidths' in config_params:
        bitwidth_choices = list(config_params['bitwidths'])
    else:
        bitwidth_choices = list(range(config_params['bitwidth_lb'], config_params['bitwidth_ub'] + 1, 2))

    size_axes = [input_sizes] + layer_size_sets
    d = len(size_axes) + 1  # + 1 for the bitwidth axis

    joint_cardinality = 1
    for choices in size_axes:
        joint_cardinality *= len(choices)
    joint_cardinality *= len(bitwidth_choices)

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    sampler = qmc.LatinHypercube(d=d, scramble=True, optimization="random-cd", seed=seed)
    unit_points = sampler.random(n=n_samples)
    # Activation choice is independent of the LHS cube (see docstring), but still needs
    # its own seeded RNG -- using the global `random` module here would make a "reproducible"
    # --lhs-seed only reproduce the LHS-covered axes (size/bitwidth) and not the full draw.
    # A local Random instance also avoids perturbing global `random` state for callers using
    # this as a library function rather than the CLI.
    activ_rng = random.Random(seed)

    def _snap_log2(u, choices):
        # Interpolate in log2-space (not linear) before snapping to the nearest valid
        # power-of-2 value -- these axes land on a sparse power-of-2 grid, and snapping
        # a linearly-interpolated value would under-sample small values relative to
        # large ones.
        lo, hi = choices[0], choices[-1]
        target_log2 = np.log2(lo) + u * (np.log2(hi) - np.log2(lo))
        return min(choices, key=lambda v: abs(np.log2(v) - target_log2))

    def _snap_index(u, choices):
        # Index-uniform stratification for axes with no assumed spacing (explicit
        # bitwidth lists, or a linear-in-exponent bitwidth_lb/ub range).
        idx = round(u * (len(choices) - 1))
        return choices[idx]

    model_dict, batch_i, total = {}, 0, 0
    for idx in tqdm(range(n_samples), desc="Building LHS models"):
        point = unit_points[idx]
        in_size = _snap_log2(point[0], input_sizes)
        layer_sizes = [_snap_log2(point[1 + j], layer_size_sets[j]) for j in range(n_layers)]
        bitwidth = _snap_index(point[-1], bitwidth_choices)

        layer_configs = [
            (size, activ_rng.choices(activs, weights=activ_weights, k=1)[0])
            for size in layer_sizes
        ]
        model = _build_dense_model(layer_configs, bitwidth, config_params, input_size=in_size)
        model_dict[f"{prefix}_{idx}"] = model.to_json()
        total += 1
        if len(model_dict) >= chunk_size:
            _write_batch(model_dict, output_dir, batch_i, prefix)
            model_dict, batch_i = {}, batch_i + 1

    if model_dict:
        _write_batch(model_dict, output_dir, batch_i, prefix)

    manifest = {
        'mode': 'lhs',
        'seed': seed,
        'n_samples': n_samples,
        'dimensions': d,
        'axes': {
            'input_size': {'bounds': [config_params['input_lb'], config_params['input_ub']],
                            'choices': input_sizes},
            **{
                f'layer_{j + 1}_size': {
                    'bounds': [config_params['layers'][j]['size_lb'], config_params['layers'][j]['size_ub']],
                    'choices': layer_size_sets[j],
                }
                for j in range(n_layers)
            },
            'bitwidth': {'choices': bitwidth_choices},
        },
        'joint_cardinality': joint_cardinality,
    }
    manifest_path = os.path.join(output_dir, f"{prefix}_lhs_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"LHS: wrote {total} designs in {batch_i + 1} file(s), seed={seed}, "
                f"joint cardinality={joint_cardinality} (manifest: {manifest_path})")
    if n_samples > 0.1 * joint_cardinality:
        logger.warning(
            f"Requested {n_samples} LHS samples, >10% of this config's joint discrete "
            f"cardinality ({joint_cardinality}) -- consider --cartesian instead to "
            f"enumerate the full space exhaustively rather than sampling most of it."
        )


def create_parser():
    """
    Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='HLS4ML Model Generator')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file (JSON format)')
    parser.add_argument('-b', '--batch_range', type=int, default=1, help='Number of files to generate')
    parser.add_argument('-s', '--batch_size', type=int, default=50, help='Number of models per file')
    parser.add_argument('-o', '--output_dir', type=str, default='dense_resource_test', help='Output directory')
    parser.add_argument('-p', '--prefix', type=str, default='model',
                       help="Prefix used for both the generated model names (e.g. '<prefix>_<n>') and the "
                            "output batch filenames (e.g. '<prefix>_batch_<i>.json'). Previously hardcoded "
                            "to 'dense_latency_fast' regardless of what was actually being generated -- set "
                            "this explicitly (e.g. --prefix dense_resource) to match the corpus you're producing.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--cartesian', action='store_true',
                       help='Enumerate the full Cartesian product of the design space instead of '
                            'random sampling. Dense-only -- see gen_models_documentation.md.')
    mode_group.add_argument('--lhs', type=int, default=None, metavar='N',
                       help='Draw N Latin Hypercube samples (stratified for better joint coverage '
                            'per sample than independent random sampling) instead of enumerating '
                            '(--cartesian) or randomly sampling (default) the design space. Requires '
                            'the general "layers" config shape -- see gen_models_documentation.md.')
    parser.add_argument('--lhs-seed', type=int, default=None,
                       help='Seed for reproducible --lhs draws (default: a fresh seed is generated, '
                            'logged, and recorded in the output <prefix>_lhs_manifest.json).')
    return parser


def threaded_exec(batch_range: int, batch_size: int, config_params: dict, output_dir: str, prefix: str = 'model'):
    """
    Execute model generation in batches, parallelized across CPU cores via Ray.

    Ray is imported and initialized here, lazily, rather than at module import time --
    this is the only generation mode that needs Ray at all (--cartesian/--lhs build
    models directly with no parallel dispatch), so importing/initializing it eagerly
    at module load would force the dependency and claim CPU cores even for callers who
    never use this path. Parallelism defaults to a conservative min(cpu_count, 4)
    rather than every core on the machine, overridable via the RAY_NUM_CPUS env var --
    matters when this runs as one step inside a larger Slurm job that also expects to
    control its own core allocation.

    Args:
        batch_range (int): Number of files to generate
        batch_size (int): Number of models per file
        config_params (dict): Configuration parameters
        output_dir (str): Output directory
        prefix (str): Prefix for generated model names and output batch filenames (default: 'model').
            Was previously hardcoded to 'dense_latency_fast' here regardless of what was actually being
            generated -- callers should pass a prefix that matches the corpus being produced.
    """
    import ray

    num_cpus = int(os.environ.get("RAY_NUM_CPUS", min(os.cpu_count() or 1, 4)))
    ray.init(num_cpus=num_cpus, log_to_driver=False)

    @ray.remote(max_retries=10, retry_exceptions=False)
    def generate_model(bitwidth, config_params):
        """
        Generate a model with specified bitwidth and configuration.

        Args:
            bitwidth (int): Bitwidth for quantization
            config_params (dict): Configuration parameters

        Returns:
            tuple: (model_name, model_json)
        """
        try:
            mg = ModelGenerator() # Latency strategy Dense jobs
            # Update config params with bitwidth
            config_params['weight_bit_width'] = bitwidth
            config_params['activ_bit_width'] = bitwidth
            config_params['weight_int_width'] = 1
            config_params['activ_int_width'] = 1

            model = mg.gen_network(add_params=config_params,
                                   total_layers=random.randint(config_params['min_layer_count'], config_params['max_layer_count']), save_file=None)
            return model.name, model.to_json()
        except Exception as e:
            logger.error(f"Error generating model: {e}")
            raise(e)

    succeeded = 0

    assert batch_range > 0
    assert batch_size > 0
    for batch_i in tqdm(range(batch_range), desc="Batch Count: "):
        model_dict = {}
        futures = [generate_model.remote(2 ** random.randint(2, config_params["max_bit_width_po2"]), config_params) for _ in range(batch_size)]
        for future in ray.get(futures):
            model_name, model_json = future
            if model_name and model_json:
                # model_name might have dupes because of multithreading, so make a new name for each model
                model_dict.update({f"{prefix}_{succeeded}": model_json})  # Store the model with its name
                succeeded += 1
        _write_batch(model_dict, output_dir, batch_i, prefix)
        #logger.info(f"Saved batch {batch_i}")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_params = load_config(args.config)
        logger.info("Using configuration from file")
    else:
        config_params = get_default_config()
        logger.info("Using default configuration")


    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.cartesian:
        cartesian_exec(config_params, args.output_dir, args.prefix)
    elif args.lhs is not None:
        lhs_exec(config_params, args.output_dir, args.lhs, args.prefix, seed=args.lhs_seed)
    else:
        # Run the threaded execution
        threaded_exec(args.batch_range, args.batch_size, config_params, args.output_dir, prefix=args.prefix)

    logger.info("Model generation completed successfully")
