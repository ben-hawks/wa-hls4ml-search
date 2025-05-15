import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
from qkeras import QDense, QConv2D, QConv1D, QAveragePooling2D, QActivation, quantized_bits, QDepthwiseConv2D, QSeparableConv2D, QSeparableConv1D, QLSTM
from keras.layers import Dense, Conv2D, Flatten, Activation, Conv1D, LSTM, Layer, Input
from keras.models import Model, model_from_json
from qkeras.utils import _add_supported_quantized_objects
import keras
import typing
import random
import time
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import sys, os
import ray
from ray.exceptions import RayTaskError

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#clip_base_2 = lambda x: 2 ** round(np.log2(x))
clip_base_2 = lambda x: 2 ** max(1, round(np.log2(max(1, x))))

class Model_Generator:
    failed_models = 0

    def __init__(self):
        self.reset_layers()

    def config_layer(self, layer_type: Layer) -> dict:
        """
        Returns hyper parameters for layer initialization as a dict

        arguments:
        layer_type -- takes in the selection of layer so it can specify
        """
        # Ensure weights match the population
        #if len(self.params['probs']['activations']) != len(self.activations):
        #    self.params['probs']['activations'] = [1 / len(self.activations)] * len(self.activations)
        print(self.activations)
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
        Takes previous layer and configuration displays and returns back layer
        
        arguments:
        last_layer -- previous keras/qkeras layer
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
                    raise NotImplemented
                
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(QActivation(activation=hyper_params['activation'])(layer_choice[-1]))
                if hyper_params['pooling']:
                    pooling = random.choices([keras.layers.MaxPooling1D, keras.layers.AveragePooling1D],
                                             weights=self.params['probs']['pooling'], k=1)[0]
                    layer_choice.append(pooling(2)(layer_choice[-1]))
            else:
                if layer_type == LSTM:
                    raise NotImplemented
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
        Generates interconnected network based on defaults or extra params, returns Model

        keyword arguments:
        total_layers -- total active layers in a network (default: 3)
        add_params -- parameters to specify besides defaults for model generation (default: {})
        q_chance -- the prob that we use qkeras over keras
        save_file -- open file descriptor for log file (default: None)
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
            'dropout_chance': .5,
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
            logging.error(f"Error generating network: {e}")
            return self.gen_network(total_layers=total_layers,
                                    add_params=add_params, callback=callback,
                                    save_file=save_file)

    def reset_layers(self) -> None:
        """
        Used to return class to initial state. Useful if generating multiple networks
        """
        self.dense_layers = [Dense, QDense]
        self.conv_layers = [QConv2D, Conv2D, QSeparableConv2D, QDepthwiseConv2D]
        self.time_layers = [Conv1D, QConv1D]
        self.start_layers = [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]

        self.activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]
        
        self.layer_depth = 0

    def filter_q(self, q_chance: float, params: dict) -> None:
        blacklist = []
        self.q_on = random.random() < q_chance

        # filter out the qkeras/non-qkeras layers
        for layer in set(self.start_layers + self.conv_layers + self.dense_layers):
            is_qkeras = layer.__module__[:6] == 'qkeras'
            if self.q_on ^ is_qkeras:
                blacklist.append(layer)
        self.start_layers = [layer for layer in self.start_layers if layer not in blacklist]
        self.dense_layers = [layer for layer in self.dense_layers if layer not in blacklist]
        self.conv_layers = [layer for layer in self.conv_layers if layer not in blacklist]
        self.time_layers = [layer for layer in self.time_layers if layer not in blacklist]

        # adjust activation layers based on quantization
        if self.q_on:
            if 'softmax' in self.activations:
                self.activations.remove('softmax')
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
        Parses and returns an iterable of generated models
        
        arguments:
        save_file -- path to batch of models
        """

        with open(save_file, "r") as chunk_file:
            models = chunk_file.read().split("--------------")[:-1]
            for model_desc in models:
                co = {}
                _add_supported_quantized_objects(co)
                yield model_from_json(model_desc, custom_objects=co)

ray.init(num_cpus=os.cpu_count(), log_to_driver=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@ray.remote(max_retries=10, retry_exceptions=True)
def generate_model(bitwidth):
    try:
        mg = Model_Generator() # Latency strategy Dense jobs
        model = mg.gen_network(add_params={'dense_lb': 8, 'dense_ub': 128, 'conv_filters_ub': 16,
                                           'conv_init_size_lb': 8, 'conv_init_size_ub': 64,
                                           'q_chance': 1, 'flatten_chance': .1, 'pooling_chance': .3,
                                           'weight_bit_width': bitwidth, 'weight_int_width': 1,
                                           'activ_bit_width': bitwidth, 'activ_int_width': 1,
                                           'activation_rate': 1,
                                           'probs': {'activations': [.30,.30,.30,.10],
                                                     # Activations: ["relu", "tanh", "sigmoid", "softmax"]
                                                     # Must set probabilities for the layers in start_layers as well!
                                                    # conv layers
                                                    # q_chance = 0 [Conv2D]
                                                    # q_chance < 1 [Conv2D, QConv2D, QSeparableConv2D, QDepthwiseConv2D]
                                                    # q_chance = 1 [QConv2D, QSeparableConv2D, QDepthwiseConv2D]
                                                    # Dense/Time are either qkeras or not in line with q_chance,
                                                    # 1 element if 0/1, else 2 elements
                                                     'dense_layers': [1], 'conv_layers': [0, 0, 0], 'time_layers': [0],
                                                     # start layers
                                                     #q_chance = 0 [Conv1D, Conv2D, Dense]
                                                     #q_chance < 1 [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]
                                                     #q_chance = 1 [QConv1D, QConv2D, QDense, QSeparableConv2D, QDepthwiseConv2D]
                                                     'start_layers': [0, 0, 1, 0, 0],
                                                     'padding': [0.5, 0.5],  # border, off
                                                     'pooling': [0.2, 0.2]  # max, avg
                                                     }
                                           },
                               total_layers=random.randint(3, 7), save_file=None)
        return model.name, model.to_json()
    except Exception as e:
        raise(e)

def threaded_exec(batch_range: int, batch_size: int):
    succeeded = 0

    assert batch_range > 0
    assert batch_size > 0
    for batch_i in tqdm(range(batch_range), desc="Batch Count:"):
        model_dict = {}
        futures = [generate_model.remote(2 ** random.randint(2, 4)) for _ in range(batch_size)]
        for future in ray.get(futures):
            model_name, model_json = future
            if model_name and model_json:
                # model_name might have dupes because of multithreading, so make a new name for each model
                model_dict.update({f"dense_resource_{succeeded}": model_json})  # Store the model with its name
                succeeded += 1
        json_models = json.dumps(model_dict)
        with open(f"dense_resource_models/dense_resource_batch_{batch_i}.json", "w") as file:
            file.write(json_models)


if __name__ == '__main__':
    batch_range = 334
    batch_size = 50
    threaded_exec(batch_range, batch_size)
    
    # left this here as an example but everything beyond this line in __name__ can be deleted
    def callback(mg: Model_Generator, layers: list):
        if mg.layer_depth > 1:
            mg.params['flatten_chance'] = 1