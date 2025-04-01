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

clip_base_2 = lambda x: 2 ** round(np.log2(x))

class Model_Generator:
    failed_models = 0

    def __init__(self):
        self.reset_layers()

    def config_layer(self, layer_type: Layer) -> dict:
        activation = random.choices(self.activations, weights=self.params['probs']['activations'], k=1)[0]
        use_bias = random.random() < self.params['bias_rate']

        if layer_type in self.dense_layers:
            layer_size = clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub']))
            dropout = random.random() < self.params['dropout_chance']
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': self.params['dropout_rate']}
        elif layer_type in self.conv_layers:
            out_filters = clip_base_2(random.randint(3, 256))
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
            out_filters = clip_base_2(random.randint(3, 256))
            kernel_size = random.randint(self.params['conv_kernel_lb'], self.params['conv_kernel_ub'])
            flatten = random.random() < self.params['flatten_chance']
            stride = random.randint(self.params['conv_stride_lb'], self.params['conv_stride_ub'])
            padding = random.choices(['same', 'valid'], weights=self.params['probs']['padding'], k=1)[0]
            hyper_params = {'out_filters': out_filters, 'kernel': kernel_size,
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'padding': padding, 'stride': stride}
        return hyper_params

    def next_layer(self, last_layer: Layer, input_layer: Layer = None, pre_config: dict = None) -> Layer:
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
                    layer_choice.append(QAveragePooling2D()(layer_choice[-1]))
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
            else:
                if layer_type == LSTM:
                    raise NotImplemented
                elif layer_type == Conv1D:
                    layer_choice = [layer_type(filters=hyper_params['out_filters'], kernel_size=hyper_params['kernel'],
                                               strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'])(layer_choice[-1]))
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
        add_params = {k: add_params[k] for k in add_params}
        self.params = {'dense_lb': 32, 'dense_ub': 1024,
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
                           'padding': [0.5, 0.5],  # border, off
                           'pooling': [0.5, 0.5]  # max, avg
                       },
                       'activation_rate': .5,  # chances we apply activation function per layer
                       'dropout_chance': .5,  # chances dropout is on
                       'dropout_rate': .4,  # how much to dropout if dropout on
                       'flatten_chance': .5,
                       'pooling_chance': .5,
                       'bias_rate': .5}
        self.params.update(add_params)
        self.filter_q(self.params['q_chance'], self.params)
        init_layer = random.choices(self.start_layers, weights=self.params['probs']['start_layers'], k=1)[0]
        layer_units = 0
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
        try:
            layers = [Input(shape=input_shape)]
            self.params['last_layer_shape'] = layers[0].shape[1:]
            if init_layer in self.dense_layers:
                self.name = "dense"
            elif init_layer in self.conv_layers:
                self.name = "conv"
            elif init_layer in self.time_layers:
                self.name = "time"
            else:
                raise Exception("Layer not of a valid type")
            layers.extend(self.next_layer(init_layer, input_layer=layers[0]))
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
            return self.gen_network(total_layers=total_layers,
                                    add_params=add_params, callback=callback,
                                    save_file=save_file)

    def reset_layers(self) -> None:
        self.dense_layers = [Dense, QDense]
        self.conv_layers = [QConv2D, Conv2D, QSeparableConv2D, QDepthwiseConv2D]
        self.time_layers = [Conv1D, QConv1D]
        self.start_layers = [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]
        self.activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]
        self.layer_depth = 0

    def filter_q(self, q_chance: float, params: dict) -> None:
        blacklist = []
        self.q_on = random.random() < q_chance
        for layer in set(self.start_layers + self.conv_layers + self.dense_layers):
            is_qkeras = layer.__module__[:6] == 'qkeras'
            if self.q_on ^ is_qkeras:
                blacklist.append(layer)
        self.start_layers = [layer for layer in self.start_layers if layer not in blacklist]
        self.dense_layers = [layer for layer in self.dense_layers if layer not in blacklist]
        self.conv_layers = [layer for layer in self.conv_layers if layer not in blacklist]
        self.time_layers = [layer for layer in self.time_layers if layer not in blacklist]
        if self.q_on:
            if 'softmax' in self.activations:
                self.activations.remove('softmax')
            self.activations = [f'quantized_{activ_func}({params["activ_bit_width"]},{params["activ_int_width"]})' for
                                activ_func in self.activations]
        pairs = {'activations': self.activations, 'start_layers': self.start_layers, 'dense_layers': self.dense_layers,
                 'conv_layers': self.conv_layers, 'time_layers': self.time_layers}
        for param_type in pairs:
            if not params['probs'][param_type]:
                params['probs'][param_type] = [1 / len(pairs[param_type]) for _ in pairs[param_type]]

    def load_models(self, save_file: str) -> list[Model]:
        with open(save_file, "r") as chunk_file:
            models = chunk_file.read().split("--------------")[:-1]
            for model_desc in models:
                co = {}
                _add_supported_quantized_objects(co)
                yield model_from_json(model_desc, custom_objects=co)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

clip_base_2 = lambda x: 2 ** round(np.log2(x))

class Model_Generator:
    failed_models = 0

    def __init__(self):
        self.reset_layers()

    def config_layer(self, layer_type: Layer) -> dict:
        activation = random.choices(self.activations, weights=self.params['probs']['activations'], k=1)[0]
        use_bias = random.random() < self.params['bias_rate']

        if layer_type in self.dense_layers:
            layer_size = clip_base_2(random.randint(self.params['dense_lb'], self.params['dense_ub']))
            dropout = random.random() < self.params['dropout_chance']
            hyper_params = {'size': layer_size, 'activation': activation, 'use_bias': use_bias,
                            'dropout': dropout, 'dropout_rate': self.params['dropout_rate']}
        elif layer_type in self.conv_layers:
            out_filters = clip_base_2(random.randint(3, 256))
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
            out_filters = clip_base_2(random.randint(3, 256))
            kernel_size = random.randint(self.params['conv_kernel_lb'], self.params['conv_kernel_ub'])
            flatten = random.random() < self.params['flatten_chance']
            stride = random.randint(self.params['conv_stride_lb'], self.params['conv_stride_ub'])
            padding = random.choices(['same', 'valid'], weights=self.params['probs']['padding'], k=1)[0]
            hyper_params = {'out_filters': out_filters, 'kernel': kernel_size,
                            'flatten': flatten, 'activation': activation, 'use_bias': use_bias,
                            'padding': padding, 'stride': stride}
        return hyper_params

    def next_layer(self, last_layer: Layer, input_layer: Layer = None, pre_config: dict = None) -> Layer:
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
                    layer_choice.append(QAveragePooling2D()(layer_choice[-1]))
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
            else:
                if layer_type == LSTM:
                    raise NotImplemented
                elif layer_type == Conv1D:
                    layer_choice = [layer_type(filters=hyper_params['out_filters'], kernel_size=hyper_params['kernel'],
                                               strides=hyper_params['stride'],
                                               use_bias=hyper_params['use_bias'], padding=hyper_params['padding'])(last_layer)]
                if "no_activation" not in hyper_params['activation']:
                    layer_choice.append(Activation(activation=hyper_params['activation'])(layer_choice[-1]))
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
        add_params = {k: add_params[k] for k in add_params}
        self.params = {'dense_lb': 32, 'dense_ub': 1024,
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
                           'padding': [0.5, 0.5],  # border, off
                           'pooling': [0.5, 0.5]  # max, avg
                       },
                       'activation_rate': .5,  # chances we apply activation function per layer
                       'dropout_chance': .5,  # chances dropout is on
                       'dropout_rate': .4,  # how much to dropout if dropout on
                       'flatten_chance': .5,
                       'pooling_chance': .5,
                       'bias_rate': .5,
                       'layers_blacklist': []}
        self.params.update(add_params)
        self.filter_q(self.params['q_chance'], self.params)
        init_layer = random.choices(self.start_layers, weights=self.params['probs']['start_layers'], k=1)[0]
        layer_units = 0
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
        try:
            layers = [Input(shape=input_shape)]
            self.params['last_layer_shape'] = layers[0].shape[1:]
            if init_layer in self.dense_layers:
                self.name = "dense"
            elif init_layer in self.conv_layers:
                self.name = "conv"
            elif init_layer in self.time_layers:
                self.name = "time"
            else:
                raise Exception("Layer not of a valid type")
            layers.extend(self.next_layer(init_layer, input_layer=layers[0]))
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
        except (RayTaskError, Exception) as e:
            self.failed_models += 1
            self.reset_layers()
            raise e

    def reset_layers(self) -> None:
        self.dense_layers = [Dense, QDense]
        self.conv_layers = [QConv2D, Conv2D, QSeparableConv2D, QDepthwiseConv2D]
        self.time_layers = [Conv1D, QConv1D]
        self.start_layers = [Conv1D, QConv1D, Conv2D, QConv2D, QDense, Dense, QSeparableConv2D, QDepthwiseConv2D]
        self.activations = ["no_activation", "relu", "tanh", "sigmoid", "softmax"]
        self.layer_depth = 0

    def filter_q(self, q_chance: float, params: dict) -> None:
        blacklist = params['layers_blacklist']
        self.q_on = random.random() < q_chance
        for layer in set(self.start_layers + self.conv_layers + self.dense_layers):
            is_qkeras = layer.__module__[:6] == 'qkeras'
            if self.q_on ^ is_qkeras:
                blacklist.append(layer)
        self.start_layers = [layer for layer in self.start_layers if layer not in blacklist]
        self.dense_layers = [layer for layer in self.dense_layers if layer not in blacklist]
        self.conv_layers = [layer for layer in self.conv_layers if layer not in blacklist]
        self.time_layers = [layer for layer in self.time_layers if layer not in blacklist]
        if self.q_on:
            if 'softmax' in self.activations:
                self.activations.remove('softmax')
            self.activations = [f'quantized_{activ_func}({params["activ_bit_width"]},{params["activ_int_width"]})' for
                                activ_func in self.activations]
        pairs = {'activations': self.activations, 'start_layers': self.start_layers, 'dense_layers': self.dense_layers,
                 'conv_layers': self.conv_layers, 'time_layers': self.time_layers}
        for param_type in pairs:
            if not params['probs'][param_type]:
                params['probs'][param_type] = [1 / len(pairs[param_type]) for _ in pairs[param_type]]

    def load_models(self, save_file: str) -> list[Model]:
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
def generate_model(bitwidth, mg):
    try:
        model = mg.gen_network(add_params={'dense_lb': 32, 'dense_ub': 128, 'conv_filters_ub': 32,
                                           'q_chance': 1, 'flatten_chance': .1, 'pooling_chance': .3,
                                           'weight_bit_width': bitwidth, 'weight_int_width': 1,
                                           'activ_bit_width': bitwidth, 'activ_int_width': 1,
                                           'probs': {'activations': [],
                                                     'dense_layers': [.25], 'conv_layers': [.25, .25, .25], 'time_layers': [],
                                                     'start_layers': [],
                                                     'padding': [0.5, 0.5],  # border, off
                                                     'pooling': [0.2, 0.2]  # max, avg
                                                     }
                                           },
                               total_layers=random.randint(3, 10), save_file=None)
        return model.name, model.to_json()
    except Exception as e:
        mg.failed_models += 1
        raise(e)



def main():
    failed_models = 0
    glob_t = time.time()
    batch_range = 4
    batch_size = 2
    succeeded = 0

    for batch_i in tqdm(range(batch_range), desc="Batch Count:"):
        models = []
        mg = Model_Generator()
        futures = [generate_model.remote(2 ** random.randint(2, 4), mg) for _ in range(batch_size)]
        for future in ray.get(futures):
            model_name, model_json = future
            if model_name and model_json:
                models.append(model_json)
                succeeded +=1
        json_models = json.dumps(models, indent=None, separators=(',', '\n'))
        with open(f"conv2d_batch_{batch_i}.json", "w") as file:
            file.write(json_models)
        print(succeeded)
        print(len(models))

if __name__ == '__main__':
    main()