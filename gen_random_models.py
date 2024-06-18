import os
import sys

import argparse
import csv

import tensorflow as tf
import tensorflow.keras.utils as conv_utils

import json
import os
import random
import hashlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras import *
from qkeras.quantizers import quantized_bits


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


def main(args):
    dense_max = args.dense_max
    dense_step = args.dense_step
    prec_max = args.precision_max
    prec_step = args.precision_step

    # dense_in = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    # dense_out = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    # prec = [b for b in range(prec_step, prec_max + prec_step, prec_step)]
    # model_list = []
    # print(dense_in)
    # print(dense_out)
    # print(prec)
    # for d_in in dense_in:
    #     for d_out in dense_out:
    #         for p in prec:
    #             model_name = 'dense_{}_{}_{}b'.format(d_in, d_out, p)
    #             model_file = args.output+"/"+model_name+".h5"
    #             print(model_name)
    #             model = Sequential()
    #             model.add(
    #                 QDense(
    #                     d_out,
    #                     input_shape=(d_in,),
    #                     name=model_name,
    #                     kernel_quantizer=quantized_bits(p, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(p, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform',
    #                     kernel_regularizer=l1(0.0001),
    #                 )
    #             )
    #             adam = Adam(learning_rate=0.0001)
    #             model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    #             model.save(model_file)
    #             model_list.append([model_name, d_in, d_out, p, model_file])
    #             print("Generated {}".format(model_file))
    # # writing model list to csv file
    # with open(args.output+"/filelist.csv", 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["model_name", "d_in", "d_out", "prec", "model_file"])
    #     csvwriter.writerows(model_list)

    layer_subset = ['QDense', 'QConv2D', 'QActivation']
    max_layers = 10
    param_ranges = {
        'QDense': {'units': (10, dense_max), 'precision': (2, prec_max)},
        'QConv2D': {'filters': (10, 100), 'kernel_size': ((1, 1), (5, 5)), 'strides': ((1, 1), (2, 2)), 'precision': (2, prec_max)},
        'QActivation': {'activation': ['relu', 'sigmoid', 'tanh']}
    }

    random_model, random_model_json, random_model_name = create_random_model(layer_subset, max_layers, param_ranges, "Trial_model")
    print(random_model_name)
    print(random_model_json)
    print(random_model.summary())

def calculate_output_shape(layer_type, layer_config, input_shape):
    if layer_type == 'QDense':
        output_shape = (layer_config['units'],)
    elif layer_type == 'QConv2D':
        filters = layer_config['filters']
        kernel_size = layer_config['kernel_size']
        strides = layer_config.get('strides', (1, 1))
        padding = layer_config.get('padding', 'valid')

        print(input_shape)
        print(kernel_size)
        print(strides)

        if padding == 'same':
            output_height = input_shape[0] // strides[0]
            output_width = input_shape[1] // strides[1]
        else:  # 'valid' padding
            output_height = (input_shape[0] - kernel_size[0]) // strides[0] + 1
            output_width = (input_shape[1] - kernel_size[1]) // strides[1] + 1

        output_shape = (output_height, output_width, filters)
    else:
        output_shape = input_shape

    return output_shape
def create_qdense_layer(layer_config):
    return QDense(
        layer_config.get('units'),
        input_shape=layer_config.get('input_shape', None),
        name=layer_config.get('name', None),
        kernel_quantizer=quantized_bits(layer_config.get('precision', 8), 0, alpha=1),
        bias_quantizer=quantized_bits(layer_config.get('precision', 8), 0, alpha=1),
        kernel_initializer=layer_config.get('kernel_initializer', 'lecun_uniform'),
        kernel_regularizer=l1(layer_config.get('kernel_regularizer', 0.0001)),
    )

def create_qconv2d_layer(layer_config):
    return QConv2D(
        layer_config['filters'],
        layer_config['kernel_size'],
        name=layer_config.get('name', None),
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
        dilation_rate=layer_config.get('dilation_rate', (1, 1)),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qactivation_layer(layer_config):
    return QActivation(
        activation=layer_config.get('activation', None),
    )

def create_conv2d_layer(layer_config):
    return Conv2D(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
        dilation_rate=layer_config.get('dilation_rate', (1, 1)),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qdepthwiseconv2d_layer(layer_config):
    return QDepthwiseConv2D(
        layer_config['kernel_size'],
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        depth_multiplier=layer_config.get('depth_multiplier', 1),
        data_format=layer_config.get('data_format', None),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        depthwise_initializer=layer_config.get('depthwise_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        depthwise_regularizer=layer_config.get('depthwise_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        depthwise_constraint=layer_config.get('depthwise_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qseparableconv1d_layer(layer_config):
    return QSeparableConv1D(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', 1),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', 'channels_last'),
        dilation_rate=layer_config.get('dilation_rate', 1),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qseparableconv2d_layer(layer_config):
    return QSeparableConv2D(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
        dilation_rate=layer_config.get('dilation_rate', (1, 1)),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qconv2dtranspose_layer(layer_config):
    return QConv2DTranspose(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        output_padding=layer_config.get('output_padding', None),
        data_format=layer_config.get('data_format', None),
        dilation_rate=layer_config.get('dilation_rate', (1, 1)),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_qaveragepooling2d_layer(layer_config):
    return QAveragePooling2D(
        pool_size=layer_config.get('pool_size', (2, 2)),
        strides=layer_config.get('strides', None),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
    )

def create_qbatchnormalization_layer(layer_config):
    return QBatchNormalization(
        axis=layer_config.get('axis', -1),
        momentum=layer_config.get('momentum', 0.99),
        epsilon=layer_config.get('epsilon', 0.001),
        center=layer_config.get('center', True),
        scale=layer_config.get('scale', True),
        beta_initializer=layer_config.get('beta_initializer', 'zeros'),
        gamma_initializer=layer_config.get('gamma_initializer', 'ones'),
        moving_mean_initializer=layer_config.get('moving_mean_initializer', 'zeros'),
        moving_variance_initializer=layer_config.get('moving_variance_initializer', 'ones'),
        beta_regularizer=layer_config.get('beta_regularizer', None),
        gamma_regularizer=layer_config.get('gamma_regularizer', None),
        beta_constraint=layer_config.get('beta_constraint', None),
        gamma_constraint=layer_config.get('gamma_constraint', None),
    )

def create_maxpooling2d_layer(layer_config):
    return MaxPooling2D(
        pool_size=layer_config.get('pool_size', (2, 2)),
        strides=layer_config.get('strides', None),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
    )
def create_conv2dtranspose_layer(layer_config):
    return Conv2DTranspose(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', (1, 1)),
        padding=layer_config.get('padding', 'valid'),
        output_padding=layer_config.get('output_padding', None),
        data_format=layer_config.get('data_format', None),
        dilation_rate=layer_config.get('dilation_rate', (1, 1)),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )
def create_dropout_layer(layer_config):
    return Dropout(layer_config['rate'])

def create_batchnorm_layer(layer_config):
    return BatchNormalization(
        axis=layer_config.get('axis', -1),
        momentum=layer_config.get('momentum', 0.99),
        epsilon=layer_config.get('epsilon', 0.001),
        center=layer_config.get('center', True),
        scale=layer_config.get('scale', True),
        beta_initializer=layer_config.get('beta_initializer', 'zeros'),
        gamma_initializer=layer_config.get('gamma_initializer', 'ones'),
        moving_mean_initializer=layer_config.get('moving_mean_initializer', 'zeros'),
        moving_variance_initializer=layer_config.get('moving_variance_initializer', 'ones'),
        beta_regularizer=layer_config.get('beta_regularizer', None),
        gamma_regularizer=layer_config.get('gamma_regularizer', None),
        beta_constraint=layer_config.get('beta_constraint', None),
        gamma_constraint=layer_config.get('gamma_constraint', None),
    )

def create_conv1d_layer(layer_config):
    return Conv1D(
        layer_config['filters'],
        layer_config['kernel_size'],
        strides=layer_config.get('strides', 1),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', 'channels_last'),
        dilation_rate=layer_config.get('dilation_rate', 1),
        activation=layer_config.get('activation', None),
        use_bias=layer_config.get('use_bias', True),
        kernel_initializer=layer_config.get('kernel_initializer', 'glorot_uniform'),
        bias_initializer=layer_config.get('bias_initializer', 'zeros'),
        kernel_regularizer=layer_config.get('kernel_regularizer', None),
        bias_regularizer=layer_config.get('bias_regularizer', None),
        activity_regularizer=layer_config.get('activity_regularizer', None),
        kernel_constraint=layer_config.get('kernel_constraint', None),
        bias_constraint=layer_config.get('bias_constraint', None),
    )

def create_maxpooling1d_layer(layer_config):
    return MaxPooling1D(
        pool_size=layer_config.get('pool_size', 2),
        strides=layer_config.get('strides', None),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', 'channels_last'),
    )

def create_averagepooling2d_layer(layer_config):
    return AveragePooling2D(
        pool_size=layer_config.get('pool_size', (2, 2)),
        strides=layer_config.get('strides', None),
        padding=layer_config.get('padding', 'valid'),
        data_format=layer_config.get('data_format', None),
    )

def create_flatten_layer(layer_config):
    return Flatten(
        data_format=layer_config.get('data_format', None),
    )

def create_activation_layer(layer_config):
    return Activation(
        activation=layer_config.get('activation', None),
    )

def create_reshape_layer(layer_config):
    return Reshape(
        target_shape=layer_config.get('target_shape', (1,)),
        input_shape=layer_config.get('input_shape', None),
        name=layer_config.get('name', None),
    )
def create_model_from_json(json_string, model_ID, model_descriptor):
    # Parse the JSON string to get the model configuration
    model_config = json.loads(json_string)

    # Create a new Sequential model
    model = Sequential()

    # Dictionary of layer creation functions
    layer_functions = {
        'QDense': create_qdense_layer,
        'QConv2D': create_qconv2d_layer,
        'QActivation': create_qactivation_layer,
        'QDepthwiseConv2D': create_qdepthwiseconv2d_layer,
        'QSeparableConv1D': create_qseparableconv1d_layer,
        'QSeparableConv2D': create_qseparableconv2d_layer,
        'QConv2DTranspose': create_qconv2dtranspose_layer,
        'QAveragePooling2D': create_qaveragepooling2d_layer,
        'QBatchNormalization': create_qbatchnormalization_layer,
        'Conv2D': create_conv2d_layer,
        'MaxPooling2D': create_maxpooling2d_layer,
        'Dropout': create_dropout_layer,
        'BatchNormalization': create_batchnorm_layer,
        'Conv1D': create_conv1d_layer,
        'MaxPooling1D': create_maxpooling1d_layer,
        'Conv2DTranspose': create_conv2dtranspose_layer,
        'AveragePooling2D': create_averagepooling2d_layer,
        'Flatten': create_flatten_layer,
        'Activation': create_activation_layer,
        'Reshape': create_reshape_layer
    }

    # Iterate over each layer in the model configuration
    print(model_config)
    print(len(model_config))
    for layer_name, layer_config in model_config.items():
        # Get the function to create this type of layer
        print(layer_config)
        layer_function = layer_functions.get(layer_config['type'])

        if layer_function is None:
            raise ValueError('Unsupported layer type: ' + layer_config['type'])

        # Create the layer and add it to the model
        layer = layer_function(layer_config)
        model.add(layer)

    # Compile the model
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
    model.build()

    model_name = model_descriptor + '_' + model_ID

    print("Generated {}".format(model_name))

    return model, model_name



def create_random_model(layer_subset, max_layers, param_ranges, descriptor):
    # Create a new Sequential model
    model_json = {}

    # Generate a random number of layers
    num_layers = random.randint(2, max_layers)

    # Initialize input_shape to None
    input_shape = None

    # For each layer
    for i in range(num_layers):
        # Randomly select a layer type
        layer_type = random.choice(layer_subset)

        # Get the parameter range dictionary for this layer type
        param_range = param_ranges[layer_type]

        # set layer_name
        layer_name = layer_type + '_' + str(i)

        # Create a layer configuration dictionary
        layer_config = {'type': layer_type, 'name': layer_name}

        # For each parameter in the range dictionary
        for param, range_ in param_range.items():
            # Generate a random value within the specified range
            if isinstance(range_[0], int):
                value = random.randint(range_[0], range_[1])
            elif isinstance(range_[0], str):
                value = random.choice(range_)
            elif isinstance(range_[0], tuple):
                value = tuple(random.randint(r[0], r[1]) for r in range_)
            else:
                value = random.uniform(range_[0], range_[1])
            # Add the parameter to the layer configuration
            layer_config[param] = value
        print_dict(layer_config)
        print(layer_name)
        # Create the layer and add it to the model
        model_json.update({layer_name: layer_config})

        # If input_shape is None and the current layer is a 2DConv layer, generate a random input shape
        if input_shape is None and layer_type == 'QConv2D':
            input_shape = (random.randint(32, 256), random.randint(32, 256), random.randint(1, 3))
            layer_config['input_shape'] = input_shape
        elif input_shape is None:  # For other layer types, generate a single-dimension input shape
            input_shape = (random.randint(32, 256),)
            layer_config['input_shape'] = input_shape

        if input_shape is not None:
            layer_config['input_shape'] = input_shape

        model_json.update({layer_name: layer_config})

        print_dict(model_json)

    for i in range(num_layers - 1):
        current_layer_type = model_json[list(model_json.keys())[i]]['type']
        if i + 1 < len(list(model_json.keys())):  # Check if the next index is within the range
            next_layer_type = model_json[list(model_json.keys())[i + 1]]['type']
        else:
            continue

        if current_layer_type == 'QDense' and next_layer_type in ['QConv2D', 'QDepthwiseConv2D', 'QSeparableConv2D',
                                                                  'QConv2DTranspose']:
            layer_name = 'Reshape_' + str(i + 1)
            next_layer_input_shape = (output_shape[0], 1, 1) if len(output_shape) == 1 else output_shape
            layer_config = {'type': 'Reshape', 'name': layer_name, 'target_shape': next_layer_input_shape}
            model_json.update({layer_name: layer_config})

        if current_layer_type in ['QConv2D', 'QDepthwiseConv2D', 'QSeparableConv2D',
                                  'QConv2DTranspose'] and next_layer_type == 'QDense':
            layer_name = 'Flatten_' + str(i + 1)
            layer_config = {'type': 'Flatten', 'name': layer_name}
            model_json.update({layer_name: layer_config})

        # Calculate output shape for the next layer
        output_shape = calculate_output_shape(current_layer_type, model_json[list(model_json.keys())[i]],
                                              input_shape)
        input_shape = output_shape

    model_json = json.dumps(model_json)


    model_hash = hashlib.shake_256(model_json.encode("utf-8")).hexdigest(5)

    model, model_name = create_model_from_json(model_json, model_hash, descriptor)

    return model, model_json, model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, default='pregen_models')
    parser.add_argument('-p', '--precision_max', type=int, default=14)
    parser.add_argument('-s', '--precision_step', type=int, default=2)
    parser.add_argument('-d', '--dense_max', type=int, default=256)
    parser.add_argument('-i', '--dense_step', type=int, default=32)

    args = parser.parse_args()

    main(args)