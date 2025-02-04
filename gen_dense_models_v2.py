import os
import sys
import argparse
import csv
import tensorflow as tf
import tensorflow.keras.utils as conv_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation
from qkeras.qlayers import QDense
from qkeras.quantizers import quantized_bits
from tqdm import tqdm
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def generate_layer_configs(dense_units, num_layers, input_size):
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    if num_layers == 1:
        return [[input_size] + [units] for units in dense_units]
    else:
        configs = []
        for units in dense_units:
            for sub_config in generate_layer_configs(dense_units, num_layers - 1, input_size):
                configs.append([input_size] + [units] + sub_config[1:])
        return configs

def generate_model_from_config(config_str, precision, output_dir=".", save_model=False):
    layer_config = list(map(int, config_str.split('_')))
    input_size = layer_config[0]
    model_name = 'dense_{}_{}b'.format(config_str, precision)
    model_file = os.path.join(output_dir, model_name + ".h5")
    print(layer_config)
    if os.path.exists(model_file) and save_model:
        co = {}
        _add_supported_quantized_objects(co)
        print("Model {} already exists, loading...".format(model_name))
        return load_model(model_file, custom_objects=co)

    print("Generating model: {}".format(model_name))
    model = Sequential()
    model.add(
        QDense(
            layer_config[1],
            input_shape=(input_size,),
            name=model_name + "_dense_1",
            kernel_quantizer=quantized_bits(precision, 0, alpha=1),
            bias_quantizer=quantized_bits(precision, 0, alpha=1),
            kernel_initializer='lecun_uniform',
            kernel_regularizer=l1(0.0001),
        )
    )
    for i, units in enumerate(layer_config[2:], start=2):
        model.add(
            QDense(
                units,
                name=model_name + "_dense_" + str(i),
                kernel_quantizer=quantized_bits(precision, 0, alpha=1),
                bias_quantizer=quantized_bits(precision, 0, alpha=1),
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
            )
        )
    adam = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])

    if save_model:
        model.save(model_file)
        print("Generated and saved model: {}".format(model_name))
    else:
        print("Generated model without saving (hls4ml saves model to project dir): {}".format(model_name))

    return model

def main(args):
    dense_max = args.dense_max
    dense_step = args.dense_step
    prec_max = args.precision_max
    prec_step = args.precision_step
    dense_layers = args.dense_layers
    input_size = args.input_size if args.input_size else dense_step
    save_models = args.save_models if hasattr(args, 'save_models') else False

    dense_units = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    prec = [b for b in range(prec_step, prec_max + prec_step, prec_step)]
    model_list = []

    for d_in in tqdm(dense_units):
        for p in prec:
            for layer_config in generate_layer_configs(dense_units, dense_layers, d_in):
                config_str = '_'.join(map(str, layer_config))
                model_name = 'dense_{}_{}b'.format(config_str, p)
                model_file = os.path.join(args.output, model_name + ".h5")
                if save_models:
                    if os.path.exists(model_file):
                        print("Model {} already exists, skipping...".format(model_name))
                        continue
                    model = generate_model_from_config(config_str, p, d_in, args.output, save_model=True)
                    print("Generated {}".format(model_file))
                model_list.append([model_name, config_str] + layer_config + [p, model_file])

    # writing model list to csv file
    with open(os.path.join(args.output, "filelist.csv"), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["model_name", "config_str", "input_size"] + [f"d_out{i}" for i in range(1, dense_layers + 1)] + ["prec", "model_file"]
        csvwriter.writerow(header)
        csvwriter.writerows(model_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, default='pregen_models')
    parser.add_argument('-p', '--precision_max', type=int, default=16)
    parser.add_argument('-s', '--precision_step', type=int, default=2)
    parser.add_argument('-d', '--dense_max', type=int, default=64)
    parser.add_argument('-i', '--dense_step', type=int, default=2)
    parser.add_argument('-l', '--dense_layers', type=int, default=1)
    parser.add_argument('-n', '--input_size', type=int)
    parser.add_argument('--save_models', action='store_true')

    args = parser.parse_args()

    main(args)