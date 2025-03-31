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
    dense_layers = args.dense_layers

    dense_in = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    dense_out = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    if dense_layers == 2:
        dense_out2 = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    prec = [b for b in range(prec_step, prec_max + prec_step, prec_step)]
    model_list = []
    print(dense_in)
    print(dense_out)
    print(prec)
    for d_in in dense_in:
        for d_out in dense_out:
            if dense_layers == 2:
                for d_out2 in dense_out2:
                    for p in prec:
                        model_name = 'dense_{}_{}_{}_{}b'.format(d_in, d_out,d_out2, p)
                        model_file = args.output+"/"+model_name+".h5"
                        if os.path.exists(model_file):
                            print("Model {} already exists, skipping...".format(model_name))
                            continue
                        print(model_name)
                        model = Sequential()
                        model.add(
                            QDense(
                                d_out,
                                input_shape=(d_in,),
                                name=model_name+"_dense_1",
                                kernel_quantizer=quantized_bits(p, 0, alpha=1),
                                bias_quantizer=quantized_bits(p, 0, alpha=1),
                                kernel_initializer='lecun_uniform',
                                kernel_regularizer=l1(0.0001),
                            )
                        )
                        model.add(
                            QDense(
                                d_out2,
                                name=model_name+"_dense_2",
                                kernel_quantizer=quantized_bits(p, 0, alpha=1),
                                bias_quantizer=quantized_bits(p, 0, alpha=1),
                                kernel_initializer='lecun_uniform',
                                kernel_regularizer=l1(0.0001),
                            )
                        )
                        adam = Adam(learning_rate=0.0001)
                        model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
                        model.save(model_file)
                        model_list.append([model_name, d_in, d_out, d_out2, p, model_file])
                        print("Generated {}".format(model_file))
            else:
                model_name = 'dense_{}_{}_{}b'.format(d_in, d_out, p)
                model_file = args.output + "/" + model_name + ".h5"
                if os.path.exists(model_file):
                    print("Model {} already exists, skipping...".format(model_name))
                    continue
                print(model_name)
                model = Sequential()
                model.add(
                    QDense(
                        d_out,
                        input_shape=(d_in,),
                        name=model_name + "_dense_1",
                        kernel_quantizer=quantized_bits(p, 0, alpha=1),
                        bias_quantizer=quantized_bits(p, 0, alpha=1),
                        kernel_initializer='lecun_uniform',
                        kernel_regularizer=l1(0.0001),
                    )
                )
                adam = Adam(learning_rate=0.0001)
                model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
                model.save(model_file)
                model_list.append([model_name, d_in, d_out, p, model_file])
                print("Generated {}".format(model_file))
    # writing model list to csv file
    with open(args.output+"/filelist.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        if dense_layers == 2:
            csvwriter.writerow(["model_name", "d_in", "d_out", "d_out2", "prec", "model_file"])
        else:
            csvwriter.writerow(["model_name", "d_in", "d_out", "prec", "model_file"])
        csvwriter.writerows(model_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, default='pregen_models')
    parser.add_argument('-p', '--precision_max', type=int, default=16)
    parser.add_argument('-s', '--precision_step', type=int, default=2)
    parser.add_argument('-d', '--dense_max', type=int, default=64)
    parser.add_argument('-i', '--dense_step', type=int, default=2)
    parser.add_argument('-l', '--dense_layers', type=int, default=1)

    args = parser.parse_args()

    main(args)