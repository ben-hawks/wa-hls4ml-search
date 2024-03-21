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

    dense_in = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    dense_out = [n for n in range(dense_step, dense_max + dense_step, dense_step)]
    prec = [b for b in range(prec_step, prec_max + prec_step, prec_step)]
    model_list = []
    print(dense_in)
    print(dense_out)
    print(prec)
    for d_in in dense_in:
        for d_out in dense_out:
            for p in prec:
                model_name = 'dense-{}-{}-{}b'.format(d_in, d_out, p)
                model_file = args.output+"/"+model_name+".h5"
                print(model_name)
                model = Sequential()
                model.add(
                    QDense(
                        d_out,
                        input_shape=(d_in,),
                        name=model_name,
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
        csvwriter.writerow(["model_name", "d_in", "d_out", "prec", "model_file"])
        csvwriter.writerows(model_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, default='pregen_models')
    parser.add_argument('-p', '--precision_max', type=int, default=12)
    parser.add_argument('-s', '--precision_step', type=int, default=2)
    parser.add_argument('-d', '--dense_max', type=int, default=256)
    parser.add_argument('-i', '--dense_step', type=int, default=32)

    args = parser.parse_args()

    main(args)