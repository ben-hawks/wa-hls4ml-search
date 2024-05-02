import argparse
import os
import sys
import pandas as pd
from run_search_iteration import run_iter

def main(args):
    rf_step = args.rf_step-1
    filelist = pd.read_csv(args.file)
    print(filelist[["model_name","model_file"]])
    for index, row in filelist[["model_name","model_file"]].iterrows():
        model_name = row["model_name"]
        model_file = args.prefix + row["model_file"]
        for rf in range(args.rf_lower, args.rf_upper, rf_step):
            print("Running hls4ml C-Synth for {} with RF of {}".format(model_file, rf))
            run_iter(model_name, model_file, rf, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='pregen_models/filelist.csv')
    parser.add_argument('-u', '--rf_upper', type=int, default=1024)
    parser.add_argument('-l', '--rf_lower', type=int, default=1)
    parser.add_argument('-s', '--rf_step', type=int, default=512)
    parser.add_argument('-o', '--output', type=str, default='output/')
    parser.add_argument("-p", "--prefix", type=str, default='/opt/repo/wa-hls4ml-search/')

    args = parser.parse_args()

    main(args)