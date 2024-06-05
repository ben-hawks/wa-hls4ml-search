import os
import sys

import argparse
import yaml
import json
import csv
import pandas as pd

synth_types = {"hls":"CSynthesisReport", "rtl":"SynthesisReport"} #TODO Validate name of rtl synth object

import hls4ml_rf_finder

def main(args):
    success_count = {K: 0 for K in synth_types}
    file_count = 0
    with open(args.base, "r") as b:
        with open(args.output, "a") as o:
            first_row = True
            fieldnames = []
            rows_to_write = []
            base_reader = csv.DictReader(b)
            rf_step = args.rf_step - 1
            for row in base_reader:
                for rf in range(args.rf_lower, args.rf_upper, rf_step):
                    rf_actual = hls4ml_rf_finder.set_closest_reuse_factor(rf, int(row["d_in"]), int(row["d_out"]))
                    rf_dict = {"rf":rf_actual, "strategy":args.hls4ml_strat}
                    try:
                        filename = row["model_name"] + "_rf" + str(rf) + "_report.json"
                        with open(os.path.join(args.input, filename)) as f:
                            file_count +=1
                            synth_report = {}
                            data = json.load(f)
                            if bool(data): #check if dict is empty, indicates failed synth run if so
                                for st in args.synth.split(","):
                                    synth_report_specific = data[synth_types[st]]
                                    if bool(synth_report_specific):
                                        synth_report_specific = {key+"_"+st:float(val) for key, val in synth_report_specific.items()}
                                        synth_report.update(synth_report_specific)
                                        rf_dict.update({st + "_synth_success": True})
                                        success_count[st] += 1
                                    else:
                                        rf_dict.update({st + "_synth_success": False})

                            else:
                                for st in args.synth.split(","):
                                    rf_dict.update({st + "_synth_success": False})
                            new_row = dict(**row, **rf_dict, **synth_report)
                            current_fieldnames = (new_row.keys())
                            # update fieldnames to include anything new (since rtl synth ensures hls synth happened, dont have to worry about missing fields)
                            if len(current_fieldnames) > len(fieldnames):
                                fieldnames = current_fieldnames
                            rows_to_write.append(new_row)
                    except Exception as e:
                        print(e)
            out_writer = csv.DictWriter(o, fieldnames=fieldnames)
            out_writer.writeheader()
            out_writer.writerows(rows_to_write)

    synth_success = {s: 0 for s in synth_types.keys()}
    print("Successful Synth Runs: ")
    print(success_count)
    print("File Count: ",file_count)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='../results/results_format_test.csv')
    parser.add_argument('-t', '--type', type=str, default='csv')
    parser.add_argument('-i', '--input', type=str, default='../results')
    parser.add_argument('-s', '--synth', type=str, default='hls')
    parser.add_argument('-b', '--base', type=str, default='results/base.csv')
    parser.add_argument('-u', '--rf_upper', type=int, default=1025)
    parser.add_argument('-l', '--rf_lower', type=int, default=1)
    parser.add_argument('-r', '--rf_step', type=int, default=512)
    parser.add_argument( '--hls4ml_strat', type=str, default="resource")

    args = parser.parse_args()

    main(args)
