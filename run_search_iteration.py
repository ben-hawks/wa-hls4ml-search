import os
import sys
import tarfile

from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import argparse
import yaml
import json

os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

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
    run_iter(args.name, args.model, args.rf, args.output, args.part, args.hlsproj, False, args.hls4ml_strat)

def run_iter(name = "model",  model_file = '/project/model.h5', rf=1, output = "/output", part = 'xcu250-figd2104-2L-e', hlsproj = '/project/hls_proj', vsynth=False, strat="latency"):
    co = {}
    _add_supported_quantized_objects(co)
    model = load_model(model_file, custom_objects=co)

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(os.path.join(output, "projects")):
        os.makedirs(os.path.join(output, "projects"))

    json_name = output+"/"+name+"_rf"+str(rf)+"_report.json"
    if os.path.exists(json_name):
        print(json_name + " Already exists, skipping...")
        return

    hls_dir = hlsproj + "/" + name + "_rf" + str(rf)
    if not os.path.exists(hls_dir):
        os.makedirs(hls_dir)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['ReuseFactor'] = rf
    config['Model']['Strategy'] = strat
    print("-----------------------------------")
    print_dict(config)
    print("-----------------------------------")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=hls_dir, part=part
    )

    print("compile hls model")
    hls_model.write()
    hls_model.compile()
    hls_model.build(csim=False, vsynth=vsynth)

    # read the report and just save that?
    report_json = hls4ml.report.vivado_report.parse_vivado_report(hls_dir)
    hls4ml.report.read_vivado_report(hls_dir)

    make_tarfile(output+"/projects/"+name+"_rf"+str(rf)+".tar.gz", hls_dir)


    with open(json_name, "w") as outfile:
        json.dump(report_json, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='model')
    parser.add_argument('-m', '--model', type=str, default='/project/model.h5')
    parser.add_argument('-p', '--part', type=str, default='xcu250-figd2104-2L-e')
    parser.add_argument('-o', '--output', type=str, default='/output')
    parser.add_argument('-h', '--hlsproj', type=str, default='/project/hls_proj/')
    parser.add_argument('-r', '--rf', type=int, default=1)
    parser.add_argument( '--hls4ml_strat', type=str, default="Resource")

    args = parser.parse_args()

    main(args)
