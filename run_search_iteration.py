import os
import sys
import tarfile

from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import argparse
import yaml
import json
from gen_dense_models_v2 import generate_model_from_config
from util.json_dataset_processor import process_json_entry

try:
    os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
    os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
    hls4ml_backend = 'Vitis'
except KeyError:
    os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
    hls4ml_backend = 'Vivado'

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
    run_iter(args.name, args.model, args.rf, args.output, args.part, args.hlsproj, args.vsynth, args.hls4ml_strat)

def run_iter(name = "model",  model_file = '/project/model.h5', rf=1, output = "/output", part = 'xcu250-figd2104-2L-e', hlsproj = '/project/hls_proj', vsynth=True, strat="latency", precision=None, config_str=None):

    if config_str is None: # load model from file, else generate model from config string
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(model_file, custom_objects=co)
    else:
        model = generate_model_from_config(config_str, precision, output_dir=".", save_model=False)
        model.summary()

    json_name = output+"/raw_reports/"+name+"_rf"+str(rf)+"_report.json"
    processed_json_name = output+"/"+name+"_rf"+str(rf)+"_processed.json"

    # Ensure parent directories exist
    if not os.path.exists(output):
        print(output)
        os.makedirs(output)

    if not os.path.exists(os.path.join(output, "projects")):
        os.makedirs(os.path.join(output, "projects"))

    if not os.path.exists(os.path.dirname(json_name)):
        os.makedirs(os.path.dirname(json_name))

    if not os.path.exists(os.path.dirname(processed_json_name)):
        os.makedirs(os.path.dirname(processed_json_name))


    if os.path.exists(json_name):
        print(json_name + " Already exists, skipping...")
        return

    hls_dir = os.path.join(hlsproj,name+"_rf"+str(rf))
    if not os.path.exists(hls_dir):
        print("Creating directory: ", hls_dir)
        os.makedirs(hls_dir)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=hls4ml_backend)
    config['Model']['ReuseFactor'] = rf
    config['Model']['Strategy'] = strat
    for layer in config['LayerName'].keys():
        if 'dense' in layer.lower() or 'conv' in layer.lower():
            config['LayerName'][layer]['ReuseFactor'] = rf
    print("-----------------------------------")
    print_dict(config)
    print("-----------------------------------")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=hls_dir, part=part, backend=hls4ml_backend
    )

    print("compile hls model")
    hls_model.write()


    #hls_model.compile()
    print(name, " - vsynth enabled: ", vsynth)
    hls_model.build(csim=False, vsynth=vsynth, cosim=False, export=False)

    # read the report and just save that?
    report_json = hls4ml.report.vivado_report.parse_vivado_report(hls_dir)
    hls4ml.report.read_vivado_report(hls_dir)

    with open(json_name, "w") as outfile:
        json.dump(report_json, outfile)

    processed_json, model_uuid = process_json_entry(model, config, report_json)
    with open(processed_json_name, "w") as outfile:
        json.dump(processed_json, outfile)

    make_tarfile(output + "/projects/" + model_uuid + ".tar.gz", hls_dir)

    print("Finished running hls4ml synthesis for ", name, " with RF of ", rf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='model')
    parser.add_argument('-m', '--model', type=str, default='/project/model.h5')
    parser.add_argument('-p', '--part', type=str, default='xcu250-figd2104-2L-e')
    parser.add_argument('-o', '--output', type=str, default='/output')
    parser.add_argument('-h', '--hlsproj', type=str, default='/project/hls_proj/')
    parser.add_argument('-r', '--rf', type=int, default=1)
    parser.add_argument("-v", "--vsynth", action='store_true')
    parser.add_argument( '--hls4ml_strat', type=str, default="Resource")

    args = parser.parse_args()

    main(args)
