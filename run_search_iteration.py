import os
import tarfile
import shutil
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import argparse
import json
from deperecated.gen_dense_models_v2 import generate_model_from_config
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

def run_iter(name = "model",  model_file = '/project/model.h5', rf=1, output = "/output", part = 'xcu250-figd2104-2L-e', hlsproj = '/project/hls_proj', vsynth=True, strat="latency", precision=None, config_str=None, model=None, conv=False):
    if model is not None: # if model is passed in directly, use it
        print("Using passed in model")
        model.summary()
    elif config_str is None and model_file is not None: # load model from file,
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(model_file, custom_objects=co)
        model.summary()
    elif config_str is not None and precision is not None:
        # else generate model from config string
        # this method is deperecated, but kept for compatibility with older RF scans
        model = generate_model_from_config(config_str, precision, output_dir=".", save_model=False)
        model.summary()
    else:
        raise ValueError("Must specify either model or config_str with correct parameters. ")

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


    if os.path.exists(processed_json_name):
        print(processed_json_name + " Already exists, skipping...")
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
    print(f'Output directory: {hls_dir}, Strategy: {strat}, Part: {part}, backend: {hls4ml_backend}')
    if conv:
        cfg_q = hls4ml.converters.create_config(backend=hls4ml_backend)
        cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
        cfg_q['HLSConfig'] = config
        cfg_q['KerasModel'] = model
        cfg_q['OutputDir'] = hls_dir
        cfg_q['Part'] = part

        hls_model = hls4ml.converters.keras_to_hls(cfg_q)
    else:
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=hls_dir, part=part, backend=hls4ml_backend
        )

    print("compile hls model")
    hls_model.write()


    #hls_model.compile()
    print(f"{name}_rf{rf} - vsynth enabled: {vsynth}")
    hls_model.build(csim=False, vsynth=vsynth, cosim=False, export=False)

    # read the report and just save that?
    print("Reading report...")
    report_json = hls4ml.report.vivado_report.parse_vivado_report(hls_dir)
    #hls4ml.report.read_vivado_report(hls_dir)

    with open(json_name, "w") as outfile:
        json.dump(report_json, outfile)

    processed_json, model_uuid = process_json_entry(model, config, json_name, part=part)
    with open(processed_json_name, "w") as outfile:
        json.dump(processed_json, outfile)

    make_tarfile(output + "/projects/" + model_uuid + ".tar.gz", hls_dir)

    print("Finished running hls4ml synthesis for ", name, " with RF of ", rf)
    print("Deleting HLS Project Directory: ", hls_dir)
    shutil.rmtree(hls_dir)
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
