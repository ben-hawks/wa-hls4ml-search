import os
import sys
import json
import uuid
import csv
from tqdm import tqdm
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
from rule4ml.parsers.network_parser import config_from_keras_model
from gen_dense_models_v2 import generate_model_from_config
try:
    os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
    os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
    hls4ml_backend = 'Vitis'
except KeyError:
    try:
        os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
        hls4ml_backend = 'Vivado'
    except KeyError:
        hls4ml_backend = 'Vivado'

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def process_json_files(input_dir, output_file, model_dir, filelist=None):
    processed_data = []

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith("_report.json"):
            file_path = os.path.join(input_dir, filename)
            model_name, rf_str = filename.split('_rf')
            reuse_factor = int(rf_str.split('_')[0])
            model_uuid = str(uuid.uuid4())
            model_path = os.path.join(model_dir, model_name +".h5")
            #print(f"Processing {filename}")
            try:
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                if filelist is None:
                    co = {}
                    _add_supported_quantized_objects(co)
                    model = load_model(model_path, custom_objects=co)
                else:
                    config_str = model_name.split('_')[-1]
                    precision = int(config_str.split('b')[0])
                    print(config_str)
                    model = generate_model_from_config(config_str[:-1], precision, output_dir=".", save_model=False)
                    model.summary()
                model_config = config_from_keras_model(model, reuse_factor)

                meta_data = {
                    'uuid': model_uuid,
                    'model_name': model_name,
                    'artifacts_file': model_uuid + ".tar.gz"
                }
                latency_report = {
                    'cycles_min': json_data['CSynthesisReport']['BestLatency'],
                    'cycles_max': json_data['CSynthesisReport']['WorstLatency'],
                    'target_clock': json_data['CSynthesisReport']['TargetClockPeriod'],
                    'estimated_clock': json_data['CSynthesisReport']['EstimatedClockPeriod'],
                }
                resource_report = {
                    'bram': json_data['VivadoSynthReport']['BRAM_18K'],
                    'dsp': json_data['VivadoSynthReport']['DSP48E'],
                    'ff': json_data['VivadoSynthReport']['FF'],
                    'lut': json_data['VivadoSynthReport']['LUT'],
                    'uram': json_data['VivadoSynthReport']['URAM'],
                }
                with HiddenPrints(): # suppress the output of hls4ml when opening and parsing the model
                    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=hls4ml_backend)
                    hls_config['Model']['ReuseFactor'] = reuse_factor
                    hls_config['Model']['Strategy'] = "Resource"
                    for layer in hls_config['LayerName'].keys():
                        if 'dense' in layer.lower() or 'conv' in layer.lower():
                            hls_config['LayerName'][layer]['ReuseFactor'] = reuse_factor

                processed_entry = {
                    'meta_data': meta_data,
                    'model_config': model_config,
                    'hls_config': hls_config,
                    'resource_report': resource_report,
                    'latency_report': latency_report,
                    'target_part': "xcu250-figd2104-2L-e",
                    'vivado_version':"2020.1",
                    'hls4ml_version': "0.8.1",
                }

                processed_data.append(processed_entry)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)

def load_filelist(filelist_path):
    filelist = {}
    with open(filelist_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            model_name = row[0]
            config_str = row[1]
            precision = int(row[-2])
            filelist[model_name] = (config_str, precision)
    return filelist

def process_json_files_filelist(input_dir, output_file, filelist_path):
    processed_data = []
    filelist = load_filelist(filelist_path)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith("_report.json"):
            file_path = os.path.join(input_dir, filename)
            model_name, rf_str = filename.split('_rf')
            reuse_factor = int(rf_str.split('_')[0])
            model_uuid = str(uuid.uuid4())
            config_str, precision = filelist[model_name]

            try:
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)

                model = generate_model_from_config(config_str, precision, output_dir=".", save_model=False)
                model.summary()
                model_config = config_from_keras_model(model, reuse_factor)

                meta_data = {
                    'uuid': model_uuid,
                    'model_name': model_name,
                    'artifacts_file': model_uuid + ".tar.gz"
                }
                latency_report = {
                    'cycles_min': json_data['CSynthesisReport']['BestLatency'],
                    'cycles_max': json_data['CSynthesisReport']['WorstLatency'],
                    'target_clock': json_data['CSynthesisReport']['TargetClockPeriod'],
                    'estimated_clock': json_data['CSynthesisReport']['EstimatedClockPeriod'],
                }
                resource_report = {
                    'bram': json_data['VivadoSynthReport']['BRAM_18K'],
                    'dsp': json_data['VivadoSynthReport']['DSP48E'],
                    'ff': json_data['VivadoSynthReport']['FF'],
                    'lut': json_data['VivadoSynthReport']['LUT'],
                    'uram': json_data['VivadoSynthReport']['URAM'],
                }
                with HiddenPrints(): # suppress the output of hls4ml when opening and parsing the model
                    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=hls4ml_backend)
                    hls_config['Model']['ReuseFactor'] = reuse_factor
                    hls_config['Model']['Strategy'] = "Resource"
                    for layer in hls_config['LayerName'].keys():
                        if 'dense' in layer.lower() or 'conv' in layer.lower():
                            hls_config['LayerName'][layer]['ReuseFactor'] = reuse_factor

                processed_entry = {
                    'meta_data': meta_data,
                    'model_config': model_config,
                    'hls_config': hls_config,
                    'resource_report': resource_report,
                    'latency_report': latency_report,
                    'target_part': "xcu250-figd2104-2L-e",
                    'vivado_version':"2020.1",
                    'hls4ml_version': "0.8.1",
                }

                processed_data.append(processed_entry)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, 'w') as outfile:
        json.dump(processed_data, outfile, indent=4)



def process_json_entry(model, hls_config, filename, part="xcu250-figd2104-2L-e", vivado_version="2020.1", hls4ml_version="0.8.1"):
    processed_data = None
    model_uuid = None
    if filename.endswith("_report.json"):
        file_path = filename
        model_name, rf_str = filename.split('_rf')
        reuse_factor = int(rf_str.split('_')[0])
        model_uuid = str(uuid.uuid4())

        try:
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)

            model_config = config_from_keras_model(model, reuse_factor)

            meta_data = {
                'uuid': model_uuid,
                'model_name': model_name,
                'artifacts_file': model_uuid + ".tar.gz"
            }
            latency_report = {
                'cycles_min': json_data['CSynthesisReport']['BestLatency'],
                'cycles_max': json_data['CSynthesisReport']['WorstLatency'],
                'target_clock': json_data['CSynthesisReport']['TargetClockPeriod'],
                'estimated_clock': json_data['CSynthesisReport']['EstimatedClockPeriod'],
            }
            resource_report = {
                'bram': json_data['VivadoSynthReport']['BRAM_18K'],
                'dsp': json_data['VivadoSynthReport']['DSP48E'],
                'ff': json_data['VivadoSynthReport']['FF'],
                'lut': json_data['VivadoSynthReport']['LUT'],
                'uram': json_data['VivadoSynthReport']['URAM'],
            }
            processed_entry = {
                'meta_data': meta_data,
                'model_config': model_config,
                'hls_config': hls_config,
                'resource_report': resource_report,
                'latency_report': latency_report,
                'target_part': part,
                'vivado_version':vivado_version,
                'hls4ml_version': hls4ml_version,
            }

            processed_data = processed_entry
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return processed_data, model_uuid

if __name__ == "__main__":
    model_dir = '../pregen_2layer_models'
    input_dir = '/home/bhawks/2layer_run_vsynth_9-25'
    output_file = 'wa-hls4ml_processed_vsynth_data.json'

    process_json_files(input_dir, output_file, model_dir)