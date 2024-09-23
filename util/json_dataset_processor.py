import os
import json
import uuid
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
from rule4ml.parsers.network_parser import config_from_keras_model
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

def process_json_files(input_dir, output_file, model_dir):
    processed_data = []

    for filename in os.listdir(input_dir):
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
                co = {}
                _add_supported_quantized_objects(co)
                model = load_model(model_path, custom_objects=co)

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

if __name__ == "__main__":
    model_dir = '../pregen_2layer_models'
    input_dir = '/home/bhawks/2layer_run_vsynth_9-23'
    output_file = 'processed_data.json'

    process_json_files(input_dir, output_file, model_dir)