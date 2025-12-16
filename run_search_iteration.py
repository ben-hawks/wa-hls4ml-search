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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _setup_hls4ml_backend():
    """
    Setup the HLS4ML backend based on available environment variables.
    
    Returns:
        str: The backend name ('Vitis' or 'Vivado')
    """
    try:
        os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
        os.environ['PATH'] = os.environ['XILINX_VITIS'] + '/bin:' + os.environ['PATH']
        hls4ml_backend = 'Vitis'
        logger.info("Using Vitis backend")
    except KeyError:
        try:
            os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']
            hls4ml_backend = 'Vivado'
            logger.info("Using Vivado backend")
        except KeyError:
            hls4ml_backend = 'Vivado'
            logger.warning("No Vivado or Vitis environment variables found, using default Vivado backend")
    
    return hls4ml_backend

def make_tarfile(output_filename, source_dir):
    """
    Create a compressed tar file from a directory.
    
    Args:
        output_filename (str): Path to output tar file
        source_dir (str): Directory to compress
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def print_dict(d, indent=0):
    """
    Print a dictionary in a formatted way.
    
    Args:
        d (dict): Dictionary to print
        indent (int): Indentation level
    """
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def _create_directories(output, hlsproj, name, rf):
    """
    Create necessary directories for the run.
    
    Args:
        output (str): Output directory
        hlsproj (str): HLS project directory
        name (str): Model name
        rf (int): Reuse factor
    """
    # Ensure parent directories exist
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, "projects"), exist_ok=True)
    
    json_name = os.path.join(output, "raw_reports", f"{name}_rf{rf}_report.json")
    processed_json_name = os.path.join(output, f"{name}_rf{rf}_processed.json")
    
    os.makedirs(os.path.dirname(json_name), exist_ok=True)
    os.makedirs(os.path.dirname(processed_json_name), exist_ok=True)
    
    hls_dir = os.path.join(hlsproj, f"{name}_rf{rf}")
    os.makedirs(hls_dir, exist_ok=True)
    
    return json_name, processed_json_name, hls_dir

def _load_model(model_file, config_str, precision, model):
    """
    Load or generate a model based on parameters.
    
    Args:
        model_file (str): Path to model file
        config_str (str): Configuration string
        precision (int): Precision value
        model: Direct model object
        
    Returns:
        Model: Loaded or generated model
    """
    if model is not None:  # if model is passed in directly, use it
        logger.info("Using passed in model")
        model.summary()
    elif config_str is None and model_file is not None:  # load model from file
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(model_file, custom_objects=co)
        model.summary()
    elif config_str is not None and precision is not None:
        # else generate model from config string
        # this method is deprecated, but kept for compatibility with older RF scans
        logger.info("Generating model from config string")
        model = generate_model_from_config(config_str, precision, output_dir=".", save_model=False)
        model.summary()
    else:
        raise ValueError("Must specify either model or config_str with correct parameters. ")
    
    return model

def run_iter(name="model", model_file='/project/model.h5', rf=1, output="/output", part='xcu250-figd2104-2L-e', 
             hlsproj='/project/hls_proj', vsynth=True, strat="latency", precision=None, config_str=None, 
             model=None, conv=False):
    """
    Run HLS4ML iteration for a single model.
    
    Args:
        name (str): Model name
        model_file (str): Path to model file
        rf (int): Reuse factor
        output (str): Output directory
        part (str): Target part
        hlsproj (str): HLS project directory
        vsynth (bool): Enable vsynth
        strat (str): HLS strategy
        precision (int): Precision value
        config_str (str): Configuration string
        model: Direct model object
        conv (bool): Enable convolutional processing
    """
    try:
        # Setup backend
        hls4ml_backend = _setup_hls4ml_backend()
        
        # Load model
        model = _load_model(model_file, config_str, precision, model)
        
        # Create directories
        json_name, processed_json_name, hls_dir = _create_directories(output, hlsproj, name, rf)
        
        # Check if already processed
        if os.path.exists(processed_json_name):
            logger.info(f"{processed_json_name} Already exists, skipping...")
            return
        
        logger.info(f"Creating directory: {hls_dir}")
        
        # Configure HLS
        config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend=hls4ml_backend)
        config['Model']['ReuseFactor'] = rf
        config['Model']['Strategy'] = strat
        for layer in config['LayerName'].keys():
            if 'dense' in layer.lower() or 'conv' in layer.lower():
                config['LayerName'][layer]['ReuseFactor'] = rf
        
        logger.info("-----------------------------------")
        print_dict(config)
        logger.info("-----------------------------------")
        logger.info(f'Output directory: {hls_dir}, Strategy: {strat}, Part: {part}, backend: {hls4ml_backend}')
        
        # Convert model
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

        logger.info("Compiling HLS model")
        hls_model.write()

        logger.info(f"{name}_rf{rf} - vsynth enabled: {vsynth}")
        hls_model.build(csim=False, vsynth=vsynth, cosim=False, export=False)

        # Read the report and save it
        logger.info("Reading report...")
        report_json = hls4ml.report.vivado_report.parse_vivado_report(hls_dir)

        with open(json_name, "w") as outfile:
            json.dump(report_json, outfile)

        processed_json, model_uuid = process_json_entry(model, config, json_name, part=part)
        with open(processed_json_name, "w") as outfile:
            json.dump(processed_json, outfile)

        make_tarfile(os.path.join(output, "projects", f"{model_uuid}.tar.gz"), hls_dir)

        logger.info(f"Finished running hls4ml synthesis for {name} with RF of {rf}")
        logger.info(f"Deleting HLS Project Directory: {hls_dir}")
        shutil.rmtree(hls_dir)
        
    except Exception as e:
        logger.error(f"Error processing {name} with RF {rf}: {str(e)}")
        raise

def main(args):
    """
    Main function to run the search iteration.
    
    Args:
        args: Parsed command line arguments
    """
    run_iter(args.name, args.model, args.rf, args.output, args.part, args.hlsproj, args.vsynth, args.hls4ml_strat)

def create_parser():
    """
    Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='HLS4ML Search Iteration Runner')
    parser.add_argument('-n', '--name', type=str, default='model',
                       help='Model name')
    parser.add_argument('-m', '--model', type=str, default='/project/model.h5',
                       help='Path to model file')
    parser.add_argument('-p', '--part', type=str, default='xcu250-figd2104-2L-e',
                       help='Target part')
    parser.add_argument('-o', '--output', type=str, default='/output',
                       help='Output directory')
    parser.add_argument('-h', '--hlsproj', type=str, default='/project/hls_proj/',
                       help='HLS project directory')
    parser.add_argument('-r', '--rf', type=int, default=1,
                       help='Reuse factor')
    parser.add_argument("-v", "--vsynth", action='store_true',
                       help='Enable vsynth')
    parser.add_argument('--hls4ml_strat', type=str, default="Resource",
                       help='HLS4ML strategy')
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
