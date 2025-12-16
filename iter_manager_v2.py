import argparse
import os
import sys
import json
import pandas as pd
from run_search_iteration import run_iter
from tensorflow.keras.models import model_from_json
from qkeras.utils import _add_supported_quantized_objects
import datetime

def main(args):
    """
    Main function to run the iteration manager with specified parameters.
    
    Args:
        args: Parsed command line arguments
    """
    rf_step = args.rf_step
    print(f"Running with RF Min: {args.rf_lower}, RF Max: {args.rf_upper}, RF Step: {rf_step}")
    
    if args.file.endswith('.csv'):
        filelist = pd.read_csv(args.file)
        print(f"Length of filelist: {len(filelist)}")
        
        for index, row in filelist[["model_name","config_str","prec"]].iterrows():
            model_name = row["model_name"]
            config_str = row["config_str"]
            prec = row["prec"]
            output_loc = args.output + row["model_name"]
            print(f'Starting run for {model_name} and precision {prec}')
            
            for rf in range(args.rf_lower, args.rf_upper + 1, rf_step):
                use_rf = 1 if rf == 0 else rf  # fix to let us start at 0 to get clean steps, but still do rf=1
                print("Running hls4ml Synth (vsynth: {}) for {} with RF of {}".format(args.vsynth, model_name, rf))
                run_iter(model_name, output_loc, use_rf, args.output, vsynth=args.vsynth, strat=args.hls4ml_strat, 
                         precision=prec, config_str=config_str, hlsproj=args.hlsproj)
    elif args.file.endswith('.json'):
        print("Found JSON File, loading...")
        with open(args.file, 'r') as file:
            co = {}
            _add_supported_quantized_objects(co)
            models = json.load(file)
            print(f"Length of models JSON: {len(models)}")
            
            for model_name, model_desc in models.items():
                model = model_from_json(model_desc, custom_objects=co)
                for rf in range(args.rf_lower, args.rf_upper, rf_step):
                    use_rf = 1 if rf == 0 else rf  # fix to let us start at 0 to get clean steps, but still do rf=1
                    print("Running hls4ml Synth (vsynth: {}) for {} with RF of {}".format(args.vsynth, model_name, rf))
                    run_iter(model_name, None, use_rf, args.output, vsynth=args.vsynth, strat=args.hls4ml_strat,
                             hlsproj=args.hlsproj, model=model, conv=args.conv)

def create_parser():
    """
    Create and configure the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='HLS4ML Search Iteration Manager')
    parser.add_argument('-f', '--file', type=str, default='pregen_3layer_models/filelist.csv',
                       help='Input file (CSV or JSON) containing model configurations')
    parser.add_argument('-u', '--rf_upper', type=int, default=1025,
                       help='Upper bound for reuse factor')
    parser.add_argument('-l', '--rf_lower', type=int, default=1,
                       help='Lower bound for reuse factor')
    parser.add_argument('-s', '--rf_step', type=int, default=512,
                       help='Step size for reuse factor')
    parser.add_argument('-o', '--output', type=str, default='./output',
                       help='Output directory')
    parser.add_argument("-p", "--prefix", type=str, default='/opt/repo/wa-hls4ml-search/',
                       help='Prefix path')
    parser.add_argument("-v", "--vsynth", action='store_true',
                       help='Enable vsynth')
    parser.add_argument("-h", "--hlsproj", type=str, default='/project/hls_proj/',
                       help='HLS project directory')
    parser.add_argument('--hls4ml_strat', type=str, default="Resource",
                       help='HLS4ML strategy')
    parser.add_argument("-c", "--conv", action='store_true',
                       help='Enable convolutional model mode (selects IO_Stream and other required settings for conv models)')
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
