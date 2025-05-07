import os
import sys
import json
import tarfile
import shutil
import argparse
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from keras_parser import config_from_keras_model
import gzip
import subprocess

from tqdm import tqdm

def extract_target_file(artifacts_path, target_filename, extract_to="./"):
    try:
        with tarfile.open(artifacts_path, "r:gz") as tar:
            member = tar.getmember(target_filename)
            tar.extract(member, path=extract_to)
            return os.path.join(extract_to, member.name)
    except KeyError:
        print(f"Error: {target_filename} not found in {artifacts_path}.")
    except Exception as e:
        print(f"Error extracting {artifacts_path}: {e}")
    return None

def process_json_directory(json_dir, output_dir=None):
    keras_models_dir = os.path.join(json_dir, "keras_models")
    os.makedirs(keras_models_dir, exist_ok=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    with tqdm(total=len(json_files), desc="Processing JSON files", unit="file") as pbar:
        for filename in json_files:
            json_path = os.path.join(json_dir, filename)
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            # Get the reuse factor from the filename
            try:
                reuse_factor = int(filename.split("_rf")[1].split("_")[0])
            except (IndexError, ValueError):
                print(f"Skipping {filename}: Could not extract reuse factor from filename.")
                pbar.update(1)
                continue

            # Get the artifacts_file entry
            artifacts_file = data.get("meta_data", {}).get("artifacts_file")
            if not artifacts_file:
                print(f"Skipping {filename}: No artifacts_file found in meta_data.")
                pbar.update(1)
                continue

            # Extract the keras_model.keras file
            model_filename = filename.replace(f"_processed.json", "") + "/keras_model.keras"
            artifacts_path = os.path.join(json_dir, "projects", artifacts_file)
            extracted_model_path = extract_target_file(artifacts_path, model_filename)
            if not extracted_model_path:
                print(f"Skipping {filename}: keras_model.keras not found in {artifacts_file}.")
                pbar.update(1)
                continue


            # Save the extracted keras_model.keras to the keras_models directory
            new_model_name = filename.replace(f"_rf{reuse_factor}_processed.json", ".keras")
            new_model_path = os.path.join(keras_models_dir, new_model_name)
            if not os.path.exists(new_model_path):
                try:
                    shutil.move(extracted_model_path, new_model_path)
                except Exception as e:
                    print(f"Error saving model to {new_model_path}: {e}")
                    pbar.update(1)
                    continue


            # Load the model with qkeras custom_objects
            try:
                co = {}
                _add_supported_quantized_objects(co)
                model = load_model(new_model_path, custom_objects=co)
            except Exception as e:
                print(f"Error loading model {new_model_path}: {e}")
                pbar.update(1)
                continue

            # Parse the model to get updated model_config
            try:
                updated_model_config = config_from_keras_model(model, reuse_factor)
            except Exception as e:
                print(f"Error parsing model {new_model_path}: {e}")
                pbar.update(1)
                continue

            # Replace the model_config in the JSON file
            data["model_config"] = updated_model_config

            # Save the updated JSON file
            if output_dir:
                updated_json_path = os.path.join(output_dir, filename)
                with open(updated_json_path, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                #print(f"Processed and saved updated {filename} to {output_dir}.")
            else:
                with open(json_path, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                #print(f"Processed and updated {filename}.")

            pbar.update(1)

def tar_and_gzip_directory(source_dir, tar_output_path, use_pigz=False):
    if use_pigz:
        # Use pigz for parallel compression, if available
        tar_command = f"tar -cf - -C {os.path.dirname(source_dir)} {os.path.basename(source_dir)} | pigz -p {os.cpu_count()} > {tar_output_path}"
        subprocess.run(tar_command, shell=True, check=True)
    else:
        # Use tarfile with gzip for single-threaded compression
        with tarfile.open(tar_output_path, "w") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        with open(tar_output_path, "rb") as f_in:
            with gzip.open(tar_output_path + ".gz", "wb", compresslevel=1) as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(tar_output_path)  # Remove the uncompressed tar file

    print(f"Directory {source_dir} has been tarred and gzipped to {tar_output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and optionally save to a different directory.")
    parser.add_argument("json_directory", help="Path to the directory containing JSON files.")
    parser.add_argument("--output-dir", help="Directory to save updated JSON files instead of overwriting.")
    parser.add_argument("--tar-output", help="Path to save the tarred and gzipped output directory.")

    args = parser.parse_args()

    if not os.path.isdir(args.json_directory):
        print(f"Error: {args.json_directory} is not a valid directory.")
        sys.exit(1)

    process_json_directory(args.json_directory, args.output_dir)

    if args.output_dir and args.tar_output:
        tar_and_gzip_directory(args.output_dir, args.tar_output)