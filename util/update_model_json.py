import os
import sys
import json
import tarfile
import shutil
import argparse
import gzip
import subprocess
from concurrent.futures import ProcessPoolExecutor
from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from keras_parser import config_from_keras_model
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

def process_single_json_file(args):
    json_path, json_dir, keras_models_dir, output_dir = args
    try:
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        filename = os.path.basename(json_path)
        reuse_factor = int(filename.split("_rf")[1].split("_")[0])

        artifacts_file = data.get("meta_data", {}).get("artifacts_file")
        if not artifacts_file:
            return f"Skipping {filename}: No artifacts_file found in meta_data."

        base_model_filename = filename.replace(f"_processed.json", "")
        model_filenames = [f"{base_model_filename}/keras_model.keras", f"{base_model_filename}/keras_model.h5"]
        extracted_model_path = None
        artifacts_path = os.path.join(json_dir, "projects", artifacts_file)

        for model_filename in model_filenames:
            extracted_model_path = extract_target_file(artifacts_path, model_filename)
            if extracted_model_path:
                break

        if not extracted_model_path:
            return f"Skipping {filename}: keras_model file not found in {artifacts_file}."

        new_model_name = filename.replace(f"_rf{reuse_factor}_processed.json", os.path.splitext(extracted_model_path)[1])
        new_model_path = os.path.join(keras_models_dir, new_model_name)
        if not os.path.exists(new_model_path):
            shutil.move(extracted_model_path, new_model_path)

        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(new_model_path, custom_objects=co)

        updated_model_config = config_from_keras_model(model, reuse_factor)
        data["model_config"] = updated_model_config

        if output_dir:
            updated_json_path = os.path.join(output_dir, filename)
            with open(updated_json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        else:
            with open(json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

        return f"Processed {filename} successfully."
    except Exception as e:
        return f"Error processing {os.path.basename(json_path)}: {e}"

def process_json_directory(json_dir, output_dir=None, max_cores=None):
    keras_models_dir = os.path.join(json_dir, "keras_models")
    os.makedirs(keras_models_dir, exist_ok=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        args = [(json_path, json_dir, keras_models_dir, output_dir) for json_path in json_files]
        results = list(tqdm(executor.map(process_single_json_file, args), total=len(json_files), desc="Processing JSON files", unit="file"))

    for result in results:
        print(result)

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
    parser.add_argument("--max-cores", type=int, help="Maximum number of CPU cores to use for parallel processing.")

    args = parser.parse_args()

    if not os.path.isdir(args.json_directory):
        print(f"Error: {args.json_directory} is not a valid directory.")
        sys.exit(1)

    process_json_directory(args.json_directory, args.output_dir, args.max_cores)

    if args.output_dir and args.tar_output:
        tar_and_gzip_directory(args.output_dir, args.tar_output)


