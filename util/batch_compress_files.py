import os
import tarfile
import json
import csv
from math import ceil
from tqdm import tqdm
import argparse


def compress_files_from_json(input_directory, output_directory, files_per_archive, master_csv_path):
    """
    Compress files specified in JSON files into multiple tar.gz files and update the master CSV file.

    Parameters:
        input_directory (str): Path to the folder containing the JSON files.
        output_directory (str): Path to the folder where the tar.gz files will be saved.
        files_per_archive (int): Number of files to include in each tar.gz archive.
        master_csv_path (str): Path to the master CSV file to be updated.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load already processed files from the master CSV
    processed_files = set()
    if os.path.exists(master_csv_path):
        with open(master_csv_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)  # Skip header
            for row in reader:
                processed_files.add((row[0], row[1]))  # (JSON File, Artifact File)

    # Get a list of all JSON files in the input directory
    json_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]

    # Extract artifact file paths from JSON files
    artifact_files = []
    for json_file in json_files:
        json_path = os.path.join(input_directory, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                artifact_file = data.get("meta_data", {}).get("artifact_file")
                if artifact_file and (json_file, artifact_file) not in processed_files:
                    artifact_files.append((json_file, artifact_file))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping invalid or malformed JSON file: {json_file}")

    total_files = len(artifact_files)
    num_archives = ceil(total_files / files_per_archive)

    # Compress files into tar.gz archives
    for i in tqdm(range(num_archives), desc="Creating archives"):
        archive_name = os.path.join(output_directory, f"archive_{i + 1}.tar.gz")
        start_index = i * files_per_archive
        end_index = min(start_index + files_per_archive, total_files)
        files_to_compress = artifact_files[start_index:end_index]

        with tarfile.open(archive_name, "w:gz") as tar:
            for json_file, artifact_file in tqdm(files_to_compress, desc=f"Adding files to {archive_name}", leave=False):
                if os.path.exists(os.path.join(input_directory, "projects", artifact_file)):
                    tar.add(os.path.join(input_directory, "projects", artifact_file), arcname=os.path.basename(artifact_file))
                    # Append to the master CSV file
                    with open(master_csv_path, "a", newline="") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([json_file, artifact_file, archive_name])
                else:
                    print(f"File not found: {artifact_file}")

        print(f"Created archive: {archive_name} with {len(files_to_compress)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress files from JSON metadata into tar.gz archives.")
    parser.add_argument("input_directory", type=str, help="Path to the input folder containing JSON files.")
    parser.add_argument("output_directory", type=str, help="Path to the output folder where archives will be saved.")
    parser.add_argument("files_per_archive", type=int, help="Number of files to include in each archive.")
    parser.add_argument("master_csv_path", type=str, help="Path to the master CSV file to be updated.")

    args = parser.parse_args()

    compress_files_from_json(args.input_directory, args.output_directory, args.files_per_archive, args.master_csv_path)