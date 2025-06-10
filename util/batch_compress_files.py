import os
import tarfile
import json
import csv
import subprocess
from math import ceil
from tqdm import tqdm
import argparse

def validate_master_csv(input_directory, master_csv_path):
    """
    Validate that all artifact files from the JSON files are present in the master CSV.

    Parameters:
        input_directory (str): Path to the folder containing the JSON files.
        master_csv_path (str): Path to the master CSV file to validate.
    """
    # Extract artifact files from JSON files
    artifact_files_from_json = set()
    json_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]

    for json_file in json_files:
        json_path = os.path.join(input_directory, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                artifact_file = data.get("meta_data", {}).get("artifacts_file")
                if artifact_file:
                    artifact_files_from_json.add((json_file, artifact_file))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping invalid or malformed JSON file: {json_file}")

    # Extract artifact files from the master CSV
    artifact_files_from_csv = set()
    if os.path.exists(master_csv_path):
        with open(master_csv_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)  # Skip header
            for row in reader:
                artifact_files_from_csv.add((row[0], row[1]))

    # Find missing artifact files
    missing_files = artifact_files_from_json - artifact_files_from_csv

    if missing_files:
        print("The following artifact files are missing from the master CSV:")
        for json_file, artifact_file in missing_files:
            print(f"JSON File: {json_file}, Artifact File: {artifact_file}")
    else:
        print("All artifact files are present in the master CSV.")


def compress_files_from_json(input_directory, output_directory, files_per_archive, master_csv_path, use_pigz, pigz_cores):
    """
    Compress files specified in JSON files into multiple tar.gz files and update the master CSV file.

    Parameters:
        input_directory (str): Path to the folder containing the JSON files.
        output_directory (str): Path to the folder where the tar.gz files will be saved.
        files_per_archive (int): Number of files to include in each tar.gz archive.
        master_csv_path (str): Path to the master CSV file to be updated.
        use_pigz (bool): Whether to use pigz for compression.
        pigz_cores (int): Number of cores to use with pigz.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Determine the starting archive number
    existing_archives = [
        f for f in os.listdir(output_directory) if f.startswith("archive_") and f.endswith(".tar.gz")
    ]
    if existing_archives:
        max_archive_number = max(
            int(f.split("_")[1].split(".")[0]) for f in existing_archives
        )
    else:
        max_archive_number = 0

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

    print(f"Found {len(json_files)} JSON files in the input directory.")

    # Extract artifact file paths from JSON files
    artifact_files = []
    total_files = 0  # Counter for all files, including already processed ones

    for json_file in json_files:
        json_path = os.path.join(input_directory, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                artifact_file = data.get("meta_data", {}).get("artifacts_file")
                if artifact_file:
                    total_files += 1  # Count all files, even if already processed
                    if (json_file, artifact_file) not in processed_files:
                        artifact_files.append((json_file, artifact_file))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping invalid or malformed JSON file: {json_file}")

    # Calculate the number of archives based on unprocessed files
    num_archives = ceil(len(artifact_files) / files_per_archive)

    print(f"Total files found: {total_files}")
    print(f"Total files to compress: {len(artifact_files)}")
    print(f"Number of archives to create: {num_archives}")

    # Compress files into tar.gz archives
    for i in tqdm(range(num_archives), desc="Creating archives"):
        archive_number = max_archive_number + i + 1
        archive_name = os.path.join(output_directory, f"archive_{archive_number}.tar.gz")
        start_index = i * files_per_archive
        end_index = min(start_index + files_per_archive, total_files)
        files_to_compress = artifact_files[start_index:end_index]

        archive_csv_data = []  # Temporary list to store data for this archive
        tar_path = os.path.join(output_directory, f"archive_{archive_number}.tar")

        # Create the tar archive
        with tarfile.open(tar_path, "w") as tar:
            for json_file, artifact_file in tqdm(files_to_compress, desc=f"Adding files to {tar_path}", leave=False):
                artifact_path = os.path.join(input_directory, "projects", artifact_file)
                if os.path.exists(artifact_path):
                    tar.add(artifact_path, arcname=os.path.basename(artifact_file))
                    archive_csv_data.append([json_file, artifact_file, archive_name])
                else:
                    print(f"File not found: {artifact_file}")

        # Compress the tar archive using pigz or gzip
        if use_pigz:
            pigz_command = ["pigz", "-f", f"-p{pigz_cores}", tar_path]
            subprocess.run(pigz_command, check=True)
        else:
            with open(tar_path, "rb") as f_in, open(archive_name, "wb") as f_out:
                subprocess.run(["gzip", "-c"], stdin=f_in, stdout=f_out, check=True)
        #os.remove(tar_path)  # Remove the tar file after compression

        # Write the archive's data to the master CSV after the archive is created
        with open(master_csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if os.stat(master_csv_path).st_size == 0:  # Add header if the file is empty
                writer.writerow(["JSON File", "Artifact File", "Archive Name"])
            writer.writerows(archive_csv_data)

        #print(f"Created archive: {archive_name} with {len(files_to_compress)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress files from JSON metadata into tar.gz archives.")
    parser.add_argument("input_directory", type=str, help="Path to the input folder containing JSON files.")
    parser.add_argument("output_directory", type=str, help="Path to the output folder where archives will be saved.")
    parser.add_argument("files_per_archive", type=int, help="Number of files to include in each archive.")
    parser.add_argument("master_csv_path", type=str, help="Path to the master CSV file to be updated.")
    parser.add_argument("--use-pigz", action="store_true", help="Use pigz for compression.")
    parser.add_argument("--pigz-cores", type=int, default=1, help="Number of cores to use with pigz (default: 1).")

    args = parser.parse_args()

    compress_files_from_json(
        args.input_directory,
        args.output_directory,
        args.files_per_archive,
        args.master_csv_path,
        args.use_pigz,
        args.pigz_cores
    )
    validate_master_csv(args.input_directory, args.master_csv_path)