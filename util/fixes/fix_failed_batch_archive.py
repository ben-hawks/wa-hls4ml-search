import os
import tarfile
import pandas as pd
import argparse
import json

def remake_failed_archive_no_dependency(input_directory, output_directory, files_per_archive, master_csv_path, failed_archive):
    """
    Remake a failed batched archive by referencing the master CSV file and including artifact files
    from JSON files not present in the master CSV.

    Parameters:
        input_directory (str): Path to the folder containing the files.
        output_directory (str): Path to the folder where the tar.gz files are saved.
        files_per_archive (int): Number of files to include in each archive.
        master_csv_path (str): Path to the master CSV file.
        failed_archive (str): Name of the failed archive to remake.
    """
    failed_archive_path = os.path.join(output_directory, failed_archive)

    if not os.path.exists(failed_archive_path):
        print(f"Error: Failed archive '{failed_archive}' does not exist in '{output_directory}'.")
        return

    print(f"Remaking failed archive: {failed_archive}")

    # Remove the failed archive
    os.remove(failed_archive_path)

    # Ensure the remade archive does not overwrite an existing file
    remade_archive_path = failed_archive_path
    counter = 1
    while os.path.exists(remade_archive_path):
        base_name, ext = os.path.splitext(failed_archive)
        remade_archive_path = os.path.join(output_directory, f"{base_name}_remade_{counter}{ext}")
        counter += 1

    # Determine files to include in the archive by referencing the master CSV
    try:
        master_csv = pd.read_csv(master_csv_path)
        master_files = set(master_csv['Artifact File'].tolist())
        files_to_include = master_csv[master_csv['Archive Name'] == failed_archive]['Artifact File'].tolist()

        if files_to_include is None:    # Add artifact files from JSON files not present in the master CSV
            for file in os.listdir(input_directory):
                if file.endswith(".json"):
                    json_path = os.path.join(input_directory, file)
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        artifact_file = data.get("meta_data", {}).get("artifacts_file", {})
                        if artifact_file and artifact_file not in master_files:
                            files_to_include.append(artifact_file)

        print(f"Total files to compress: {len(files_to_include)}")

    except Exception as e:
        print(f"Error reading master CSV or processing JSON files: {e}")
        return

    # Create the new archive
    try:
        with tarfile.open(remade_archive_path, "w:gz") as tar:
            for file in files_to_include:
                file_path = os.path.join(input_directory, "projects", artifact_file)
                if os.path.exists(file_path):
                    tar.add(file_path, arcname=file)
                else:
                    print(f"Warning: File '{file}' not found in input directory.")
        print(f"Successfully remade archive: {remade_archive_path}")
    except Exception as e:
        print(f"Error creating archive: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remake a failed batched archive by referencing the master CSV file.")
    parser.add_argument("input_directory", type=str, help="Path to the input folder containing files.")
    parser.add_argument("output_directory", type=str, help="Path to the output folder where archives are saved.")
    parser.add_argument("files_per_archive", type=int, help="Number of files to include in each archive.")
    parser.add_argument("master_csv_path", type=str, help="Path to the master CSV file.")
    parser.add_argument("failed_archive", type=str, help="Name of the failed archive to remake.")

    args = parser.parse_args()

    remake_failed_archive_no_dependency(
        args.input_directory,
        args.output_directory,
        args.files_per_archive,
        args.master_csv_path,
        args.failed_archive
    )