import os
import argparse
import pandas as pd
from util.batch_compress_files import compress_files_from_json

def remake_failed_archive(input_directory, output_directory, files_per_archive, master_csv_path, failed_archive, use_pigz=False, pigz_cores=1):
    """
    Remake a failed batched archive and update the master CSV file.

    Parameters:
        input_directory (str): Path to the folder containing the JSON files.
        output_directory (str): Path to the folder where the tar.gz files are saved.
        files_per_archive (int): Number of files to include in each archive.
        master_csv_path (str): Path to the master CSV file.
        failed_archive (str): Name of the failed archive to remake.
        use_pigz (bool): Whether to use pigz for compression.
        pigz_cores (int): Number of cores to use with pigz.
    """
    failed_archive_path = os.path.join(output_directory, failed_archive)

    if not os.path.exists(failed_archive_path):
        print(f"Error: Failed archive '{failed_archive}' does not exist in '{output_directory}'.")
        return

    print(f"Remaking failed archive: {failed_archive}")

    # Remove the failed archive
    os.remove(failed_archive_path)

    # Update the master CSV file to remove associated artifact files
    try:
        master_csv = pd.read_csv(master_csv_path)
        updated_csv = master_csv[master_csv['Archive Name'] != failed_archive]
        updated_csv.to_csv(master_csv_path, index=False)
        print(f"Updated master CSV file: Removed entries for '{failed_archive}'.")
    except Exception as e:
        print(f"Error updating master CSV file: {e}")
        return

    # Recreate the archive
    try:
        compress_files_from_json(
            input_directory,
            output_directory,
            files_per_archive,
            master_csv_path,
            use_pigz,
            pigz_cores
        )
        print(f"Successfully remade archive: {failed_archive}")
    except Exception as e:
        print(f"Error remaking archive: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remake a failed batched archive.")
    parser.add_argument("input_directory", type=str, help="Path to the input folder containing JSON files.")
    parser.add_argument("output_directory", type=str, help="Path to the output folder where archives are saved.")
    parser.add_argument("files_per_archive", type=int, help="Number of files to include in each archive.")
    parser.add_argument("master_csv_path", type=str, help="Path to the master CSV file.")
    parser.add_argument("failed_archive", type=str, help="Name of the failed archive to remake.")
    parser.add_argument("--use-pigz", action="store_true", help="Use pigz for compression.")
    parser.add_argument("--pigz-cores", type=int, default=1, help="Number of cores to use with pigz (default: 1).")

    args = parser.parse_args()

    remake_failed_archive(
        args.input_directory,
        args.output_directory,
        args.files_per_archive,
        args.master_csv_path,
        args.failed_archive,
        args.use_pigz,
        args.pigz_cores
    )