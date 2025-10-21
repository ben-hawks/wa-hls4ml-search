import os
import json
from tqdm import tqdm

def merge_json_files_per_subdirectory(input_directory, output_directory, prefix):
    """
    Merge JSON files per subdirectory in the input directory and save them with names based on the subdirectory names.

    Parameters:
        input_directory (str): Path to the directory containing subdirectories with JSON files.
        output_directory (str): Path to the directory where merged JSON files will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through each subdirectory in the input directory
    for subdir in os.listdir(input_directory):
        subdir_path = os.path.join(input_directory, subdir)
        if os.path.isdir(subdir_path):
            merged_data = []
            json_files = [
                os.path.join(subdir_path, file)
                for file in os.listdir(subdir_path)
                if file.endswith(".json")
            ]

            # Merge JSON files in the subdirectory
            for file_path in tqdm(json_files, desc=f"Merging JSON files in {subdir}"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            merged_data.extend(data)
                        else:
                            merged_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

            # Save the merged data to an output file named after the subdirectory
            output_file = os.path.join(output_directory, f"{prefix}_{subdir}_merged.json")
            with open(output_file, "w") as f:
                json.dump(merged_data, f, indent=4)

            print(f"Merged JSON saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_directory = "../dataset/final/split_dataset/test"  # Replace with the path to your input directory
    output_directory = "../dataset/final/merged/test"  # Replace with the path to your output directory
    merge_json_files_per_subdirectory(input_directory, output_directory, "test")

    input_directory = "../dataset/final/split_dataset/val"  # Replace with the path to your input directory
    output_directory = "../dataset/final/merged/val"  # Replace with the path to your output directory
    merge_json_files_per_subdirectory(input_directory, output_directory, "val")

    input_directory = "../dataset/final/split_dataset/train"  # Replace with the path to your input directory
    output_directory = "../dataset/final/merged/train"  # Replace with the path to your output directory
    merge_json_files_per_subdirectory(input_directory, output_directory, "train")