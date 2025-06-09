from huggingface_hub import delete_folder, list_repo_files
import os
from tqdm import tqdm

def delete_all_folders_in_dataset(repo_id, token_file):
    """
    Delete all folders in a Hugging Face dataset repository.

    Parameters:
        repo_id (str): The repository ID (e.g., "username/dataset_name").
        token_file (str): Path to the file containing the Hugging Face authentication token.
    """
    # Read the token from the file
    with open(token_file, "r") as f:
        token = f.read().strip()

    # List all files and folders in the repository
    repo_files = list_repo_files(repo_id, token=token, repo_type="dataset")

    # Identify folders (directories) in the repository
    folders = {os.path.dirname(file) for file in repo_files if "/" in file}

    # Delete each folder
    for folder in tqdm(folders):
        if folder:  # Ensure it's not the root directory
            delete_folder(repo_id=repo_id, folder_path=folder, token=token, repo_type="dataset")
            print(f"Deleted folder: {folder}")

# Example usage
if __name__ == "__main__":
    repo_id = "fastmachinelearning/wa-hls4ml"  # Replace with your dataset repository ID
    token_file = "/home/bhawks/.secrets/hf_token.txt"  # Replace with the path to your token file
    delete_all_folders_in_dataset(repo_id, token_file)