import os
import argparse
from huggingface_hub import upload_large_folder, login


def upload_dataset_to_huggingface(dataset_folder, repo_id, token_file):
    """
    Upload a dataset folder to Hugging Face using the upload_large_folder function.

    Parameters:
        dataset_folder (str): Path to the dataset folder to upload.
        repo_id (str): The repository ID on Hugging Face (e.g., "username/repo_name").
        token_file (str): Path to the file containing the Hugging Face API token.
    """
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"The dataset folder '{dataset_folder}' does not exist.")

    #if not os.path.exists(token_file):
    #    raise FileNotFoundError(f"The token file '{token_file}' does not exist.")

    # Read the token from the file
    #with open(token_file, "r") as f:
    #    token = f.read().strip()

    # Upload the dataset folder
    try:
        #login(token=token) #Login via CLI before running!
        print("Logged in to Hugging Face successfully.")
        print(f"Uploading dataset to Hugging Face repository: {repo_id}")
        upload_large_folder(folder_path=dataset_folder, repo_id=repo_id, repo_type="dataset")
        print(f"Dataset successfully uploaded to Hugging Face at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload the dataset: {e} (You might need to login via CLI first!)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a large dataset folder to Hugging Face.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=".",
        help="Path to the dataset folder to upload."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="fastmachinelearning/wa-hls4ml-projects",
        help="The repository ID on Hugging Face (e.g., 'username/repo_name')."
    )
    #parser.add_argument(
    #    "--token_file",
    #   type=str,
    #   default="~/.secrets/hf_token.txt",
    #    help="Path to the file containing the Hugging Face API token."
    #)

    args = parser.parse_args()

    upload_dataset_to_huggingface(args.dataset_folder, args.repo_id, args.token_file)