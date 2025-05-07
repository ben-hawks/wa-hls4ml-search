import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_file(processed_file_path, raw_report_path):
    try:
        # Load the processed JSON file
        with open(processed_file_path, 'r') as processed_file:
            processed_data = json.load(processed_file)

        # Check if "hls_resource_report" is not empty
        if processed_data.get("hls_resource_report"):
            # Load the raw report JSON file
            with open(raw_report_path, 'r') as raw_report_file:
                raw_report_data = json.load(raw_report_file)

            # Add new keys to the "latency_report" section
            if "latency_report" in processed_data:
                processed_data["latency_report"]["interval_max"] = raw_report_data['CSynthesisReport']['IntervalMax']
                processed_data["latency_report"]["interval_min"] = raw_report_data['CSynthesisReport']['IntervalMin']

        # Save the updated processed JSON file
        with open(processed_file_path, 'w') as processed_file:
            json.dump(processed_data, processed_file, indent=4)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except KeyError as e:
        print(f"Missing key in JSON data: {e}")
    except Exception as e:
        print(f"Error processing {processed_file_path}: {e}")

def process_json_files(directory):
    tasks = []
    with ProcessPoolExecutor() as executor:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith("_processed.json"):
                    processed_file_path = os.path.join(root, filename)
                    model_name = filename.replace("_processed.json", "")
                    raw_report_path = os.path.join(root, "raw_reports", f"{model_name}_report.json")
                    tasks.append(executor.submit(process_file, processed_file_path, raw_report_path))

        # Use tqdm to track progress
        for _ in tqdm(tasks, desc="Processing files"):
            _.result()  # Wait for each task to complete

if __name__ == "__main__":
    directory = "/mnt/d/hls4ml/wa-hls4ml-search/dataset/extracted" 
    process_json_files(directory)