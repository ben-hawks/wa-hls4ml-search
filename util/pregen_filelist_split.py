import csv
import sys
import os
import argparse

# example usage: python split.py example.csv 200
# above command would split the `example.csv` into smaller CSV files of 200 rows each (with header included)
# if example.csv has 401 rows for instance, this creates 3 files in same directory:
#   - `example_1.csv` (row 1 - 200)
#   - `example_2.csv` (row 201 - 400)
#   - `example_3.csv` (row 401)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# default filename (relative to this script)
DEFAULT_FILENAME = "../pregen_3layer_models/filelist.csv"

# parse command-line arguments
parser = argparse.ArgumentParser(description='Split a CSV file into multiple smaller CSV files.')
parser.add_argument('filename', nargs='?', default=DEFAULT_FILENAME,
                    help='Input CSV filename (absolute or relative to this script).')
parser.add_argument('-r', '--rows-per-csv', type=int, default=2000,
                    help='Number of data rows per output CSV (positive integer). Default: 2000')
args = parser.parse_args()

# resolve filename (if relative, make it relative to the script dir)
if os.path.isabs(args.filename):
    full_file_path = args.filename
else:
    full_file_path = os.path.join(CURRENT_DIR, args.filename)

file_name = os.path.splitext(full_file_path)[0]

rows_per_csv = args.rows_per_csv

if rows_per_csv <= 0:
    print('Error: --rows-per-csv must be a positive integer')
    sys.exit(1)

if not os.path.exists(full_file_path):
    print(f"Error: input file not found: {full_file_path}")
    sys.exit(1)

with open(full_file_path, newline='') as infile:
    reader = csv.DictReader(infile)
    header = reader.fieldnames
    if header is None:
        print(f"Error: input file {full_file_path} appears empty or has no header")
        sys.exit(1)
    rows = [row for row in reader]
    pages = []

    row_count = len(rows)
    start_index = 0
    # here, we slice the total rows into pages, each page having [row_per_csv] rows
    while start_index < row_count:
        pages.append(rows[start_index: start_index+rows_per_csv])
        start_index += rows_per_csv

    for i, page in enumerate(pages):
        out_path = f"{file_name}_{i+1}.csv"
        # ensure output directory exists
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w+', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=header)
            writer.writeheader()
            for row in page:
                writer.writerow(row)

    print('DONE splitting {} into {} files'.format(full_file_path, len(pages)))
