"""Run exactly one (model, RF) synthesis unit.

This is the per-unit entry point invoked by run_synthesis_array.py's chunk driver as
its own subprocess -- deliberately a subprocess, not an in-process function call, so
that a native crash (a segfault somewhere in the TF/hls4ml/qkeras C-extension stack) or
an OOM-kill can't silently take the rest of its chunk's units down with it. The chunk
driver observes this process's exit code to decide success/failure; a nonzero exit
(including an uncaught exception's default traceback + exit(1)) is exactly the signal
it's looking for, so no extra try/except wrapping is needed here.

Usage:
    python run_one_unit.py --batch-file <path> --model-name <name> --rf <int> \\
        -o <output_dir> --hlsproj <hls_scratch_dir> [--strat Resource] \\
        [--part xcu250-figd2104-2L-e] [--vsynth] [--conv]
"""

import argparse

from iter_manager_v2 import load_named_model_from_batch_file
from run_search_iteration import run_iter


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-file', required=True, help='Path to the batch JSON file containing this model.')
    parser.add_argument('--model-name', required=True, help='Key of this model within the batch JSON.')
    parser.add_argument('--rf', type=int, required=True, help='Reuse factor for this unit.')
    parser.add_argument('-o', '--output', required=True, help='Output directory.')
    parser.add_argument('--hlsproj', required=True, help='HLS scratch project directory.')
    parser.add_argument('--strat', default='Resource', help='HLS4ML strategy (default: Resource).')
    parser.add_argument('--part', default='xcu250-figd2104-2L-e', help='Target part.')
    parser.add_argument('-v', '--vsynth', action='store_true', help='Enable Vivado synthesis.')
    parser.add_argument('-c', '--conv', action='store_true',
                         help='Enable convolutional model mode (io_stream + related settings).')
    return parser


def main():
    args = create_parser().parse_args()
    model = load_named_model_from_batch_file(args.batch_file, args.model_name)
    run_iter(
        args.model_name, None, args.rf, args.output,
        part=args.part, hlsproj=args.hlsproj, vsynth=args.vsynth,
        strat=args.strat, model=model, conv=args.conv,
    )


if __name__ == "__main__":
    main()
