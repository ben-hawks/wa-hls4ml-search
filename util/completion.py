import json
import os


def unit_paths(output_dir, name, rf):
    """Return (raw_report_path, processed_json_path) for one (name, rf) synthesis unit."""
    raw_report_path = os.path.join(output_dir, "raw_reports", f"{name}_rf{rf}_report.json")
    processed_json_path = os.path.join(output_dir, f"{name}_rf{rf}_processed.json")
    return raw_report_path, processed_json_path


def is_unit_complete(output_dir, name, rf):
    """True if (name, rf) already has a real, successful synthesis result.

    A processed JSON existing on disk is not enough: run_search_iteration.py's
    process_json_entry() still writes a valid-looking processed JSON with empty
    report dicts when synthesis fails partway through. `resource_report` (sourced
    from VivadoSynthReport, the actual place-and-routed numbers) is only populated
    on the true-success path; `hls_resource_report` (sourced from CSynthesisReport,
    the pre-synthesis HLS estimate) survives into the failure-fallback branches too,
    so checking that alone would misclassify "HLS succeeded, Vivado synthesis
    failed" runs as done.
    """
    _, processed_json_path = unit_paths(output_dir, name, rf)
    if not os.path.exists(processed_json_path):
        return False
    try:
        with open(processed_json_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("resource_report"))


def count_complete_in_dir(output_dir):
    """Scan output_dir for *_processed.json files and count attempted vs. actually
    succeeded (non-empty resource_report) -- the same real-success definition as
    is_unit_complete, applied in bulk for a monitoring dashboard rather than a single
    (name, rf) lookup. Returns (attempted, succeeded).
    """
    attempted = 0
    succeeded = 0
    if not os.path.isdir(output_dir):
        return attempted, succeeded
    for entry in os.scandir(output_dir):
        if not entry.is_file() or not entry.name.endswith("_processed.json"):
            continue
        attempted += 1
        try:
            with open(entry.path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("resource_report"):
            succeeded += 1
    return attempted, succeeded


if __name__ == "__main__":
    import sys
    a, s = count_complete_in_dir(sys.argv[1])
    print(f"{a} {s}")
