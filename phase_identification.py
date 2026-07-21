"""
Phase identification for USB power-meter logs.

Each per-device inference script prints a Start/End timestamp for every phase of the
5-step measurement protocol (initialize/baseline, load dataset, load model, inference,
memory + accuracy). The USB power meter, meanwhile, logs time-stamped readings continuously
to a text file with one "timestamp,value" pair per line.

This script takes that raw power-meter log and the phase time ranges (copied from the
inference script's printed timestamps) and rewrites the file so its lines are grouped under
the phase they belong to. That makes it easy to isolate the samples recorded during the
inference phase for the energy computation in `measurments.py`.

Note: `modify_file_with_grouped_lines` overwrites the input file in place, so keep a copy
of the original raw log if you need it.
"""


def group_lines_by_phase(input_file, phases):
    """
    Groups lines from the input file into phases based on start and end timestamps.
    Args:
        input_file (str): Path to the input file.
        phases (list of tuples): Each tuple contains (phase_name, start_timestamp, end_timestamp).

    Returns:
        dict: A dictionary where keys are phase names and values are lists of lines belonging to each phase.
    """
    # Start with an empty list of lines for every phase, preserving the given order.
    phase_lines = {phase_name: [] for phase_name, _, _ in phases}

    with open(input_file, 'r') as f:
        for line in f:
            # Each log line is "timestamp,value" (e.g. "2024-04-05 22:05:15.028,2.13").
            timestamp, value = line.strip().split(',')
            timestamp = timestamp.strip()

            # Assign the line to the first phase whose [start, end] window contains it.
            # String comparison works because the timestamps are zero-padded and
            # lexicographically ordered (YYYY-MM-DD HH:MM:SS.mmm). `break` ensures each
            # line lands in a single phase; lines outside every window are dropped.
            for phase_name, start_time, end_time in phases:
                if start_time <= timestamp <= end_time:
                    phase_lines[phase_name].append(f"{timestamp},{value}")
                    break

    return phase_lines

def modify_file_with_grouped_lines(input_file, phase_lines):
    """
    Modifies the input file with the grouped lines.
    Args:
        input_file (str): Path to the input file.
        phase_lines (dict): Dictionary containing phase names and their corresponding lines.
    """
    # Read the original file first (kept for reference; the write below replaces it).
    with open(input_file, 'r') as f:
        original_lines = f.readlines()

    # Rewrite the file grouped by phase: a "<phase name>:" header followed by its samples.
    with open(input_file, 'w') as f:
        for phase_name, lines in phase_lines.items():
            f.write(f"{phase_name}:\n")
            for line in lines:
                f.write(f"{line}\n")

if __name__ == "__main__":
    # Path to the raw USB power-meter log to be grouped in place.
    input_file_path = "knn.txt"

    # Phase windows for a single run, taken from the Start/End timestamps that the
    # inference script printed for this experiment. Update these per run/model/device.
    phases = [
        ("initial", "2024-04-05 22:05:15.028", "2024-04-05 22:05:15.153"),
        ("loading test dataset", "2024-04-05 22:05:15.216", "2024-04-05 22:05:15.403"),
        ("loading saved model image", "2024-04-05 22:05:15.466", "2024-04-05 22:05:15.591"),
        ("inference", "2024-04-05 22:05:15.653", "2024-04-05 22:05:15.841"),
        ("memory utilization + accuracy", "2024-04-05 22:05:15.903", "2024-04-05 22:05:16.216")
    ]

    # Group the samples by phase, then overwrite the log with the grouped version.
    phase_lines = group_lines_by_phase(input_file_path, phases)
    modify_file_with_grouped_lines(input_file_path, phase_lines)
