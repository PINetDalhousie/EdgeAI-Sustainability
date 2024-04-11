def group_lines_by_phase(input_file, phases):
    """
    Groups lines from the input file into phases based on start and end timestamps.
    Args:
        input_file (str): Path to the input file.
        phases (list of tuples): Each tuple contains (phase_name, start_timestamp, end_timestamp).

    Returns:
        dict: A dictionary where keys are phase names and values are lists of lines belonging to each phase.
    """
    phase_lines = {phase_name: [] for phase_name, _, _ in phases}

    with open(input_file, 'r') as f:
        for line in f:
            timestamp, value = line.strip().split(',')
            timestamp = timestamp.strip()

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
    with open(input_file, 'r') as f:
        original_lines = f.readlines()

    with open(input_file, 'w') as f:
        for phase_name, lines in phase_lines.items():
            f.write(f"{phase_name}:\n")
            for line in lines:
                f.write(f"{line}\n")

if __name__ == "__main__":
    input_file_path = "knn.txt"
    phases = [
        ("initial", "2024-04-05 22:05:15.028", "2024-04-05 22:05:15.153"),
        ("loading test dataset", "2024-04-05 22:05:15.216", "2024-04-05 22:05:15.403"),
        ("loading saved model image", "2024-04-05 22:05:15.466", "2024-04-05 22:05:15.591"),
        ("inference", "2024-04-05 22:05:15.653", "2024-04-05 22:05:15.841"),
        ("memory utilization + accuracy", "2024-04-05 22:05:15.903", "2024-04-05 22:05:16.216")
    ]

    phase_lines = group_lines_by_phase(input_file_path, phases)
    modify_file_with_grouped_lines(input_file_path, phase_lines)