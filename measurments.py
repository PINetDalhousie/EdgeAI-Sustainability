"""
Energy computation from USB power-meter logs.

The USB power meter exports a CSV of readings sampled at a fixed rate (16 Hz in this study,
i.e. one point every 1/16 s). The rows themselves carry no absolute time, so this script
reconstructs a timestamp per row from a known start time plus the sampling interval, then
sums the energy recorded within a requested time range.

Typical use: give the inference-phase window (from the inference script's printed Start/End
timestamps, see `phase_identification.py`) to obtain the energy consumed during inference,
which — after subtracting the idle baseline — is the compute-specific energy reported in the
paper.
"""

import pandas as pd
from datetime import datetime, timedelta


def add_timestamp_column(csv_file, start_timestamp, sampling_rate):
    # Read the raw power-meter CSV into a DataFrame (one reading per row).
    data = pd.read_csv(csv_file)

    # Anchor the first row to the recording's start time.
    start_time = datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S')

    # Reconstruct an absolute timestamp for each row: row i occurs at
    # start_time + i * sampling_rate seconds (sampling_rate is seconds-per-point,
    # e.g. 1/16 = 0.0625 s for a 16 Hz meter).
    timestamps = []
    for index, row in data.iterrows():
        timestamp = start_time + timedelta(seconds=index * sampling_rate)
        timestamps.append(timestamp)

    # Attach the reconstructed timestamps as a new column for range filtering.
    data['timestamp'] = timestamps

    return data


def calculate_energy_sum_within_range(data, start_range, end_range):
    # Parse the requested [start, end] window (e.g. the inference phase).
    start_time = datetime.strptime(start_range, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_range, '%Y-%m-%d %H:%M:%S')

    # Keep only the rows whose reconstructed timestamp falls inside the window.
    filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

    # Sum every column whose name contains "energy" (the meter may export more than one),
    # over all rows in the window: .sum() collapses rows per column, the outer .sum()
    # collapses across those energy columns to a single total.
    energy_columns = [col for col in data.columns if 'energy' in col]
    energy_sum = filtered_data[energy_columns].sum().sum()

    return energy_sum

def main():
    # Prompt for the raw meter CSV and the parameters needed to time-stamp it.
    csv_file = input("Enter the path to the CSV file: ")
    start_timestamp = input("Enter the start timestamp (format: YYYY-MM-DD HH:MM:SS): ")
    # Sampling interval in seconds-per-point (e.g. 0.0625 for a 16 Hz meter).
    sampling_rate = float(input("Enter the sampling rate (seconds/point): "))

    # Reconstruct per-row timestamps from the start time and sampling interval.
    data = add_timestamp_column(csv_file, start_timestamp, sampling_rate)

    # Ask for the window to integrate (e.g. the inference phase) and total its energy.
    start_range = input("Enter the start timestamp of the range: ")
    end_range = input("Enter the end timestamp of the range: ")
    energy_sum = calculate_energy_sum_within_range(data, start_range, end_range)

    print(f"Sum of energy columns within the range: {energy_sum}")


if __name__ == "__main__":
    main()
