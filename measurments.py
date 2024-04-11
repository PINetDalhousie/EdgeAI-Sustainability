import pandas as pd
from datetime import datetime, timedelta


def add_timestamp_column(csv_file, start_timestamp, sampling_rate):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Convert start timestamp to datetime object
    start_time = datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S')

    # Calculate timestamp for each row based on sampling rate
    timestamps = []
    for index, row in data.iterrows():
        timestamp = start_time + timedelta(seconds=index * sampling_rate)
        timestamps.append(timestamp)

    # Add timestamp column to the dataframe
    data['timestamp'] = timestamps

    return data


def calculate_energy_sum_within_range(data, start_range, end_range):
    # Convert range timestamps to datetime objects
    start_time = datetime.strptime(start_range, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_range, '%Y-%m-%d %H:%M:%S')

    # Filter data within the given range
    filtered_data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]

    # Calculate the sum of energy columns within the range
    energy_columns = [col for col in data.columns if 'energy' in col]
    energy_sum = filtered_data[energy_columns].sum().sum()

    return energy_sum

def main():
    # Input parameters
    csv_file = input("Enter the path to the CSV file: ")
    start_timestamp = input("Enter the start timestamp (format: YYYY-MM-DD HH:MM:SS): ")
    sampling_rate = float(input("Enter the sampling rate (seconds/point): "))

    # Add timestamp column to the CSV data
    data = add_timestamp_column(csv_file, start_timestamp, sampling_rate)

    # Calculate the sum of energy columns within a given range
    start_range = input("Enter the start timestamp of the range: ")
    end_range = input("Enter the end timestamp of the range: ")
    energy_sum = calculate_energy_sum_within_range(data, start_range, end_range)

    print(f"Sum of energy columns within the range: {energy_sum}")


if __name__ == "__main__":
    main()