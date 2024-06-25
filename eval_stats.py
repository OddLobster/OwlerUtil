import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

def extract_data(file_path):
    data = {
        "harvest_rate": [],
        "num_relevant_urls": [],
        "num_total_urls": [],
        "sum_of_information": [],
        "processing_time": [],
        "total_processing_time": []
    }

    with open(file_path, 'r') as file:
        content = file.read()

        harvest_rates = re.findall(r'EVAL HARVEST RATE: (.+)', content)
        num_relevant_urls = re.findall(r'EVAL NUM RELEVANT URLS: (.+)', content)
        num_total_urls = re.findall(r'EVAL NUM TOTAL URLS: (.+)', content)
        sum_of_information = re.findall(r'EVAL SUM OF INFORMATION: (.+)', content)
        processing_times = re.findall(r'EVAL PROCESSING TIME: (.+?) ms', content)
        total_processing_times = re.findall(r'EVAL TOTAL PROCESSING TIME: (.+?) ms', content)

        # Convert to appropriate types
        harvest_rates = list(map(float, harvest_rates))
        num_relevant_urls = list(map(int, num_relevant_urls))
        num_total_urls = list(map(int, num_total_urls))
        sum_of_information = list(map(float, sum_of_information))
        processing_times = list(map(int, processing_times))
        total_processing_times = list(map(int, total_processing_times))

        # Initialize previous values and cumulative sums
        previous_num_relevant_urls = 0
        previous_num_total_urls = 0
        previous_sum_of_information = 0.0
        previous_total_processing_time = 0
        previous_processing_time = 0

        reset_num_relevant_urls = 0
        reset_num_total_urls = 0
        reset_sum_of_information = 0.0
        reset_processing_time = 0
        reset_total_processing_time = 0

        new_harvest_rates = list()
        new_num_relevant_urls = list()
        new_num_total_urls = list()
        new_sum_of_information = list()
        new_processing_times = list()
        new_total_processing_times = list()
        reset_detected = False

        for i in range(len(harvest_rates)):
            if num_total_urls[i] == 1 and i != 0:  # Reset detected
                # Adjust cumulative sums to continue from the previous state
                print("Detected reset at i: ", i, file_path)
                print("PREV reset_num_relevant_urls: ", previous_num_relevant_urls)
                print("PREV reset_num_total_urls: ", previous_num_total_urls)
                print("PREV reset_sum_of_information: ", previous_sum_of_information)
                print("PREV reset_processing_time: ", previous_processing_time)
                print("PREV reset_total_processing_time: ", previous_total_processing_time)

                reset_num_relevant_urls = previous_num_relevant_urls
                reset_num_total_urls = previous_num_total_urls
                reset_sum_of_information = previous_sum_of_information
                reset_total_processing_time = previous_total_processing_time
                reset_processing_time = previous_processing_time

            previous_num_relevant_urls = num_relevant_urls[i] + reset_num_relevant_urls
            previous_num_total_urls = num_total_urls[i] + reset_num_total_urls
            previous_sum_of_information = sum_of_information[i] + reset_sum_of_information
            previous_processing_time = processing_times[i] + reset_processing_time
            previous_total_processing_time = total_processing_times[i] + reset_total_processing_time

            if num_total_urls[i] == 1 and i != 0: 
                print("@@@ reset_num_relevant_urls: ", previous_num_relevant_urls)
                print("@@@ reset_num_total_urls: ", previous_num_total_urls)
                print("@@@ reset_sum_of_information: ", previous_sum_of_information)
                print("@@@ reset_processing_time: ", previous_processing_time)
                print("@@@ reset_total_processing_time: ", previous_total_processing_time)

            new_harvest_rate = (previous_num_relevant_urls) / (previous_num_total_urls)
            print(new_harvest_rate, previous_num_total_urls)
            new_harvest_rates.append(new_harvest_rate)
            new_num_relevant_urls.append(previous_num_relevant_urls)
            new_num_total_urls.append(previous_num_total_urls)
            new_sum_of_information.append(previous_sum_of_information)
            new_processing_times.append(previous_processing_time)
            new_total_processing_times.append(previous_total_processing_time)
            


        data["harvest_rate"] = list(map(float, new_harvest_rates))
        data["num_relevant_urls"] = list(map(int, new_num_relevant_urls))
        data["num_total_urls"] = list(map(int, new_num_total_urls))
        data["sum_of_information"] = list(map(float, new_sum_of_information))
        data["processing_time"] = list(map(int, new_processing_times))
        data["total_processing_time"] = list(map(int, new_total_processing_times))

        # Debug prints
        print(f"File: {file_path}")
        print(f"Harvest Rates: {len(data['harvest_rate'])}")
        print(f"Num Relevant URLs: {len(data['num_relevant_urls'])}")
        print(f"Num Total URLs: {len(data['num_total_urls'])}")
        print(f"Sum of Information: {len(data['sum_of_information'])}")
        print(f"Processing Times: {len(data['processing_time'])}")
        print(f"Total Processing Times: {len(data['total_processing_time'])}")

    # Ensure all lists are the same length
    min_length = min(len(data[key]) for key in data)
    min_length = 360
    for key in data:
        print(len(data[key]),key)
        data[key] = data[key][:min_length]

    return pd.DataFrame(data)

# Function to plot data
def plot_data(data, stat_name, file_name):
    plt.figure(figsize=(10, 6))
    for label, df in data.items():
        plt.plot(df.index, df[stat_name], label=label)
    plt.xlabel('Entry Index')
    plt.ylabel(stat_name.replace('_', ' ').title())
    plt.title(f'{stat_name.replace("_", " ").title()} Over Time')
    plt.legend()
    plt.savefig(f'{file_name}_{stat_name}.png')
    plt.show()

# Main script
files = ["RUN_NO_TUNNELING_NOSHARK_wg2.txt", "RUN_NO_TUNNELING_wg2.txt", "RUN_ALL_wg2.txt", "RUN_NO_TUNNELING_NOSHARK_BLOCKEVAL_wg2.txt", "RUN.txt"]  # Add your file names here
# files = ["RUN_NO_TUNNELING_wg2.txt"]  # Add your file names here

data = {}

for file in files:
    if os.path.exists(file):
        data[file] = extract_data(file)
    else:
        print(f"File {file} does not exist.")


# List of stats to plot
stats_to_plot = ["harvest_rate", "num_relevant_urls", "sum_of_information", "processing_time"]

for stat in stats_to_plot:
    plot_data(data, stat, 'stats_plot')