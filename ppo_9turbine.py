import os
import pandas as pd

# Assuming 'power_data' is obtained from your simulation
# This function gets called at the end of each episode
def log_episode_data(episode, power_data):
    csv_file_path = "episode_data.csv"
    avg_power_mw = sum(power_data) / len(power_data) if power_data else 0
    total_power_mw = sum(power_data)
    max_power_mw = max(power_data) if power_data else 0

    # Prepare the data to be appended
    data_to_append = {
        "episode": episode,
        "avg_power_mw": avg_power_mw,
        "total_power_mw": total_power_mw,
        "max_power_mw": max_power_mw
    }
    
    # Check if the file exists to write header or append data
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a') as f:
        # Write header only if the file is new (does not exist)
        if not file_exists:
            f.write("episode,avg_power_mw,total_power_mw,max_power_mw\n")
        # Write data row
        f.write(f"{data_to_append['episode']},{data_to_append['avg_power_mw']},{data_to_append['total_power_mw']},{data_to_append['max_power_mw']}\n")

    # Further processing...