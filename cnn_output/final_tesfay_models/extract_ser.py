import re
import json

# Path to the log file
log_file_path = "cnn_output/final_tesfay_models/train_cnn.log"
output_json_path = "cnn_output/final_tesfay_models/symbol_error_rates.json"

# Initialize a dictionary to store results
results = {}

# Regular expression to match the relevant lines
pattern = r"Trained and evaluated model for SNR: (-?\d+) and rate: ([\d.]+) SER is: ([\d.]+)"

# Read the log file and extract data
with open(log_file_path, 'r') as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            snr = int(match.group(1))
            rate = float(match.group(2))
            ser = float(match.group(3))
            
            # Organize data by rate and then by snr
            if rate not in results:
                results[rate] = {}
            results[rate][snr] = ser

# Save the results to a JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Data successfully processed and saved to {output_json_path}.")
