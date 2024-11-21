import argparse

def extract_ser_data(log_file_path):
    # Initialize dictionary to store results
    ser_data = {}
    current_snr = None
    current_rate = None
    
    # Read through the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            # Look for lines that indicate new SNR and rate values
            if "Calculating SER for snr:" in line:
                # Extract SNR and rate from line like "Calculating SER for snr: -16, rate 0"
                parts = line.split()
                current_snr = int(float(parts[5].strip(',')))  # Convert "-16," to -16
                current_rate = float(parts[7])  # Convert "0" to 0.0
                
                # Initialize dictionary for this rate if it doesn't exist
                if current_rate not in ser_data:
                    ser_data[current_rate] = {}
                    
            # Look for lines containing Symbol Error Rate results
            elif "Symbol Error Rate (SER):" in line:
                # Extract SER value
                ser = float(line.split(': ')[1])
                # Store in nested dictionary
                ser_data[current_rate][current_snr] = ser
    
    return ser_data

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract SER data from log file')
    parser.add_argument('log_file', type=str, help='Path to log file')
    args = parser.parse_args()
    
    # Get the log file path from the command line
    log_file = args.log_file
     
    # Extract the data
    ser_data = extract_ser_data(log_file)
    
    # Optional: Print the results to verify
    for rate in sorted(ser_data.keys()):
        print(f"\nRate: {rate}")
        for snr in sorted(ser_data[rate].keys()):
            print(f"  SNR: {snr}, SER: {ser_data[rate][snr]}")