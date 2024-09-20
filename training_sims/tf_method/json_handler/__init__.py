import json

class json_handler:
    data = {}
    filename = ""
    def __init__(self, filename):
        self.filename = filename
        self._load_data()

    def _load_data(self):
        """Load data from the JSON file into the data attribute."""
        try:
            with open(self.filename, 'r') as json_file:
                self.data = json.load(json_file)
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.filename}.")

    def get_snr_values(self):
        """Getter function for SNR values."""
        return self.data.get("snr_values", [])

    def get_test_id(self):
        """Getter function for Test ID."""
        return self.data.get("test_id", "")

    def get_spreading_factor(self):
        """Getter function for Spreading Factor."""
        return self.data.get("spreading_factor", 0)

    def number_of_samples(self):
        """Getter function for number_of_samples."""
        return self.data.get("number_of_samples", 0)

if __name__ == '__main__':
    filename = 'snr_data.json'
    data = json_handler(filename)
    print(f"SNR Values: {data.get_snr_values()}")
    print(f"Test ID: {data.get_test_id()}")
    print(f"Sample ID: {data.get_sample_id()}")
    print(f"Spreading Factor: {data.get_spreading_factor()}")
    print(f"Sample Number: {data.get_sample_number()}")