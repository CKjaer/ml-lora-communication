import os
import json
import uuid
from plot_stuff.generate_plots import generate_plots
from plot_stuff.load_files import load_data


if __name__ == "__main__":
    # generate a unique
    with open('config.json') as f:
        config = json.load(f)
    test_id = str(uuid.uuid4())
    os.makedirs(os.join("output", test_id), exist_ok=True)
    
    # generate data
    
    # generate plots
    generate_plots(load_data(os.join("output", test_id, "csv")), config["spreading_factor"], config["number_of_samples"], os.join("output", test_id))
    
    # save config
    with open(os.join("output", test_id, "config.json"), 'w') as f:
        json.dump(config, f)
    
        

        

    
        