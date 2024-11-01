import wandb
import yaml

# Login into W&B
# API key: f39f5b62dc920f2d392889028e3d672700fecb83
#wandb.login()

# Load sweep configuation
with open('cnn_folder/sweep_config.yaml') as file:
    sweep_config = yaml.safe_load(file)
print("Sweep_config file loaded")

# Create the sweep in W&B using the configuration loaded from the YAML
sweep_id = wandb.sweep(sweep_config, project="lora-symbol-detection")


