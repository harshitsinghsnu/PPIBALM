# print_config.py
import json
import os

config_path = "../config/absolute_paths_config.json"
print(f"Reading config from: {os.path.abspath(config_path)}")

with open(config_path, 'r') as f:
    config = json.load(f)

print("\nModels section:")
for model_name, model_info in config["models"].items():
    print(f"  Model: {model_name}")
    for key, value in model_info.items():
        print(f"    {key}: {repr(value)}")  # Use repr to show special characters