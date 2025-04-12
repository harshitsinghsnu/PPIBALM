# create_clean_config.py
import json
import os

config = {
    "output_dir": "D:/BALM_Fineclone/BALM-PPI/Benchmark/results",
    "timestamp": "20250329_224500",
    "dataset_path": "D:/BALM_Fineclone/BALM-PPI/scripts/notebooks/Data.csv",
    "test_size": 0.8,
    "random_seeds": [123],
    "metrics": ["pearson", "rmse", "spearman", "ci"],
    "models": {
        "BALM-PPI": {
            "type": "balm",
            "config_path": "D:/BALM_Fineclone/BALM-PPI/configs/esm_lora_cosinemse_training_1.yaml",
            "checkpoint_path": "D:/BALM_Fineclone/BALM-PPI/scripts/notebooks/latest_checkpointcuratedmain.pth"
        }
    }
}

output_path = "../config/clean_config.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Clean config saved to: {os.path.abspath(output_path)}")