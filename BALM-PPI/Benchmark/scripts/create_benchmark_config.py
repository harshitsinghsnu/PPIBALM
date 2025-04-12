import os
import json
import argparse
from datetime import datetime

# Default configuration template
DEFAULT_CONFIG = {
    "output_dir": "../results",
    "timestamp": "{timestamp}",
    "dataset_path": "../../scripts/notebooks/Data.csv",
    "test_size": 0.8,
    "random_seeds": [123],
    "metrics": ["pearson", "rmse", "spearman", "ci"],
    "models": {
        "BALM-PPI": {
            "type": "balm",
            "config_path": "D:\BALM_Fineclone\BALM-PPI\default_configs\balm_peft.yaml",
            "checkpoint_path": "D:\BALM_Fineclone\BALM-PPI\scripts\notebooks\latest_checkpointcuratedmainATTPFT.pth"
        },
        "MINT": {
            "type": "mint",
            "config_path": "../../data/esm2_t33_650M_UR50D.json",
            "checkpoint_path": "../models/mint.ckpt"
        }
    }
}

def create_config(args):
    """Create a benchmark configuration file."""
    config = DEFAULT_CONFIG.copy()
    
    # Update timestamp
    config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update configuration based on arguments
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    
    if args.test_size is not None:
        config["test_size"] = args.test_size
    
    if args.random_seeds:
        config["random_seeds"] = args.random_seeds
    
    # Update model configurations
    if args.balm_config:
        config["models"]["BALM-PPI"]["config_path"] = args.balm_config
    
    if args.balm_checkpoint:
        config["models"]["BALM-PPI"]["checkpoint_path"] = args.balm_checkpoint
    
    if args.mint_config:
        config["models"]["MINT"]["config_path"] = args.mint_config
    
    if args.mint_checkpoint:
        config["models"]["MINT"]["checkpoint_path"] = args.mint_checkpoint
    
    # Add or remove models
    if args.no_balm:
        del config["models"]["BALM-PPI"]
    
    if args.no_mint:
        del config["models"]["MINT"]
    
    # Add BALM variants
    if args.include_balm_variants:
        config["models"]["BALM-LoRA"] = {
            "type": "balm",
            "config_path": "../../configs/esm_lora_cosinemse_training_1.yaml",
            "checkpoint_path": "../../outputs/lora/best_model.pth"
        }
        config["models"]["BALM-LoKr"] = {
            "type": "balm",
            "config_path": "../../configs/esm_lokr_cosinemse_training.yaml",
            "checkpoint_path": "../../outputs/lokr/best_model.pth"
        }
        config["models"]["BALM-LoHa"] = {
            "type": "balm",
            "config_path": "../../configs/esm_loha_cosinemse_training.yaml",
            "checkpoint_path": "../../outputs/loha/best_model.pth"
        }
        config["models"]["BALM-IA3"] = {
            "type": "balm",
            "config_path": "../../configs/esm_ia3_cosinemse_training.yaml",
            "checkpoint_path": "../../outputs/ia3/best_model.pth"
        }
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Write configuration to file
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Benchmark configuration created at {args.output}")
    print(f"  - Models included: {list(config['models'].keys())}")
    print(f"  - Dataset: {config['dataset_path']}")

def main():
    parser = argparse.ArgumentParser(description="Create a benchmark configuration file")
    
    # Dataset options
    parser.add_argument("--dataset_path", type=str, help="Path to dataset CSV file")
    parser.add_argument("--test_size", type=float, help="Fraction of data for testing")
    parser.add_argument("--random_seeds", type=int, nargs="+", help="Random seeds for reproducibility")
    
    # Model options
    parser.add_argument("--balm_config", type=str, help="Path to BALM-PPI config file")
    parser.add_argument("--balm_checkpoint", type=str, help="Path to BALM-PPI checkpoint file")
    parser.add_argument("--mint_config", type=str, help="Path to MINT config file")
    parser.add_argument("--mint_checkpoint", type=str, help="Path to MINT checkpoint file")
    
    # Configuration options
    parser.add_argument("--no_balm", action="store_true", help="Exclude BALM-PPI from benchmark")
    parser.add_argument("--no_mint", action="store_true", help="Exclude MINT from benchmark")
    parser.add_argument("--include_balm_variants", action="store_true", help="Include BALM-PPI variants (LoRA, LoKr, LoHa, IA3)")
    
    # Output options
    parser.add_argument("--output", type=str, default="../config/benchmark_config.json", help="Output path for configuration file")
    
    args = parser.parse_args()
    create_config(args)

if __name__ == "__main__":
    main()