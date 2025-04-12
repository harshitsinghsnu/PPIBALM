import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime

def run_benchmark(config_path, output_dir, results_file):
    """Run benchmark with specific configuration."""
    start_time = time.time()
    print(f"\n{'=' * 60}")
    print(f"Running benchmark with config: {config_path}")
    print(f"{'=' * 60}\n")
    
    # Run benchmark script
    cmd = [
        "python", "benchmark_pipeline.py",
        "--config", config_path,
        "--output_dir", output_dir
    ]
    
    process = subprocess.run(cmd, check=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log results
    with open(results_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Config: {config_path} - Duration: {duration:.2f}s\n")
    
    print(f"\nBenchmark completed in {duration:.2f} seconds")
    return duration

def main():
    parser = argparse.ArgumentParser(description="Run multiple benchmarks with different configurations")
    parser.add_argument("--config_dir", type=str, default="../config/variations", help="Directory with benchmark config variations")
    parser.add_argument("--output_base_dir", type=str, default="../results", help="Base directory for benchmark results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Create results log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_base_dir, f"benchmark_runs_{timestamp}.log")
    
    # Find all configuration files
    config_files = []
    if os.path.exists(args.config_dir):
        for file in os.listdir(args.config_dir):
            if file.endswith('.json'):
                config_files.append(os.path.join(args.config_dir, file))
    
    if not config_files:
        print(f"No configuration files found in {args.config_dir}")
        print("Creating default configurations...")
        
        # Create config directory
        os.makedirs(args.config_dir, exist_ok=True)
        
        # Create default configurations for different scenarios
        configs = {
            "default": {
                "dataset_path": "../../scripts/notebooks/Data.csv",
                "test_size": 0.8,
                "random_seeds": [123, 456, 789],
                "models": {
                    "BALM-PPI": {
                        "type": "balm",
                        "config_path": "../../configs/esm_lora_cosinemse_training_1.yaml",
                        "checkpoint_path": "../../outputs/lora/best_model.pth"
                    },
                    "MINT": {
                        "type": "mint",
                        "config_path": "../../data/esm2_t33_650M_UR50D.json",
                        "checkpoint_path": "../models/mint.ckpt"
                    }
                }
            },
            "balm_variants": {
                "dataset_path": "../../scripts/notebooks/Data.csv",
                "test_size": 0.8,
                "random_seeds": [123],
                "models": {
                    "BALM-LoRA": {
                        "type": "balm",
                        "config_path": "../../configs/esm_lora_cosinemse_training_1.yaml",
                        "checkpoint_path": "../../outputs/lora/best_model.pth"
                    },
                    "BALM-LoKr": {
                        "type": "balm",
                        "config_path": "../../configs/esm_lokr_cosinemse_training.yaml",
                        "checkpoint_path": "../../outputs/lokr/best_model.pth"
                    },
                    "BALM-LoHa": {
                        "type": "balm",
                        "config_path": "../../configs/esm_loha_cosinemse_training.yaml",
                        "checkpoint_path": "../../outputs/loha/best_model.pth"
                    },
                    "BALM-IA3": {
                        "type": "balm",
                        "config_path": "../../configs/esm_ia3_cosinemse_training.yaml",
                        "checkpoint_path": "../../outputs/ia3/best_model.pth"
                    }
                }
            },
            "dataset_splits": {
                "dataset_path": "../../scripts/notebooks/Data.csv",
                "test_size": 0.8,
                "random_seeds": [123],
                "data_splits": {
                    "train_ratio": [0.1, 0.2, 0.3]
                },
                "models": {
                    "BALM-PPI": {
                        "type": "balm",
                        "config_path": "../../configs/esm_lora_cosinemse_training_1.yaml",
                        "checkpoint_path": "../../outputs/lora/best_model.pth"
                    },
                    "MINT": {
                        "type": "mint",
                        "config_path": "../../data/esm2_t33_650M_UR50D.json",
                        "checkpoint_path": "../models/mint.ckpt"
                    }
                }
            }
        }
        
        # Write default configurations
        for name, config in configs.items():
            config_path = os.path.join(args.config_dir, f"{name}_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            config_files.append(config_path)
        
        print(f"Created {len(configs)} default configurations")
    
    # Run benchmarks for each configuration
    print(f"Running {len(config_files)} benchmark configurations")
    
    with open(results_file, "w") as f:
        f.write(f"Benchmark runs started at {timestamp}\n")
        f.write("=" * 60 + "\n\n")
    
    total_start_time = time.time()
    
    for i, config_path in enumerate(config_files):
        print(f"\nRunning benchmark {i+1}/{len(config_files)}")
        
        # Create specific output directory based on config name
        config_name = os.path.basename(config_path).replace('.json', '')
        output_dir = os.path.join(args.output_base_dir, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run benchmark
        try:
            duration = run_benchmark(config_path, output_dir, results_file)
        except Exception as e:
            print(f"Error running benchmark: {e}")
            with open(results_file, "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Config: {config_path} - ERROR: {e}\n")
    
    total_duration = time.time() - total_start_time
    
    with open(results_file, "a") as f:
        f.write(f"\nAll benchmark runs completed in {total_duration:.2f} seconds\n")
    
    print(f"\nAll benchmark runs completed in {total_duration:.2f} seconds")
    print(f"Results logged to {results_file}")

if __name__ == "__main__":
    main()