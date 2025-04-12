# run_all_benchmark.py
import os
import subprocess
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark(seed, data_path, balm_config, balm_checkpoint, mint_config, mint_checkpoint, output_dir):
    """Run a single benchmark with the given seed"""
    seed_output_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    
    cmd = [
        "python", "benchmark_mint_vs_balm.py",
        "--data_path", data_path,
        "--balm_config", balm_config,
        "--balm_checkpoint", balm_checkpoint,
        "--mint_config", mint_config,
        "--mint_checkpoint", mint_checkpoint,
        "--output_dir", seed_output_dir,
        "--seed", str(seed)
    ]
    
    subprocess.run(cmd, check=True)
    
    return seed_output_dir

def aggregate_results(seed_dirs, output_dir):
    """Aggregate results from multiple seed runs"""
    metrics = ["RMSE", "Pearson", "Spearman", "CI"]
    models = ["BALM", "MINT"]
    
    # Initialize results dictionary
    results = {
        model: {metric: [] for metric in metrics}
        for model in models
    }
    
    # Collect results from each seed run
    for seed_dir in seed_dirs:
        for model in models:
            with open(os.path.join(seed_dir, model.lower(), "metrics.json"), "r") as f:
                metrics_data = json.load(f)
                for metric in metrics:
                    results[model][metric].append(metrics_data["test"][metric])
    
    # Calculate mean and std for each metric
    summary = {
        model: {
            f"{metric}_mean": np.mean(results[model][metric]),
            f"{metric}_std": np.std(results[model][metric])
        }
        for model in models
        for metric in metrics
    }
    
    # Save summary
    with open(os.path.join(output_dir, "aggregated_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create comparison dataframe
    comparison = []
    for metric in metrics:
        for model in models:
            mean = np.mean(results[model][metric])
            std = np.std(results[model][metric])
            comparison.append({
                "Metric": metric,
                "Model": model,
                "Mean": mean,
                "Std": std
            })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, "aggregated_comparison.csv"), index=False)
    
    # Plot comparison with error bars
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        metric_data = comparison_df[comparison_df["Metric"] == metric]
        
        # Plot bar chart with error bars
        sns.barplot(x="Model", y="Mean", data=metric_data, yerr=metric_data["Std"])
        
        plt.title(metric)
        plt.ylim(0, max(metric_data["Mean"]) * 1.2)
        
        # Add value labels
        for j, row in metric_data.iterrows():
            plt.text(j, row["Mean"] + 0.01, f'{row["Mean"]:.3f}±{row["Std"]:.3f}', 
                    ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aggregated_comparison.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run multiple benchmarks with different seeds")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--balm_config", type=str, default="default_configs/balm_peft.yaml", help="Path to BALM config")
    parser.add_argument("--balm_checkpoint", type=str, default=None, help="Path to BALM checkpoint")
    parser.add_argument("--mint_config", type=str, default="./scripts/mint/data/esm2_t33_650M_UR50D.json", help="Path to MINT config")
    parser.add_argument("--mint_checkpoint", type=str, default="./models/mint.ckpt", help="Path to MINT checkpoint")
    parser.add_argument("--output_dir", type=str, default="multi_seed_benchmark", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for each seed
    seed_dirs = []
    for seed in args.seeds:
        print(f"\n=== Running benchmark with seed {seed} ===\n")
        seed_dir = run_benchmark(
            seed,
            args.data_path,
            args.balm_config,
            args.balm_checkpoint,
            args.mint_config,
            args.mint_checkpoint,
            args.output_dir
        )
        seed_dirs.append(seed_dir)
    
    # Aggregate results
    print("\n=== Aggregating results ===\n")
    aggregate_results(seed_dirs, args.output_dir)
    
    print(f"\nAll benchmarks completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()