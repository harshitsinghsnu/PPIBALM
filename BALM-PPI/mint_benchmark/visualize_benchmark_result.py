import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import json
from matplotlib.colors import LinearSegmentedColormap

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.figsize': (10, 6)})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 14})

# Define custom color palettes
MODEL_COLORS = {
    "BALM-PPI": "#1f77b4",  # Blue
    "ESM2": "#ff7f0e"       # Orange
}

SPLIT_COLORS = {
    "random": "#2ca02c",       # Green
    "cold_target": "#d62728",  # Red
    "seq_similarity": "#9467bd" # Purple
}

METRIC_COLORS = {
    "RMSE": "#8c564b",      # Brown
    "Pearson": "#e377c2",   # Pink
    "Spearman": "#7f7f7f",  # Gray
    "R²": "#bcbd22",        # Olive
    "CI": "#17becf"         # Cyan
}

# Custom color map for heatmaps
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_blue_red", 
    ["#2171b5", "#6baed6", "#bdd7e7", "#eff3ff", "#fee0d2", "#fc9272", "#de2d26"]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize benchmark results with pretty plots")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Directory containing benchmark results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations (default: results_dir/visualizations)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures")
    return parser.parse_args()

def load_all_results(results_dir):
    """Load all results from the results directory"""
    # Find all summary directories
    summary_dirs = glob.glob(os.path.join(results_dir, "*_summary"))
    
    # Extract split types
    split_types = [os.path.basename(d).replace("_summary", "") for d in summary_dirs]
    
    # Load metrics summary for each split
    all_metrics = {}
    for split_type, summary_dir in zip(split_types, summary_dirs):
        metrics_file = os.path.join(summary_dir, "metrics_summary.csv")
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            all_metrics[split_type] = metrics_df
    
    # Load all seed-level data
    all_seed_data = {}
    for split_type, summary_dir in zip(split_types, summary_dirs):
        seed_file = os.path.join(summary_dir, "seed_level_metrics.csv")
        if os.path.exists(seed_file):
            seed_df = pd.read_csv(seed_file)
            all_seed_data[split_type] = seed_df
    
    # Load target analysis for cold_target and seq_similarity splits
    target_analysis = {}
    for split_type in ["cold_target", "seq_similarity"]:
        if split_type in split_types:
            target_file = os.path.join(results_dir, f"{split_type}_summary", "target_analysis.csv")
            if os.path.exists(target_file):
                target_df = pd.read_csv(target_file)
                target_analysis[split_type] = target_df
    
    return {
        "metrics": all_metrics,
        "seed_data": all_seed_data,
        "target_analysis": target_analysis,
        "split_types": split_types
    }

def create_cross_split_comparison(all_results, output_dir):
    """Create cross-split comparison plots"""
    metrics = ["RMSE", "Pearson", "Spearman", "R²", "CI"]
    split_types = all_results["split_types"]
    
    # Prepare data for bar plots
    comparison_data = []
    
    for split_type in split_types:
        metrics_df = all_results["metrics"][split_type]
        
        for metric in metrics:
            for model in ["BALM-PPI", "ESM2"]:
                row = metrics_df[(metrics_df["Metric"] == metric) & (metrics_df["Model"] == model)]
                if not row.empty:
                    comparison_data.append({
                        "Split": split_type,
                        "Model": model,
                        "Metric": metric,
                        "Value": row["Mean"].values[0],
                        "Std": row["Std"].values[0]
                    })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create separate plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_df = comparison_df[comparison_df["Metric"] == metric]
        
        # Create bar plot
        ax = plt.figure(figsize=(10, 6))
        
        split_labels = {"random": "Random Split", "cold_target": "Cold Target Split", "seq_similarity": "Sequence Similarity Split"}
        
        # Create grouped bar plot
        g = sns.catplot(
            data=metric_df, 
            kind="bar",
            x="Split", 
            y="Value", 
            hue="Model",
            palette=[MODEL_COLORS["BALM-PPI"], MODEL_COLORS["ESM2"]],
            alpha=0.8,
            height=6,
            aspect=1.5,
            legend=False
        )
        
        # Customize x-axis labels
        plt.xticks([i for i in range(len(split_types))], [split_labels.get(s, s) for s in split_types])
        
        # Add error bars
        plt.errorbar(
            x=np.arange(len(metric_df)),
            y=metric_df["Value"],
            yerr=metric_df["Std"],
            fmt="none",
            ecolor="black",
            capsize=5
        )
        
        # Customize plot
        plt.title(f"{metric} Comparison Across Split Types", fontsize=18)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel("", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add legend
        plt.legend(title="Model", loc="upper right", frameon=True)
        
        # Add value labels on top of bars
        ax = plt.gca()
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'{height:.3f}',
                ha="center", 
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cross_split_{metric}_comparison.png"), dpi=300)
        plt.close()
    
    # Create heatmap summary
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    pivot_data = []
    
    for split_type in split_types:
        metrics_df = all_results["metrics"][split_type]
        
        for metric in metrics:
            for model in ["BALM-PPI", "ESM2"]:
                row = metrics_df[(metrics_df["Metric"] == metric) & (metrics_df["Model"] == model)]
                if not row.empty:
                    split_name = split_type.replace("_", " ").title()
                    col_name = f"{model} - {metric}"
                    pivot_data.append({
                        "Split": split_name,
                        "MetricModel": col_name,
                        "Value": row["Mean"].values[0]
                    })
    
    pivot_df = pd.pivot_table(
        pd.DataFrame(pivot_data),
        values="Value",
        index="Split",
        columns="MetricModel"
    )
    
    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        linewidths=0.5,
        cbar_kws={"label": "Value"}
    )
    
    plt.title("Performance Across Models and Split Types", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_split_heatmap.png"), dpi=300)
    plt.close()

def create_model_performance_plots(all_results, output_dir):
    """Create model performance comparison plots"""
    metrics = ["RMSE", "Pearson", "Spearman", "R²", "CI"]
    split_types = all_results["split_types"]
    
    for split_type in split_types:
        # Radar chart comparing models
        metrics_df = all_results["metrics"][split_type]
        
        # Prepare data for radar chart
        radar_data = {}
        
        for model in ["BALM-PPI", "ESM2"]:
            model_data = {}
            for metric in metrics:
                row = metrics_df[(metrics_df["Metric"] == metric) & (metrics_df["Model"] == model)]
                if not row.empty:
                    model_data[metric] = row["Mean"].values[0]
            radar_data[model] = model_data
        
        # Normalize metrics for radar chart (RMSE needs to be inverse since lower is better)
        max_vals = {}
        min_vals = {}
        
        for metric in metrics:
            values = [data[metric] for data in radar_data.values() if metric in data]
            max_vals[metric] = max(values)
            min_vals[metric] = min(values)
        
        # Special case for RMSE (lower is better)
        if "RMSE" in max_vals:
            max_vals["RMSE"], min_vals["RMSE"] = min_vals["RMSE"], max_vals["RMSE"]
        
        # Normalize data to 0-1 scale
        normalized_data = {}
        for model, data in radar_data.items():
            normalized_model_data = {}
            for metric, value in data.items():
                if max_vals[metric] == min_vals[metric]:
                    normalized_model_data[metric] = 0.5
                else:
                    if metric == "RMSE":
                        # For RMSE, lower is better, so we invert the normalization
                        normalized_model_data[metric] = (max_vals[metric] - value) / (max_vals[metric] - min_vals[metric])
                    else:
                        normalized_model_data[metric] = (value - min_vals[metric]) / (max_vals[metric] - min_vals[metric])
            normalized_data[model] = normalized_model_data
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Number of metrics
        N = len(metrics)
        
        # Create angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize plot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per metric and add labels
        plt.xticks(angles[:-1], metrics, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot data
        for model, color in MODEL_COLORS.items():
            if model in normalized_data:
                values = [normalized_data[model].get(metric, 0) for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=model)
                ax.fill(angles, values, alpha=0.1, color=color)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        split_title = split_type.replace("_", " ").title()
        plt.title(f"Model Performance Comparison - {split_title}", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{split_type}_radar_chart.png"), dpi=300)
        plt.close()
        
        # Create bar chart for each metric
        seed_df = all_results["seed_data"][split_type]
        
        for metric in metrics:
            metric_col = f"test/{metric.lower()}" if metric.lower() in ["pearson", "rmse", "spearman", "ci"] else f"test/r2"
            
            plt.figure(figsize=(10, 6))
            
            # Group by model and seed
            metric_data = []
            for model in ["BALM-PPI", "ESM2"]:
                model_rows = seed_df[seed_df["Model"] == model]
                for _, row in model_rows.iterrows():
                    metric_data.append({
                        "Model": model,
                        "Seed": row["Random_seed"],
                        "Value": row[metric_col]
                    })
            
            metric_df = pd.DataFrame(metric_data)
            
            # Create bar plot
            ax = sns.barplot(
                data=metric_df,
                x="Seed",
                y="Value",
                hue="Model",
                palette=[MODEL_COLORS["BALM-PPI"], MODEL_COLORS["ESM2"]],
                alpha=0.8
            )
            
            # Add value labels on top of bars
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height + 0.01,
                    f'{height:.3f}',
                    ha="center", 
                    fontsize=9
                )
            
            # Customize plot
            split_title = split_type.replace("_", " ").title()
            plt.title(f"{metric} by Random Seed - {split_title}", fontsize=16)
            plt.xlabel("Random Seed", fontsize=14)
            plt.ylabel(metric, fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.legend(title="Model")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{split_type}_{metric}_by_seed.png"), dpi=300)
            plt.close()

def create_generalizability_plots(all_results, output_dir):
    """Create plots for generalizability analysis"""
    target_analysis = all_results["target_analysis"]
    
    for split_type, analysis_df in target_analysis.items():
        # Create density plots for RMSE and Pearson
        metrics = ["rmse", "pearson"]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            sns.kdeplot(
                data=analysis_df,
                x=metric,
                hue="model",
                fill=True,
                common_norm=False,
                palette=[MODEL_COLORS["BALM-PPI"], MODEL_COLORS["ESM2"]],
                alpha=0.5
            )
            
            # Customize plot
            metric_title = "RMSE" if metric == "rmse" else "Pearson Correlation"
            split_title = split_type.replace("_", " ").title()
            plt.title(f"{metric_title} Distribution by Model - {split_title}", fontsize=16)
            plt.xlabel(metric_title, fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.grid(linestyle='--', alpha=0.3)
            plt.legend(title="Model")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{split_type}_{metric}_density.png"), dpi=300)
            plt.close()
        
        # Create scatter plot of RMSE vs Sample Count
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(
            data=analysis_df,
            x="n_samples",
            y="rmse",
            hue="model",
            style="model",
            s=100,
            alpha=0.7,
            palette=[MODEL_COLORS["BALM-PPI"], MODEL_COLORS["ESM2"]]
        )
        
        # Add trend lines
        for model, color in MODEL_COLORS.items():
            model_data = analysis_df[analysis_df["model"] == model]
            if len(model_data) > 1:
                sns.regplot(
                    x="n_samples",
                    y="rmse",
                    data=model_data,
                    scatter=False,
                    color=color,
                    line_kws={"linestyle": "--"}
                )
        
        # Customize plot
        split_title = split_type.replace("_", " ").title()
        plt.title(f"RMSE vs Sample Count - {split_title}", fontsize=16)
        plt.xlabel("Number of Samples", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.grid(linestyle='--', alpha=0.3)
        plt.legend(title="Model")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{split_type}_rmse_vs_samples.png"), dpi=300)
        plt.close()
        
        # Create box plots for RMSE and Pearson
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            
            sns.boxplot(
                data=analysis_df,
                x="model",
                y=metric,
                palette=[MODEL_COLORS["BALM-PPI"], MODEL_COLORS["ESM2"]],
                width=0.5
            )
            
            # Add swarm plot
            sns.swarmplot(
                data=analysis_df,
                x="model",
                y=metric,
                color="black",
                alpha=0.5,
                size=4
            )
            
            # Customize plot
            metric_title = "RMSE" if metric == "rmse" else "Pearson Correlation"
            split_title = split_type.replace("_", " ").title()
            plt.title(f"{metric_title} Distribution by Model - {split_title}", fontsize=16)
            plt.xlabel("Model", fontsize=14)
            plt.ylabel(metric_title, fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{split_type}_{metric}_boxplot.png"), dpi=300)
            plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    all_results = load_all_results(args.results_dir)
    
    # Create plots
    create_cross_split_comparison(all_results, output_dir)
    create_model_performance_plots(all_results, output_dir)
    
    if all_results["target_analysis"]:
        create_generalizability_plots(all_results, output_dir)
    
    print(f"All visualizations have been saved to: {output_dir}")

if __name__ == "__main__":
    main()