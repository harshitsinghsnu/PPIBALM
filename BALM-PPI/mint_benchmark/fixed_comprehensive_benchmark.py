import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import yaml
import re
from collections import OrderedDict, defaultdict
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
import random
from Bio import pairwise2, SeqIO
from transformers import AutoTokenizer, AutoModel

# Set seaborn style for all plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SPLIT_TITLES = {
    "random": "Random Split",
    "cold_target": "Cold Target Split",
    "seq_similarity": "Sequence Similarity Split"
}

# -------------------- Configuration and Arguments --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive benchmark for protein-protein binding affinity prediction models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data CSV file")
    parser.add_argument("--balm_checkpoint", type=str, required=True, help="Path to BALM-PPI checkpoint")
    parser.add_argument("--balm_config", type=str, required=True, help="Path to BALM-PPI config file")
    parser.add_argument("--output_dir", type=str, default="./comprehensive_benchmark_results", 
                       help="Directory to save benchmark results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs with different random seeds")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated list of random seeds")
    parser.add_argument("--split_type", type=str, default="random", 
                       choices=["random", "cold_target", "seq_similarity"], 
                       help="Type of data split to use")
    parser.add_argument("--train_ratio", type=float, default=0.2, help="Ratio of data to use for training")
    parser.add_argument("--generate_plots", action="store_true", help="Generate plots for benchmarking results")
    parser.add_argument("--run_all_splits", action="store_true", help="Run all three data split strategies")
    return parser.parse_args()

# -------------------- Sequence Manipulation and Analysis --------------------

def calculate_sequence_identity(seq1, seq2):
    """Calculate sequence identity between two protein sequences"""
    # Align the sequences
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    
    # Count identical positions
    matches = sum(a == b for a, b in zip(alignment.seqA, alignment.seqB))
    
    # Calculate identity
    identity = matches / len(alignment.seqA)
    
    return identity

def cluster_sequences(sequences, n_clusters=10):
    """Cluster sequences based on simple k-mer frequency"""
    # Create a simple k-mer frequency feature
    k = 3
    all_kmers = set()
    
    # First pass to get all possible k-mers
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            all_kmers.add(seq[i:i+k])
    
    # Convert to sorted list for indexing
    all_kmers = sorted(list(all_kmers))
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    # Create feature matrix
    X = np.zeros((len(sequences), len(all_kmers)))
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:
                X[i, kmer_to_idx[kmer]] += 1
    
    # Normalize by sequence length
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            X[i] = X[i] / len(seq)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    return clusters

def create_data_splits(df, split_type, train_ratio, seed):
    """Create different types of data splits"""
    # Make sure there are appropriate columns
    required_cols = ["Target", "proteina", "Y"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Setup random state
    random.seed(seed)
    np.random.seed(seed)
    
    if split_type == "random":
        # Standard random split
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=seed)
        
        # Split train into train/val
        train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=seed)
        
    elif split_type == "cold_target":
        # Extract unique targets
        unique_targets = df["Target"].unique()
        
        # Cluster targets to create representative splits
        target_clusters = cluster_sequences(unique_targets, n_clusters=10)
        
        # Select clusters for training
        n_train_clusters = max(1, int(0.3 * 10))  # Use 30% of clusters for training
        train_clusters = random.sample(range(10), n_train_clusters)
        
        # Create masks
        train_mask = df["Target"].apply(lambda x: target_clusters[list(unique_targets).index(x)] in train_clusters)
        
        # Split data
        train_val_df = df[train_mask].reset_index(drop=True)
        test_df = df[~train_mask].reset_index(drop=True)
        
        # Ensure we have enough training data
        if len(train_val_df) < 100:
            # Fall back to random split if not enough data
            print("Warning: Not enough data for cold-target split, falling back to random split")
            return create_data_splits(df, "random", train_ratio, seed)
        
        # Split train into train/val
        train_df, val_df = train_test_split(train_val_df, train_size=0.8, random_state=seed)
        
    elif split_type == "seq_similarity":
        # Calculate pairwise sequence identity matrix (this could be computationally expensive)
        # For practicality, we'll use clustering as a proxy for sequence similarity
        
        # Cluster both target and proteina sequences
        unique_targets = df["Target"].unique()
        unique_proteina = df["proteina"].unique()
        
        target_clusters = cluster_sequences(unique_targets, n_clusters=10)
        proteina_clusters = cluster_sequences(unique_proteina, n_clusters=10)
        
        # Map sequences to cluster IDs
        target_to_cluster = {seq: cluster for seq, cluster in zip(unique_targets, target_clusters)}
        proteina_to_cluster = {seq: cluster for seq, cluster in zip(unique_proteina, proteina_clusters)}
        
        # Create combined cluster IDs for each row
        df["cluster_id"] = df.apply(
            lambda row: (target_to_cluster[row["Target"]], proteina_to_cluster[row["proteina"]]), 
            axis=1
        )
        
        # Get unique cluster pairs
        unique_cluster_pairs = df["cluster_id"].unique()
        
        # Select some cluster pairs for training
        n_train_pairs = max(1, int(train_ratio * len(unique_cluster_pairs)))
        train_cluster_pairs = random.sample(list(unique_cluster_pairs), n_train_pairs)
        
        # Create masks
        train_mask = df["cluster_id"].isin(train_cluster_pairs)
        
        # Split data
        train_val_df = df[train_mask].reset_index(drop=True)
        test_df = df[~train_mask].reset_index(drop=True)
        
        # Drop the temporary column
        train_val_df = train_val_df.drop("cluster_id", axis=1)
        test_df = test_df.drop("cluster_id", axis=1)
        
        # Ensure we have enough training data
        if len(train_val_df) < 100:
            # Fall back to random split if not enough data
            print("Warning: Not enough data for sequence similarity split, falling back to random split")
            return create_data_splits(df, "random", train_ratio, seed)
        
        # Split train into train/val
        train_df, val_df = train_test_split(train_val_df, train_size=0.8, random_state=seed)
    
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    print(f"Split data ({split_type}): {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    return train_df, val_df, test_df

# -------------------- Metrics Calculation --------------------

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for regression"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = stats.pearsonr(y_true, y_pred)[0]
    spearman = stats.spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    
    # Calculate concordance index (CI)
    def get_ci(y_true, y_pred):
        n = len(y_true)
        concordant = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if y_true[i] != y_true[j]:  # Only consider pairs with different true values
                    total_pairs += 1
                    if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                       (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                        concordant += 1
        
        return concordant / total_pairs if total_pairs > 0 else 0.5
    
    ci = get_ci(y_true, y_pred)
    
    metrics = {
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "R²": r2,
        "CI": ci
    }
    
    return metrics

# -------------------- BALM-PPI Evaluation --------------------

def load_balm_model(checkpoint_path, config_path, device):
    """Load a BALM-PPI model from a checkpoint"""
    # Import needed modules for BALM
    try:
        sys.path.append("D:/BALM_Fineclone/BALM-PPI")
        from balm.models import BALM
        from balm.configs import Configs
    except ImportError:
        raise ImportError("Could not import BALM-PPI modules. Make sure the path is correct.")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configs namespace
    configs = Configs(**config_dict)
    
    # Initialize model
    print("Initializing BALM-PPI model...")
    model = BALM(configs.model_configs)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading BALM-PPI checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def evaluate_balm(model, test_df, batch_size, device):
    """Evaluate BALM-PPI model on test data"""
    print("Evaluating BALM-PPI model...")
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[i:i+batch_size]
        
        # Scale targets to cosine similarity range
        data_min = test_df['Y'].min()
        data_max = test_df['Y'].max()
        
        batch_inputs = []
        for _, row in batch_df.iterrows():
            # Scale to cosine similarity range
            cosine_target = 2 * (row['Y'] - data_min) / (data_max - data_min) - 1
            
            batch_inputs.append({
                "protein_sequences": [row["Target"]],
                "proteina_sequences": [row["proteina"]],
                "labels": torch.tensor([cosine_target], dtype=torch.float32).to(device)
            })
        
        batch_predictions = []
        with torch.no_grad():
            for inputs in batch_inputs:
                output = model(inputs)
                
                # Convert cosine similarity back to pKd
                if "cosine_similarity" in output:
                    # Use model's conversion function if available
                    pred = model.cosine_similarity_to_pkd(
                        output["cosine_similarity"], 
                        pkd_upper_bound=data_max,
                        pkd_lower_bound=data_min
                    )
                else:
                    # Or use direct prediction if available
                    pred = output["logits"]
                
                batch_predictions.append(pred.item())
        
        predictions.extend(batch_predictions)
    
    return np.array(predictions)

# -------------------- ESM2 Evaluation --------------------

def get_esm2_embeddings(model, tokenizer, sequences, device):
    """Generate embeddings for protein sequences using ESM2"""
    embeddings = []
    batch_size = 4  # Use small batch size to avoid OOM
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating ESM2 embeddings"):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get embeddings from the last hidden state
            last_hidden_states = outputs.hidden_states[-1]
            # Average pooling over sequence length
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            seq_embeddings = (last_hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.append(seq_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)

def train_esm2_regressor(train_protein_embs, train_proteina_embs, y_train, val_protein_embs=None, val_proteina_embs=None, y_val=None):
    """Train a simple regressor on ESM2 embeddings"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Concatenate embeddings for protein pairs
    X_train = np.hstack([train_protein_embs, train_proteina_embs])
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Prepare validation data if provided
    if val_protein_embs is not None and val_proteina_embs is not None and y_val is not None:
        X_val = np.hstack([val_protein_embs, val_proteina_embs])
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None
    
    # Train regressor - GradientBoostingRegressor usually works well for this task
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_esm2(model, tokenizer, test_df, device):
    """Evaluate ESM2-based model on test data"""
    print("Evaluating ESM2-based model...")
    
    # Generate embeddings for test proteins
    test_protein_embs = get_esm2_embeddings(
        model, tokenizer, test_df["Target"].tolist(), device
    ).numpy()
    
    test_proteina_embs = get_esm2_embeddings(
        model, tokenizer, test_df["proteina"].tolist(), device
    ).numpy()
    
    return test_protein_embs, test_proteina_embs

# -------------------- Visualization and Results --------------------

def plot_regression(y_true, y_pred, title, output_path):
    """Create regression plot for predictions"""
    plt.figure(figsize=(10, 8))
    g = sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5})
    plt.title(title, fontsize=16)
    plt.xlabel('Actual pKd Values', fontsize=14)
    plt.ylabel('Predicted pKd Values', fontsize=14)
    
    # Calculate metrics for the plot
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics to plot
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Add diagonal reference line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return metrics

def plot_metrics_with_error_bars(all_metrics, output_dir):
    """Plot metrics with error bars from multiple runs"""
    # Prepare data for plotting
    metrics = list(all_metrics["BALM-PPI"][0].keys())
    models = ["BALM-PPI", "ESM2"]
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        # Get data for each model
        means = []
        stds = []
        
        for model in models:
            values = [run[metric] for run in all_metrics[model]]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        # Create bar plot with error bars
        x = np.arange(len(models))
        plt.bar(x, means, yerr=stds, capsize=10, width=0.4, color=COLORS[:len(models)])
        
        # Add values on top of bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            plt.text(x[j], mean + std + 0.02, f"{mean:.4f}±{std:.4f}", ha='center', va='bottom', fontsize=8)
        
        plt.title(metric)
        plt.xticks(x, models)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison_with_error_bars.png"), dpi=300)
    plt.close()

def create_comparison_plots(all_results_by_split, output_dir):
    """Create comparison plots across splits and models"""
    # Prepare data for plots
    metrics = ["RMSE", "Pearson", "Spearman", "R²", "CI"]
    splits = list(all_results_by_split.keys())
    models = ["BALM-PPI", "ESM2"]
    
    # 1. Model vs Split comparison for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        model_data = {model: [] for model in models}
        model_std = {model: [] for model in models}
        
        for split in splits:
            for model in models:
                values = [run[model + "_metrics"][metric] for run in all_results_by_split[split]]
                model_data[model].append(np.mean(values))
                model_std[model].append(np.std(values))
        
        # Create plot
        x = np.arange(len(splits))
        width = 0.35
        
        for i, model in enumerate(models):
            plt.bar(x + (i - 0.5) * width, model_data[model], width, 
                   label=model, color=COLORS[i], yerr=model_std[model], capsize=5)
        
        plt.xlabel('Split Type')
        plt.ylabel(metric)
        plt.title(f'{metric} by Split Type and Model')
        plt.xticks(x, [SPLIT_TITLES[s] for s in splits])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_by_split_{metric}.png"), dpi=300)
        plt.close()
    
    # 2. Combined metrics plot for each split
    for split in splits:
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        metric_data = {metric: [] for metric in metrics}
        metric_std = {metric: [] for metric in metrics}
        
        for metric in metrics:
            for model in models:
                values = [run[model + "_metrics"][metric] for run in all_results_by_split[split]]
                metric_data[metric].append(np.mean(values))
                metric_std[metric].append(np.std(values))
        
        # Create subplots for each metric
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            
            x = np.arange(len(models))
            plt.bar(x, metric_data[metric], yerr=metric_std[metric], capsize=5, color=COLORS[:len(models)])
            
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.title(f'{metric} - {SPLIT_TITLES[split]}')
            plt.xticks(x, models)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_by_model_{split}.png"), dpi=300)
        plt.close()
    
    # 3. Create summary heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    heatmap_data = []
    for split in splits:
        for metric in metrics:
            for model in models:
                values = [run[model + "_metrics"][metric] for run in all_results_by_split[split]]
                mean_val = np.mean(values)
                std_val = np.std(values)
                heatmap_data.append({
                    "Split": SPLIT_TITLES[split],
                    "Metric": metric,
                    "Model": model,
                    "Value": mean_val,
                    "Std": std_val
                })
    
    # Convert to DataFrame for seaborn
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create pivot table
    pivot_table = heatmap_df.pivot_table(
        index=["Split", "Metric"], 
        columns="Model", 
        values="Value"
    )
    
    # Create heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
    plt.title("Performance Comparison Across Splits and Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_heatmap.png"), dpi=300)
    plt.close()

def save_run_results(train_df, test_df, balm_preds, esm2_preds, balm_metrics, esm2_metrics, 
                   output_dir, seed, split_type):
    """Save results for a single run"""
    run_dir = os.path.join(output_dir, f"{split_type}_seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save prediction CSV
    results_df = test_df.copy()
    results_df['BALM_predictions'] = balm_preds
    results_df['ESM2_predictions'] = esm2_preds
    results_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(balm_metrics.keys()),
        'BALM-PPI': list(balm_metrics.values()),
        'ESM2': list(esm2_metrics.values()),
        'Difference': [esm2_metrics[k] - balm_metrics[k] for k in balm_metrics.keys()]
    })
    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    
    # Save data splits for reproducibility
    train_df.to_csv(os.path.join(run_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(run_dir, "test_split.csv"), index=False)
    
    # Save run metadata
    metadata = {
        "seed": seed,
        "split_type": split_type,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "balm_metrics": balm_metrics,
        "esm2_metrics": esm2_metrics
    }
    
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Create regression plots
    _ = plot_regression(
        test_df["Y"].values, 
        balm_preds, 
        f"BALM-PPI: Seed {seed}, {split_type.replace('_', ' ').title()} Split",
        os.path.join(run_dir, "balm_regression_plot.png")
    )
    
    _ = plot_regression(
        test_df["Y"].values, 
        esm2_preds, 
        f"ESM2: Seed {seed}, {split_type.replace('_', ' ').title()} Split",
        os.path.join(run_dir, "esm2_regression_plot.png")
    )
    
    # Compare error distributions
    plt.figure(figsize=(10, 6))
    
    balm_errors = np.abs(test_df['Y'].values - balm_preds)
    esm2_errors = np.abs(test_df['Y'].values - esm2_preds)
    
    sns.kdeplot(balm_errors, label='BALM-PPI', color=COLORS[0])
    sns.kdeplot(esm2_errors, label='ESM2', color=COLORS[1])
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title(f'Error Distribution Comparison (Seed {seed}, {split_type.replace("_", " ").title()} Split)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(run_dir, "error_distribution.png"), dpi=300)
    plt.close()
    
    return run_dir

def save_all_results(all_results, all_metrics, output_dir, split_type):
    """Save aggregated results from all runs"""
    # Create summary directory
    summary_dir = os.path.join(output_dir, f"{split_type}_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save mean and std of metrics across runs
    models = ["BALM-PPI", "ESM2"]
    metrics = list(all_metrics["BALM-PPI"][0].keys())
    
    summary_data = []
    
    for model in models:
        for metric in metrics:
            values = [run[metric] for run in all_metrics[model]]
            mean = np.mean(values)
            std = np.std(values)
            
            summary_data.append({
                "Model": model,
                "Metric": metric,
                "Mean": mean,
                "Std": std,
                "Min": np.min(values),
                "Max": np.max(values),
                "Values": str(values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(summary_dir, "metrics_summary.csv"), index=False)
    
    # Save consolidated predictions from all runs
    all_predictions = pd.DataFrame()
    
    for i, (seed, run_results) in enumerate(all_results.items()):
        test_df = run_results["test_df"].copy()
        test_df["BALM_predictions"] = run_results["balm_preds"]
        test_df["ESM2_predictions"] = run_results["esm2_preds"]
        test_df["seed"] = seed
        
        if i == 0:
            all_predictions = test_df
        else:
            all_predictions = pd.concat([all_predictions, test_df], ignore_index=True)
    
    all_predictions.to_csv(os.path.join(summary_dir, "all_predictions.csv"), index=False)
    
    # Create visualizations with error bars
    plot_metrics_with_error_bars(all_metrics, summary_dir)
    
    # Save all seeds results in pickle format for further analysis
    with open(os.path.join(summary_dir, "all_results.pkl"), "wb") as f:
        import pickle
        pickle.dump({
            "all_results": all_results,
            "all_metrics": all_metrics
        }, f)
    
    # Generate generalizability analysis if appropriate
    if split_type in ["cold_target", "seq_similarity"]:
        analyze_generalizability(all_results, all_metrics, summary_dir, split_type)
    
    # Create tabular summary in requested format
    create_tabular_summary(all_results, all_metrics, summary_dir, split_type)

def create_tabular_summary(all_results, all_metrics, output_dir, split_type):
    """Create tabular summary in the requested format"""
    # Format: Model,Split,Metric,Value
    summary_rows = []
    
    # Add split-level metrics
    for model in ["BALM-PPI", "ESM2"]:
        for metric in all_metrics[model][0].keys():
            values = [run[metric] for run in all_metrics[model]]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            summary_rows.append({
                "Model": model,
                "Split": split_type,
                "Metric": metric,
                "Value": mean_val,
                "Std": std_val
            })
    
    # Save as CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "model_split_metric_summary.csv"), index=False)
    
    # Also save in seed-level format
    # Model,Random seed,test/ci,test/loss,test/pearson,test/rmse,test/spearman
    seed_level_rows = []
    
    for seed, run_results in all_results.items():
        for model in ["BALM-PPI", "ESM2"]:
            model_key = f"{model.lower().replace('-', '_')}_metrics"
            if model == "BALM-PPI":
                metrics = run_results["balm_metrics"]
            else:
                metrics = run_results["esm2_metrics"]
                
            row = {
                "Model": model,
                "Random_seed": seed,
                "test/ci": metrics["CI"],
                "test/pearson": metrics["Pearson"],
                "test/rmse": metrics["RMSE"],
                "test/spearman": metrics["Spearman"],
                "test/r2": metrics["R²"]
            }
            seed_level_rows.append(row)
    
    # Save as CSV
    seed_level_df = pd.DataFrame(seed_level_rows)
    seed_level_df.to_csv(os.path.join(output_dir, "seed_level_metrics.csv"), index=False)

def analyze_generalizability(all_results, all_metrics, output_dir, split_type):
    """Analyze model generalizability on unseen targets or sequence similarity"""
    if split_type == "cold_target":
        print("Analyzing generalizability to unseen targets...")
    else:
        print("Analyzing generalizability across sequence similarity...")
    
    # Create data for analysis
    all_analysis_data = []
    
    for seed, run_results in all_results.items():
        test_df = run_results["test_df"]
        balm_preds = run_results["balm_preds"]
        esm2_preds = run_results["esm2_preds"]
        
        # Extract target information
        targets = test_df["Target"].unique()
        
        # Analyze performance by target
        for target in targets:
            target_mask = test_df["Target"] == target
            if sum(target_mask) < 5:  # Skip targets with few samples
                continue
                
            target_df = test_df[target_mask]
            target_balm_preds = balm_preds[target_mask]
            target_esm2_preds = esm2_preds[target_mask]
            
            # Calculate metrics for this target
            target_balm_metrics = calculate_metrics(target_df["Y"].values, target_balm_preds)
            target_esm2_metrics = calculate_metrics(target_df["Y"].values, target_esm2_preds)
            
            # Store for analysis
            all_analysis_data.append({
                "seed": seed,
                "target": target,
                "n_samples": sum(target_mask),
                "model": "BALM-PPI",
                "rmse": target_balm_metrics["RMSE"],
                "pearson": target_balm_metrics["Pearson"],
                "spearman": target_balm_metrics["Spearman"],
                "r2": target_balm_metrics["R²"],
                "ci": target_balm_metrics["CI"]
            })
            
            all_analysis_data.append({
                "seed": seed,
                "target": target,
                "n_samples": sum(target_mask),
                "model": "ESM2",
                "rmse": target_esm2_metrics["RMSE"],
                "pearson": target_esm2_metrics["Pearson"],
                "spearman": target_esm2_metrics["Spearman"],
                "r2": target_esm2_metrics["R²"],
                "ci": target_esm2_metrics["CI"]
            })
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(all_analysis_data)
    
    # Save the analysis
    analysis_df.to_csv(os.path.join(output_dir, "target_analysis.csv"), index=False)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE by sample count
    plt.subplot(2, 2, 1)
    sns.boxplot(x="model", y="rmse", data=analysis_df, palette=[COLORS[0], COLORS[1]])
    plt.title("RMSE Distribution by Model")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Pearson by sample count
    plt.subplot(2, 2, 2)
    sns.boxplot(x="model", y="pearson", data=analysis_df, palette=[COLORS[0], COLORS[1]])
    plt.title("Pearson Correlation Distribution by Model")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot RMSE histograms
    plt.subplot(2, 2, 3)
    sns.histplot(data=analysis_df, x="rmse", hue="model", element="step", 
                common_norm=False, bins=20, palette=[COLORS[0], COLORS[1]])
    plt.title("RMSE Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot Pearson histograms
    plt.subplot(2, 2, 4)
    sns.histplot(data=analysis_df, x="pearson", hue="model", element="step", 
                common_norm=False, bins=20, palette=[COLORS[0], COLORS[1]])
    plt.title("Pearson Correlation Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generalizability_analysis.png"), dpi=300)
    plt.close()
    
    # Comparison of model performance on the same targets
    target_comparison = []
    
    for seed in analysis_df["seed"].unique():
        seed_df = analysis_df[analysis_df["seed"] == seed]
        
        for target in seed_df["target"].unique():
            target_df = seed_df[seed_df["target"] == target]
            
            if len(target_df) == 2:  # Both models have results
                balm_row = target_df[target_df["model"] == "BALM-PPI"].iloc[0]
                esm2_row = target_df[target_df["model"] == "ESM2"].iloc[0]
                
                target_comparison.append({
                    "seed": seed,
                    "target": target,
                    "n_samples": balm_row["n_samples"],
                    "balm_rmse": balm_row["rmse"],
                    "esm2_rmse": esm2_row["rmse"],
                    "rmse_diff": esm2_row["rmse"] - balm_row["rmse"],
                    "balm_pearson": balm_row["pearson"],
                    "esm2_pearson": esm2_row["pearson"],
                    "pearson_diff": esm2_row["pearson"] - balm_row["pearson"],
                })
    
    comparison_df = pd.DataFrame(target_comparison)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison_by_target.csv"), index=False)
    
    # Plot the difference in performance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(comparison_df["rmse_diff"], kde=True, color=COLORS[2])
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("RMSE Difference (ESM2 - BALM-PPI)")
    plt.xlabel("RMSE Difference")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    sns.histplot(comparison_df["pearson_diff"], kde=True, color=COLORS[3])
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("Pearson Difference (ESM2 - BALM-PPI)")
    plt.xlabel("Pearson Difference")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_difference_by_target.png"), dpi=300)
    plt.close()
    
    # Generate target-specific statistics
    print("Target-specific performance summary:")
    print(f"Number of targets analyzed: {len(comparison_df['target'].unique())}")
    
    balm_better_rmse = sum(comparison_df["rmse_diff"] > 0)
    esm2_better_rmse = sum(comparison_df["rmse_diff"] < 0)
    print(f"RMSE: BALM-PPI better on {balm_better_rmse} targets, ESM2 better on {esm2_better_rmse} targets")
    
    balm_better_pearson = sum(comparison_df["pearson_diff"] < 0)
    esm2_better_pearson = sum(comparison_df["pearson_diff"] > 0)
    print(f"Pearson: BALM-PPI better on {balm_better_pearson} targets, ESM2 better on {esm2_better_pearson} targets")
    
    # Save these statistics
    with open(os.path.join(output_dir, "target_performance_summary.txt"), "w") as f:
        f.write(f"Target-specific performance summary:\n")
        f.write(f"Number of targets analyzed: {len(comparison_df['target'].unique())}\n\n")
        f.write(f"RMSE: BALM-PPI better on {balm_better_rmse} targets, ESM2 better on {esm2_better_rmse} targets\n")
        f.write(f"Pearson: BALM-PPI better on {balm_better_pearson} targets, ESM2 better on {esm2_better_pearson} targets\n")

# -------------------- Main Function --------------------

def run_single_benchmark(train_df, val_df, test_df, balm_model, esm2_model, tokenizer, batch_size, device, seed, split_type):
    """Run a single benchmark with given data splits and models"""
    print(f"\n--- Running benchmark with seed {seed}, {split_type} split ---")
    
    # Evaluate BALM-PPI
    balm_preds = evaluate_balm(balm_model, test_df, batch_size, device)
    
    # Evaluate ESM2
    # Generate embeddings for train/val/test sets
    print("Generating embeddings for ESM2 model...")
    train_protein_embs, train_proteina_embs = evaluate_esm2(
        esm2_model, tokenizer, train_df, device
    )
    
    val_protein_embs, val_proteina_embs = evaluate_esm2(
        esm2_model, tokenizer, val_df, device
    )
    
    test_protein_embs, test_proteina_embs = evaluate_esm2(
        esm2_model, tokenizer, test_df, device
    )
    
    # Train ESM2 regressor
    print("Training regressor on ESM2 embeddings...")
    esm2_regressor, esm2_scaler = train_esm2_regressor(
        train_protein_embs, 
        train_proteina_embs, 
        train_df["Y"].values,
        val_protein_embs,
        val_proteina_embs,
        val_df["Y"].values
    )
    
    # Make predictions
    X_test = np.hstack([test_protein_embs, test_proteina_embs])
    X_test_scaled = esm2_scaler.transform(X_test)
    esm2_preds = esm2_regressor.predict(X_test_scaled)
    
    # Calculate metrics
    balm_metrics = calculate_metrics(test_df["Y"].values, balm_preds)
    esm2_metrics = calculate_metrics(test_df["Y"].values, esm2_preds)
    
    # Print results
    print("\nBALM-PPI Results:")
    for k, v in balm_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nESM2 Results:")
    for k, v in esm2_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Return results
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "balm_preds": balm_preds,
        "esm2_preds": esm2_preds,
        "balm_metrics": balm_metrics,
        "esm2_metrics": esm2_metrics
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Extract seeds
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if len(seeds) < args.num_runs:
        # Generate additional seeds if needed
        additional_seeds = [np.random.randint(1000, 10000) for _ in range(args.num_runs - len(seeds))]
        seeds.extend(additional_seeds)
    elif len(seeds) > args.num_runs:
        # Truncate seeds if too many
        seeds = seeds[:args.num_runs]
    
    # Create output directory
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config["seeds"] = seeds
    with open(os.path.join(base_output_dir, "benchmark_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Check if required columns exist
    required_cols = ["Target", "proteina", "Y"]
    if not all(col in df.columns for col in required_cols):
        # Try to identify likely columns
        print("Warning: Expected columns 'Target', 'proteina', and 'Y' not found in dataset")
        print("Available columns:", df.columns.tolist())
        
        # Find sequence columns
        seq_cols = []
        for col in df.columns:
            if df[col].dtype == object and len(df[col].iloc[0]) > 20:
                seq_cols.append(col)
        
        # Find numeric column for binding affinity
        y_col = None
        for col in df.columns:
            if col not in seq_cols and df[col].dtype in [np.float64, np.int64]:
                y_col = col
                break
        
        if len(seq_cols) >= 2 and y_col:
            print(f"Using {seq_cols[0]} as Target, {seq_cols[1]} as proteina, and {y_col} as Y")
            df = df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina", y_col: "Y"})
        else:
            raise ValueError("Could not identify appropriate columns in dataset")
    
    # Load BALM-PPI model
    try:
        balm_model = load_balm_model(args.balm_checkpoint, args.balm_config, device)
    except Exception as e:
        print(f"Error loading BALM-PPI model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load ESM2 model
    try:
        print("Loading ESM2 model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm2_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        esm2_model = esm2_model.to(device)
        esm2_model.eval()
    except Exception as e:
        print(f"Error loading ESM2 model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Define splits to benchmark
    if args.run_all_splits:
        split_types = ["random", "cold_target", "seq_similarity"]
    else:
        split_types = [args.split_type]
    
    # Store results from all splits
    all_results_by_split = {}
    
    # Run benchmarks for each split type
    for split_type in split_types:
        print(f"\n\n=== Running benchmarks for {split_type} split ===\n")
        
        output_dir = os.path.join(base_output_dir, f"{split_type}_split")
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        all_metrics = {
            "BALM-PPI": [],
            "ESM2": []
        }
        
        # Run multiple benchmarks with different seeds
        for seed in seeds:
            # Create data split
            train_df, val_df, test_df = create_data_splits(df, split_type, args.train_ratio, seed)
            
            # Run benchmark
            run_results = run_single_benchmark(
                train_df, val_df, test_df, balm_model, esm2_model, tokenizer,
                args.batch_size, device, seed, split_type
            )
            
            # Save individual run results
            run_dir = save_run_results(
                train_df, test_df, 
                run_results["balm_preds"], run_results["esm2_preds"],
                run_results["balm_metrics"], run_results["esm2_metrics"],
                output_dir, seed, split_type
            )
            
            # Store results for aggregation
            all_results[seed] = run_results
            all_metrics["BALM-PPI"].append(run_results["balm_metrics"])
            all_metrics["ESM2"].append(run_results["esm2_metrics"])
            
            print(f"Run with seed {seed} completed. Results saved to {run_dir}")
        
        # Save aggregated results
        save_all_results(all_results, all_metrics, output_dir, split_type)
        
        # Store for cross-split comparison
        all_results_by_split[split_type] = []
        for seed in seeds:
            all_results_by_split[split_type].append({
                "seed": seed,
                "BALM-PPI_metrics": all_results[seed]["balm_metrics"],
                "ESM2_metrics": all_results[seed]["esm2_metrics"]
            })
        
        print(f"\nAll benchmarks for {split_type} split completed. Results saved to {output_dir}")
        print(f"Run summary ({split_type} split, {len(seeds)} seeds):")
        
        # Print final summary
        print("\nBALM-PPI Average Metrics:")
        for metric in all_metrics["BALM-PPI"][0].keys():
            values = [run[metric] for run in all_metrics["BALM-PPI"]]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        print("\nESM2 Average Metrics:")
        for metric in all_metrics["ESM2"][0].keys():
            values = [run[metric] for run in all_metrics["ESM2"]]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # Create cross-split comparison if multiple splits were run
    if len(split_types) > 1:
        print("\n=== Creating cross-split comparison ===")
        create_comparison_plots(all_results_by_split, base_output_dir)
        print(f"Cross-split comparison created. Results saved to {base_output_dir}")

if __name__ == "__main__":
    main()