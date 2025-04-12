# benchmark_mint_vs_balm_updated.py
import os
import sys
import time
import json
import argparse
import random  # Make sure this is imported at the top
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge

# Add MINT path to Python path
mint_path = os.path.abspath("D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts/mint")
if mint_path not in sys.path:
    sys.path.append(mint_path)
    print(f"Added MINT path to Python path: {mint_path}")

# Add BALM-PPI path to Python path
balm_path = os.path.abspath("D:/BALM_Fineclone/BALM-PPI")
if balm_path not in sys.path:
    sys.path.append(balm_path)
    print(f"Added BALM path to Python path: {balm_path}")

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_data_splits(data_path, output_dir, seed, test_size=0.8):
    """Create train-test splits and save them"""
    print(f"Creating data splits from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"Created splits: {len(train_df)} train, {len(test_df)} test examples")
    return train_df, test_df

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    
    # Calculate CI (concordance index)
    indices = np.argsort(y_true)
    y_true_sorted = y_true[indices]
    y_pred_sorted = y_pred[indices]
    
    concordant_pairs = 0
    total_pairs = 0
    
    for i in range(len(y_true_sorted)-1):
        for j in range(i+1, len(y_true_sorted)):
            if y_true_sorted[i] < y_true_sorted[j]:
                total_pairs += 1
                if y_pred_sorted[i] < y_pred_sorted[j]:
                    concordant_pairs += 1
    
    ci = concordant_pairs / total_pairs if total_pairs > 0 else 0
    
    return {
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "CI": ci
    }

# Define a custom CollateFn class to avoid import issues
class CustomCollateFn:
    def __init__(self, truncation_seq_length=None):
        # Import here to ensure mint is in the path
        import mint.data
        self.alphabet = mint.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        heavy_chain, light_chain = zip(*batches)
        chains = [self.convert(c) for c in [heavy_chain, light_chain]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        return chains, chain_ids

    def convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens

def run_balm_benchmark(train_df, test_df, config_path, checkpoint_path, output_dir, seed):
    """Run benchmark for BALM model"""
    print("Starting BALM benchmark...")
    start_time = time.time()
    
    try:
        # Import BALM components
        try:
            from balm.common_utils import load_yaml
            from balm.configs import Configs
            from balm.models import BALM
            print("Successfully imported BALM")
        except ImportError as e:
            print(f"Error importing BALM: {e}")
            print("Available paths:", sys.path)
            raise
        
        # Load config
        configs = Configs(**load_yaml(config_path))
        
        # Initialize model
        model = BALM(configs.model_configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading BALM checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Prepare data
        train_embs = []
        test_embs = []
        train_labels = []
        test_labels = []
        
        model.eval()
        with torch.no_grad():
            # Process training data
            print(f"Processing {len(train_df)} training examples...")
            for _, row in train_df.iterrows():
                try:
                    inputs = {
                        "protein_sequences": [row["Target"]],
                        "proteina_sequences": [row["proteina"]]
                    }
                    outputs = model(inputs)
                    embedding = torch.cat([outputs["protein_embedding"], outputs["proteina_embedding"]], dim=-1)
                    train_embs.append(embedding.cpu().numpy())
                    train_labels.append(row["Y"])
                except Exception as e:
                    print(f"Error processing training row: {e}")
                
            # Process test data
            print(f"Processing {len(test_df)} test examples...")
            for _, row in test_df.iterrows():
                try:
                    inputs = {
                        "protein_sequences": [row["Target"]],
                        "proteina_sequences": [row["proteina"]]
                    }
                    outputs = model(inputs)
                    embedding = torch.cat([outputs["protein_embedding"], outputs["proteina_embedding"]], dim=-1)
                    test_embs.append(embedding.cpu().numpy())
                    test_labels.append(row["Y"])
                except Exception as e:
                    print(f"Error processing test row: {e}")
        
        train_embs = np.vstack(train_embs)
        test_embs = np.vstack(test_embs)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        print(f"Extracted embeddings: train {train_embs.shape}, test {test_embs.shape}")
        
        # Train a ridge regression model on the embeddings
        print("Training ridge regression model...")
        ridge_model = Ridge(alpha=1.0, random_state=seed)
        ridge_model.fit(train_embs, train_labels)
        
        # Make predictions
        train_preds = ridge_model.predict(train_embs)
        test_preds = ridge_model.predict(test_embs)
        
        # Calculate metrics
        train_metrics = evaluate_metrics(train_labels, train_preds)
        test_metrics = evaluate_metrics(test_labels, test_preds)
        
        print(f"BALM test metrics: RMSE={test_metrics['RMSE']:.3f}, Pearson={test_metrics['Pearson']:.3f}")
        
        # Save results
        os.makedirs(os.path.join(output_dir, "balm"), exist_ok=True)
        
        pd.DataFrame({
            "True": test_labels,
            "Predicted": test_preds
        }).to_csv(os.path.join(output_dir, "balm", "predictions.csv"), index=False)
        
        with open(os.path.join(output_dir, "balm", "metrics.json"), "w") as f:
            json.dump({
                "train": train_metrics,
                "test": test_metrics,
                "time": time.time() - start_time,
                "embedding_dims": train_embs.shape[1]
            }, f, indent=2)
        
        # Plot regression
        plt.figure(figsize=(10, 8))
        sns.regplot(x=test_labels, y=test_preds)
        plt.title(f"BALM - Test RMSE: {test_metrics['RMSE']:.3f}, Pearson: {test_metrics['Pearson']:.3f}")
        plt.xlabel("True pKd")
        plt.ylabel("Predicted pKd")
        plt.savefig(os.path.join(output_dir, "balm", "regression_plot.png"))
        plt.close()
        
        return test_metrics
    
    except Exception as e:
        print(f"Error in BALM benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_mint_benchmark(train_df, test_df, config_path, checkpoint_path, output_dir, seed):
    """Run benchmark for MINT model"""
    print("Starting MINT benchmark...")
    start_time = time.time()
    
    try:
        # Import MINT components
        try:
            import mint
            from mint.helpers.extract import load_config, MINTWrapper
            print("Successfully imported MINT")
        except ImportError as e:
            print(f"Error importing MINT: {e}")
            print("Available paths:", sys.path)
            raise
        
        # Load config
        cfg = load_config(config_path)
        
        # Initialize model wrapper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print(f"Loading MINT model from checkpoint: {checkpoint_path}")
        wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)
        
        # Prepare data
        train_embs = []
        test_embs = []
        train_labels = []
        test_labels = []
        
        # Define our custom collate function
        collate_fn = CustomCollateFn(512)
        
        # Process training data
        print(f"Processing {len(train_df)} training examples...")
        for _, row in train_df.iterrows():
            try:
                chains, chain_ids = collate_fn([(row["Target"], row["proteina"])])
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                
                with torch.no_grad():
                    embedding = wrapper(chains, chain_ids)
                    train_embs.append(embedding.cpu().numpy())
                    train_labels.append(row["Y"])
            except Exception as e:
                print(f"Error processing training row: {e}")
        
        # Process test data
        print(f"Processing {len(test_df)} test examples...")
        for _, row in test_df.iterrows():
            try:
                chains, chain_ids = collate_fn([(row["Target"], row["proteina"])])
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                
                with torch.no_grad():
                    embedding = wrapper(chains, chain_ids)
                    test_embs.append(embedding.cpu().numpy())
                    test_labels.append(row["Y"])
            except Exception as e:
                print(f"Error processing test row: {e}")
        
        train_embs = np.vstack(train_embs)
        test_embs = np.vstack(test_embs)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        print(f"Extracted embeddings: train {train_embs.shape}, test {test_embs.shape}")
        
        # Train a ridge regression model on the embeddings
        print("Training ridge regression model...")
        ridge_model = Ridge(alpha=1.0, random_state=seed)
        ridge_model.fit(train_embs, train_labels)
        
        # Make predictions
        train_preds = ridge_model.predict(train_embs)
        test_preds = ridge_model.predict(test_embs)
        
        # Calculate metrics
        train_metrics = evaluate_metrics(train_labels, train_preds)
        test_metrics = evaluate_metrics(test_labels, test_preds)
        
        print(f"MINT test metrics: RMSE={test_metrics['RMSE']:.3f}, Pearson={test_metrics['Pearson']:.3f}")
        
        # Save results
        os.makedirs(os.path.join(output_dir, "mint"), exist_ok=True)
        
        pd.DataFrame({
            "True": test_labels,
            "Predicted": test_preds
        }).to_csv(os.path.join(output_dir, "mint", "predictions.csv"), index=False)
        
        with open(os.path.join(output_dir, "mint", "metrics.json"), "w") as f:
            json.dump({
                "train": train_metrics,
                "test": test_metrics,
                "time": time.time() - start_time,
                "embedding_dims": train_embs.shape[1]
            }, f, indent=2)
        
        # Plot regression
        plt.figure(figsize=(10, 8))
        sns.regplot(x=test_labels, y=test_preds)
        plt.title(f"MINT - Test RMSE: {test_metrics['RMSE']:.3f}, Pearson: {test_metrics['Pearson']:.3f}")
        plt.xlabel("True pKd")
        plt.ylabel("Predicted pKd")
        plt.savefig(os.path.join(output_dir, "mint", "regression_plot.png"))
        plt.close()
        
        return test_metrics
    
    except Exception as e:
        print(f"Error in MINT benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(balm_metrics, mint_metrics, output_dir):
    """Compare results between BALM and MINT"""
    metrics = ["RMSE", "Pearson", "Spearman", "CI"]
    
    comparison = pd.DataFrame({
        "Metric": metrics,
        "BALM": [balm_metrics[m] for m in metrics],
        "MINT": [mint_metrics[m] for m in metrics]
    })
    
    comparison.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        data = [balm_metrics[metric], mint_metrics[metric]]
        bars = plt.bar(["BALM", "MINT"], data)
        plt.title(metric)
        plt.ylim(0, max(data) * 1.2)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
    plt.close()
    
    # Create a more detailed comparison with bar chart and error distribution
    plt.figure(figsize=(18, 6))
    
    # Bar chart for metrics
    plt.subplot(1, 3, 1)
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [balm_metrics[m] for m in metrics], width, label='BALM')
    plt.bar(x + width/2, [mint_metrics[m] for m in metrics], width, label='MINT')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Comparison of Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Summary in text
    plt.subplot(1, 3, 2)
    plt.axis('off')
    summary_text = "Model Comparison Summary:\n\n"
    
    for metric in metrics:
        balm_value = balm_metrics[metric]
        mint_value = mint_metrics[metric]
        diff = mint_value - balm_value
        diff_percent = (diff / balm_value) * 100
        
        better = "MINT" if (metric != "RMSE" and diff > 0) or (metric == "RMSE" and diff < 0) else "BALM"
        
        summary_text += f"{metric}:\n"
        summary_text += f"  BALM: {balm_value:.4f}\n"
        summary_text += f"  MINT: {mint_value:.4f}\n"
        summary_text += f"  Diff: {diff:.4f} ({diff_percent:.2f}%)\n"
        summary_text += f"  Better: {better}\n\n"
    
    plt.text(0, 1, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_comparison.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Benchmark MINT against BALM results")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--balm_config", type=str, default="D:/BALM_Fineclone/BALM-PPI/default_configs/balm_peft.yaml", help="Path to BALM config")
    parser.add_argument("--balm_checkpoint", type=str, default="D:/BALM_Fineclone/BALM-PPI/outputs/latest_checkpoint.pth", help="Path to BALM checkpoint")
    parser.add_argument("--mint_config", type=str, default="D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts/mint/data/esm2_t33_650M_UR50D.json", help="Path to MINT config")
    parser.add_argument("--mint_checkpoint", type=str, default="D:/BALM_Fineclone/BALM-PPI/Benchmark/models/mint.ckpt", help="Path to MINT checkpoint")
    parser.add_argument("--output_dir", type=str, default="mint_benchmark_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.8, help="Test split size")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data splits
    train_df, test_df = create_data_splits(args.data_path, args.output_dir, args.seed, args.test_size)
    
    # Run BALM benchmark
    balm_metrics = run_balm_benchmark(
        train_df,
        test_df,
        args.balm_config,
        args.balm_checkpoint,
        args.output_dir,
        args.seed
    )
    
    # Run MINT benchmark
    mint_metrics = run_mint_benchmark(
        train_df,
        test_df,
        args.mint_config,
        args.mint_checkpoint,
        args.output_dir,
        args.seed
    )
    
    if balm_metrics and mint_metrics:
        # Compare results
        compare_results(balm_metrics, mint_metrics, args.output_dir)
        
        # Save summary
        with open(os.path.join(args.output_dir, "comparison_summary.txt"), "w") as f:
            f.write(f"Benchmark with seed {args.seed}\n")
            f.write(f"Data: {args.data_path}\n")
            f.write(f"Test size: {args.test_size}\n\n")
            f.write("BALM Results:\n")
            for metric, value in balm_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\nMINT Results:\n")
            for metric, value in mint_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            # Add comparison
            f.write("\nComparison:\n")
            for metric in ["RMSE", "Pearson", "Spearman", "CI"]:
                balm_value = balm_metrics[metric]
                mint_value = mint_metrics[metric]
                diff = mint_value - balm_value
                diff_percent = (diff / balm_value) * 100
                
                better = "MINT" if (metric != "RMSE" and diff > 0) or (metric == "RMSE" and diff < 0) else "BALM"
                
                f.write(f"  {metric}:\n")
                f.write(f"    BALM: {balm_value:.4f}\n")
                f.write(f"    MINT: {mint_value:.4f}\n")
                f.write(f"    Diff: {diff:.4f} ({diff_percent:.2f}%)\n")
                f.write(f"    Better: {better}\n\n")

if __name__ == "__main__":
    main()