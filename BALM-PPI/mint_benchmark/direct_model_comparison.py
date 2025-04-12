import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import json
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# -------------------- Configuration and Arguments --------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Direct comparison between BALM-PPI and MINT")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV file")
    parser.add_argument("--balm_checkpoint", type=str, required=True, help="Path to BALM-PPI checkpoint")
    parser.add_argument("--balm_config", type=str, required=True, help="Path to BALM-PPI config file")
    parser.add_argument("--mint_checkpoint", type=str, required=True, help="Path to MINT checkpoint")
    parser.add_argument("--mint_config", type=str, required=True, help="Path to MINT config file")
    parser.add_argument("--output_dir", type=str, default="./direct_comparison_results", 
                       help="Directory to save comparison results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    return parser.parse_args()

# -------------------- Metrics Calculation --------------------

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for regression"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = stats.pearsonr(y_true, y_pred)[0]
    spearman = stats.spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    
    # Calculate concordance index (CI)
    def get_ci(y, f):
        ind = np.argsort(y)
        y = y[ind]
        f = f[ind]
        
        # Calculate the total number of comparable pairs
        z = np.sum(y[:-1, None] < y[1:, None]).astype(float)
        
        # Calculate the number of concordant pairs
        S = np.sum((y[:-1, None] < y[1:, None]) * (f[:-1, None] < f[1:, None]))
        
        return S / z
    
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

# -------------------- MINT Evaluation --------------------

def setup_mint_model(config_path, checkpoint_path, device):
    """Set up MINT model"""
    # Add MINT to path and import
    try:
        mint_dir = "D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts"
        if mint_dir not in sys.path:
            sys.path.append(mint_dir)
        
        # Import MINT modules
        import mint
        from mint.model.esm2 import ESM2
    except ImportError:
        raise ImportError("Could not import MINT modules. Make sure the path is correct.")
    
    # Load config
    with open(config_path) as f:
        cfg_dict = json.load(f)
        cfg = argparse.Namespace()
        cfg.__dict__.update(cfg_dict)
    
    # Create model wrapper
    class MINTWrapper(torch.nn.Module):
        def __init__(self, cfg, checkpoint_path, device):
            super().__init__()
            self.cfg = cfg
            self.model = ESM2(
                num_layers=cfg.encoder_layers,
                embed_dim=cfg.encoder_embed_dim,
                attention_heads=cfg.encoder_attention_heads,
                token_dropout=cfg.token_dropout,
                use_multimer=True
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Process checkpoint
            from collections import OrderedDict
            new_checkpoint = OrderedDict(
                (key.replace("model.", ""), value)
                for key, value in checkpoint["state_dict"].items()
            )
            
            self.model.load_state_dict(new_checkpoint)
            self.model.to(device)
            self.model.eval()
            
            # Set up alphabet for tokenization
            self.alphabet = mint.data.Alphabet.from_architecture("ESM-1b")
        
        def tokenize(self, sequences):
            """Tokenize sequences for MINT"""
            batch_size = len(sequences)
            seq_encoded_list = [
                self.alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
                for seq in sequences
            ]
            
            max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
            tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
            tokens.fill_(self.alphabet.padding_idx)
            
            for i, seq_encoded in enumerate(seq_encoded_list):
                seq = torch.tensor(seq_encoded, dtype=torch.int64)
                tokens[i, : len(seq_encoded)] = seq
            
            return tokens
        
        def embed_pair(self, seq1, seq2):
            """Generate embeddings for a pair of sequences"""
            # Tokenize
            tokens1 = self.tokenize([seq1])
            tokens2 = self.tokenize([seq2])
            
            # Prepare chain IDs
            chain_ids1 = torch.zeros_like(tokens1, dtype=torch.int32)
            chain_ids2 = torch.ones_like(tokens2, dtype=torch.int32)
            
            # Concatenate
            tokens = torch.cat([tokens1, tokens2], dim=1)
            chain_ids = torch.cat([chain_ids1, chain_ids2], dim=1)
            
            # Move to device
            tokens = tokens.to(next(self.model.parameters()).device)
            chain_ids = chain_ids.to(next(self.model.parameters()).device)
            
            # Get embeddings
            with torch.no_grad():
                results = self.model(tokens, chain_ids, repr_layers=[33])
                embeddings = results["representations"][33]
            
            # Mask embeddings
            mask = (
                (~tokens.eq(self.model.cls_idx))
                & (~tokens.eq(self.model.eos_idx))
                & (~tokens.eq(self.model.padding_idx))
            )
            
            # Extract embeddings for each chain
            mask_chain_0 = (chain_ids.eq(0) & mask).unsqueeze(-1).expand_as(embeddings)
            mask_chain_1 = (chain_ids.eq(1) & mask).unsqueeze(-1).expand_as(embeddings)
            
            # For chain 0
            masked_chain_0 = embeddings * mask_chain_0
            sum_masked_0 = masked_chain_0.sum(dim=1)
            mask_counts_0 = (chain_ids.eq(0) & mask).sum(dim=1, keepdim=True).float()
            mean_chain_0 = sum_masked_0 / mask_counts_0
            
            # For chain 1
            masked_chain_1 = embeddings * mask_chain_1
            sum_masked_1 = masked_chain_1.sum(dim=1)
            mask_counts_1 = (chain_ids.eq(1) & mask).sum(dim=1, keepdim=True).float()
            mean_chain_1 = sum_masked_1 / mask_counts_1
            
            # Concatenate
            return torch.cat([mean_chain_0, mean_chain_1], dim=1)
    
    # Initialize model
    print("Setting up MINT model...")
    model = MINTWrapper(cfg, checkpoint_path, device)
    return model

def train_mint_regressor(model, train_df, device):
    """Train a regressor on MINT embeddings"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    
    print("Training regression model on MINT embeddings...")
    
    # Generate embeddings for training data
    train_embeddings = []
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        with torch.no_grad():
            emb = model.embed_pair(row["Target"], row["proteina"])
        train_embeddings.append(emb.cpu().numpy())
    
    train_embeddings = np.vstack(train_embeddings)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_embeddings)
    
    # Train model
    regressor = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
    
    regressor.fit(X_train_scaled, train_df["Y"].values)
    
    return regressor, scaler

def evaluate_mint(model, regressor, scaler, test_df, batch_size, device):
    """Evaluate MINT on test data"""
    print("Evaluating MINT model...")
    test_embeddings = []
    
    # Generate embeddings in batches
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            with torch.no_grad():
                emb = model.embed_pair(row["Target"], row["proteina"])
            test_embeddings.append(emb.cpu().numpy())
    
    test_embeddings = np.vstack(test_embeddings)
    
    # Scale features
    X_test_scaled = scaler.transform(test_embeddings)
    
    # Make predictions
    predictions = regressor.predict(X_test_scaled)
    
    return predictions

# -------------------- Visualization --------------------

def plot_regression(y_true, y_pred, title, output_path):
    """Create regression plot for predictions"""
    plt.figure(figsize=(10, 8))
    sns.regplot(x=y_true, y=y_pred)
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Calculate metrics for the plot
    metrics = calculate_metrics(y_true, y_pred)
    
    # Add metrics to plot
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.savefig(output_path)
    plt.close()
    
    return metrics

def plot_comparison(balm_metrics, mint_metrics, output_path):
    """Create comparison bar plot for metrics"""
    # Prepare data for plotting
    metrics = list(balm_metrics.keys())
    balm_values = [balm_metrics[m] for m in metrics]
    mint_values = [mint_metrics[m] for m in metrics]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, balm_values, width, label='BALM-PPI')
    plt.bar(x + width/2, mint_values, width, label='MINT')
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('BALM-PPI vs MINT Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(balm_values):
        plt.text(i - width/2, v + 0.02, f"{v:.4f}", ha='center')
    
    for i, v in enumerate(mint_values):
        plt.text(i + width/2, v + 0.02, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_comparison_results(test_df, balm_preds, mint_preds, balm_metrics, mint_metrics, output_dir):
    """Save all comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prediction CSV
    results_df = test_df.copy()
    results_df['BALM_predictions'] = balm_preds
    results_df['MINT_predictions'] = mint_preds
    results_df.to_csv(os.path.join(output_dir, "prediction_comparison.csv"), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': list(balm_metrics.keys()),
        'BALM-PPI': list(balm_metrics.values()),
        'MINT': list(mint_metrics.values()),
        'Difference': [mint_metrics[k] - balm_metrics[k] for k in balm_metrics.keys()]
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    
    # Save summary text
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
        f.write("BALM-PPI vs MINT Performance Comparison\n")
        f.write("=======================================\n\n")
        
        f.write("BALM-PPI Metrics:\n")
        for k, v in balm_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nMINT Metrics:\n")
        for k, v in mint_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nDifference (MINT - BALM-PPI):\n")
        for k in balm_metrics.keys():
            diff = mint_metrics[k] - balm_metrics[k]
            better = "better" if ((k == "RMSE" and diff < 0) or (k != "RMSE" and diff > 0)) else "worse"
            f.write(f"  {k}: {diff:.4f} ({better})\n")
    
    # Create error distribution plot
    plt.figure(figsize=(10, 6))
    
    balm_errors = np.abs(test_df['Y'].values - balm_preds)
    mint_errors = np.abs(test_df['Y'].values - mint_preds)
    
    sns.kdeplot(balm_errors, label='BALM-PPI')
    sns.kdeplot(mint_errors, label='MINT')
    
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    plt.close()
    
    print(f"All comparison results saved to {output_dir}")

# -------------------- Main Function --------------------

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    # Make a copy for training MINT regressor (take a small subset for efficiency)
    train_size = min(2000, len(test_df) // 5)  # Use at most 2000 samples or 20% of test data
    train_df = test_df.sample(n=train_size, random_state=42)
    print(f"Using {len(train_df)} samples for training MINT regressor")
    
    # Save train/test splits for reproducibility
    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test_split.csv"), index=False)
    
    # Check if required columns exist
    if "Target" not in test_df.columns or "proteina" not in test_df.columns or "Y" not in test_df.columns:
        # Try to identify likely columns
        print("Warning: Expected columns 'Target', 'proteina', and 'Y' not found in test data")
        print("Available columns:", test_df.columns.tolist())
        
        # Find sequence columns
        seq_cols = []
        for col in test_df.columns:
            if test_df[col].dtype == object and len(test_df[col].iloc[0]) > 20:
                seq_cols.append(col)
        
        # Find numeric column for binding affinity
        y_col = None
        for col in test_df.columns:
            if col not in seq_cols and test_df[col].dtype in [np.float64, np.int64]:
                y_col = col
                break
        
        if len(seq_cols) >= 2 and y_col:
            print(f"Using {seq_cols[0]} as Target, {seq_cols[1]} as proteina, and {y_col} as Y")
            test_df = test_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina", y_col: "Y"})
            train_df = train_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina", y_col: "Y"})
        else:
            raise ValueError("Could not identify appropriate columns in test data")
    
    # Load BALM-PPI model
    try:
        balm_model = load_balm_model(args.balm_checkpoint, args.balm_config, device)
        # Evaluate BALM-PPI
        balm_preds = evaluate_balm(balm_model, test_df, args.batch_size, device)
        # Plot BALM-PPI results
        balm_metrics = plot_regression(
            test_df["Y"].values, 
            balm_preds, 
            "BALM-PPI: Predicted vs Actual Values",
            os.path.join(args.output_dir, "balm_regression_plot.png")
        )
        # Save BALM results
        test_df["BALM_predictions"] = balm_preds
        test_df[["Target", "proteina", "Y", "BALM_predictions"]].to_csv(
            os.path.join(args.output_dir, "balm_results.csv"), index=False
        )
        
        print("\nBALM-PPI Results:")
        for k, v in balm_metrics.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Error evaluating BALM-PPI: {str(e)}")
        balm_preds = np.zeros(len(test_df))
        balm_metrics = {
            "RMSE": 0.0,
            "Pearson": 0.0,
            "Spearman": 0.0,
            "R²": 0.0,
            "CI": 0.0
        }
    
    # Load MINT model
    try:
        mint_model = setup_mint_model(args.mint_config, args.mint_checkpoint, device)
        # Train regressor on MINT embeddings
        mint_regressor, mint_scaler = train_mint_regressor(mint_model, train_df, device)
        # Evaluate MINT
        mint_preds = evaluate_mint(mint_model, mint_regressor, mint_scaler, test_df, args.batch_size, device)
        # Plot MINT results
        mint_metrics = plot_regression(
            test_df["Y"].values, 
            mint_preds, 
            "MINT: Predicted vs Actual Values",
            os.path.join(args.output_dir, "mint_regression_plot.png")
        )
        
        print("\nMINT Results:")
        for k, v in mint_metrics.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Error evaluating MINT: {str(e)}")
        mint_preds = np.zeros(len(test_df))
        mint_metrics = {
            "RMSE": 0.0,
            "Pearson": 0.0,
            "Spearman": 0.0,
            "R²": 0.0,
            "CI": 0.0
        }
    
    # Compare results
    try:
        # Create comparison plots
        plot_comparison(balm_metrics, mint_metrics, os.path.join(args.output_dir, "metrics_comparison.png"))
        
        # Create side-by-side regression plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # BALM-PPI plot
        sns.regplot(x=test_df["Y"].values, y=balm_preds, ax=ax1)
        ax1.set_title('BALM-PPI: Predicted vs Actual Values')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        
        # Add metrics to BALM plot
        balm_metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in balm_metrics.items()])
        ax1.annotate(balm_metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # MINT plot
        sns.regplot(x=test_df["Y"].values, y=mint_preds, ax=ax2)
        ax2.set_title('MINT: Predicted vs Actual Values')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        
        # Add metrics to MINT plot
        mint_metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in mint_metrics.items()])
        ax2.annotate(mint_metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_regression_plots.png"))
        plt.close()
        
        # Save all results
        save_comparison_results(test_df, balm_preds, mint_preds, balm_metrics, mint_metrics, args.output_dir)
    except Exception as e:
        print(f"Error creating comparison results: {str(e)}")

if __name__ == "__main__":
    main()