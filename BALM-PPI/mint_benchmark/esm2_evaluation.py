import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="ESM2 Evaluation")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data CSV file")
    parser.add_argument("--output_dir", type=str, default="./esm2_comparison_results", 
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    return parser.parse_args()

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
        z = np.sum(y[:-1, None] < y[1:, None]).astype(float)
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

def get_esm2_embeddings(model, tokenizer, sequences, device, batch_size=2):
    """Generate protein embeddings using ESM2"""
    embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get embeddings from the last layer
            last_hidden = outputs.hidden_states[-1]
            
            # Average over sequence length (excluding padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            seq_embeddings = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            embeddings.append(seq_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    # Make a copy for training the regressor
    train_size = min(2000, len(test_df) // 5)
    train_df = test_df.sample(n=train_size, random_state=42)
    print(f"Using {len(train_df)} samples for training and {len(test_df)} for testing")
    
    # Check columns
    if "Target" not in test_df.columns or "proteina" not in test_df.columns or "Y" not in test_df.columns:
        # Find sequence columns
        seq_cols = []
        for col in test_df.columns:
            if test_df[col].dtype == object and len(test_df[col].iloc[0]) > 20:
                seq_cols.append(col)
        
        # Find numeric column
        y_col = None
        for col in test_df.columns:
            if col not in seq_cols and test_df[col].dtype in [np.float64, np.int64]:
                y_col = col
                break
        
        if len(seq_cols) >= 2 and y_col:
            print(f"Using {seq_cols[0]} as Target, {seq_cols[1]} as proteina, and {y_col} as Y")
            test_df = test_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina", y_col: "Y"})
            train_df = train_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina", y_col: "Y"})
    
    # Load ESM2 model from Hugging Face
    print("Loading ESM2 model from Hugging Face")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    model.eval()
    
    # Generate embeddings for training data
    print("Generating embeddings for training data")
    train_emb1 = get_esm2_embeddings(model, tokenizer, train_df["Target"].tolist(), device, args.batch_size)
    train_emb2 = get_esm2_embeddings(model, tokenizer, train_df["proteina"].tolist(), device, args.batch_size)
    train_embeddings = torch.cat([train_emb1, train_emb2], dim=1)
    
    # Train regression model
    print("Training regression model")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_embeddings.numpy())
    
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
    
    # Generate embeddings for test data
    print("Generating embeddings for test data")
    test_emb1 = get_esm2_embeddings(model, tokenizer, test_df["Target"].tolist(), device, args.batch_size)
    test_emb2 = get_esm2_embeddings(model, tokenizer, test_df["proteina"].tolist(), device, args.batch_size)
    test_embeddings = torch.cat([test_emb1, test_emb2], dim=1)
    
    # Make predictions
    X_test_scaled = scaler.transform(test_embeddings.numpy())
    predictions = regressor.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(test_df["Y"].values, predictions)
    
    print("\nESM2 Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Create regression plot
    plt.figure(figsize=(10, 8))
    sns.regplot(x=test_df["Y"].values, y=predictions)
    plt.title('ESM2: Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add metrics to plot
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.savefig(os.path.join(args.output_dir, "esm2_regression_plot.png"))
    
    # Save results
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("ESM2 Model Results\n")
        f.write("=================\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    test_df["ESM2_predictions"] = predictions
    test_df[["Target", "proteina", "Y", "ESM2_predictions"]].to_csv(
        os.path.join(args.output_dir, "esm2_results.csv"), index=False
    )
    
    print(f"Results saved to {args.output_dir}")
    
    # Compare with BALM results if available
    balm_results_path = os.path.join("direct_comparison_results", "balm_results.csv")
    if os.path.exists(balm_results_path):
        print("\nComparing with BALM-PPI results...")
        balm_df = pd.read_csv(balm_results_path)
        
        comparison_df = test_df[["Target", "proteina", "Y", "ESM2_predictions"]].copy()
        comparison_df["BALM_predictions"] = balm_df["BALM_predictions"]
        
        comparison_df.to_csv(os.path.join(args.output_dir, "comparison_results.csv"), index=False)
        
        # Create comparison bar chart
        balm_metrics = {
            "RMSE": 1.1355,
            "Pearson": 0.8549,
            "Spearman": 0.8532,
            "R²": 0.7256,
            "CI": 0.4883
        }
        
        metric_names = list(metrics.keys())
        esm2_values = [metrics[m] for m in metric_names]
        balm_values = [balm_metrics[m] for m in metric_names]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, balm_values, width, label='BALM-PPI')
        plt.bar(x + width/2, esm2_values, width, label='ESM2')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('BALM-PPI vs ESM2 Performance Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        
        for i, v in enumerate(balm_values):
            plt.text(i - width/2, v + 0.02, f"{v:.4f}", ha='center')
        
        for i, v in enumerate(esm2_values):
            plt.text(i + width/2, v + 0.02, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "comparison_chart.png"))
        
        # Save comparison summary
        with open(os.path.join(args.output_dir, "comparison_summary.txt"), "w") as f:
            f.write("BALM-PPI vs ESM2 Comparison\n")
            f.write("==========================\n\n")
            
            f.write("BALM-PPI Metrics:\n")
            for k, v in balm_metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nESM2 Metrics:\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nDifference (ESM2 - BALM-PPI):\n")
            for k in metric_names:
                diff = metrics[k] - balm_metrics[k]
                better = "better" if ((k == "RMSE" and diff < 0) or (k != "RMSE" and diff > 0)) else "worse"
                f.write(f"  {k}: {diff:.4f} ({better})\n")

if __name__ == "__main__":
    main()