import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Simplified ESM2 Benchmark")
    parser.add_argument("--data_path", type=str, required=True, help="Path to BALM-PPI data CSV")
    parser.add_argument("--output_dir", type=str, default="./esm2_benchmark_results", 
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for embedding generation")
    parser.add_argument("--split_ratio", type=float, default=0.2,
                        help="Train/test split ratio")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed for data splitting")
    parser.add_argument("--data_dir", type=str, default="mint_benchmark_results/data",
                        help="Directory with pre-processed data splits")
    return parser.parse_args()

def process_data(args):
    """Process data and load splits"""
    if os.path.exists(os.path.join(args.data_dir, "train.csv")):
        print(f"Loading preprocessed data from {args.data_dir}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        
        print(f"Using preprocessed data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    else:
        # Load from original file
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(args.data_path)
        print(f"Loaded data with {len(df)} samples")
        
        # Split into train/test
        train_val_df, test_df = train_test_split(
            df, train_size=args.split_ratio, random_state=args.random_seed
        )
        
        # Split train into train/val
        train_df, val_df = train_test_split(
            train_val_df, train_size=0.8, random_state=args.random_seed
        )
        
        print(f"Split dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    # Make sure columns are correctly named
    cols = train_df.columns.tolist()
    if "Target" not in train_df.columns or "proteina" not in train_df.columns:
        # Find likely protein sequence columns
        seq_cols = []
        for col in cols:
            if train_df[col].dtype == object and len(train_df[col].iloc[0]) > 20:
                seq_cols.append(col)
        
        if len(seq_cols) >= 2:
            # Rename columns
            train_df = train_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina"})
            val_df = val_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina"})
            test_df = test_df.rename(columns={seq_cols[0]: "Target", seq_cols[1]: "proteina"})
    
    return train_df, val_df, test_df

def get_esm2_embeddings(model, tokenizer, sequences, device, batch_size=2):
    """Generate embeddings using ESM2 model"""
    embeddings = []
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
        batch_seqs = sequences[i:i+batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get embeddings from the last layer
            last_hidden_states = outputs.hidden_states[-1]
            # Take average over sequence length (excluding padding)
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings_batch = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(embeddings_batch.cpu())
    
    return torch.cat(embeddings, dim=0)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process data
    train_df, val_df, test_df = process_data(args)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ESM2 model from Hugging Face
    print("Loading ESM2 model from Hugging Face")
    from transformers import AutoTokenizer, AutoModel
    
    # Load the ESM2 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    model.eval()
    
    # Generate embeddings
    print("Generating embeddings for proteins")
    train_emb1 = get_esm2_embeddings(model, tokenizer, train_df["Target"].tolist(), device, args.batch_size)
    train_emb2 = get_esm2_embeddings(model, tokenizer, train_df["proteina"].tolist(), device, args.batch_size)
    
    val_emb1 = get_esm2_embeddings(model, tokenizer, val_df["Target"].tolist(), device, args.batch_size)
    val_emb2 = get_esm2_embeddings(model, tokenizer, val_df["proteina"].tolist(), device, args.batch_size)
    
    test_emb1 = get_esm2_embeddings(model, tokenizer, test_df["Target"].tolist(), device, args.batch_size)
    test_emb2 = get_esm2_embeddings(model, tokenizer, test_df["proteina"].tolist(), device, args.batch_size)
    
    # Concatenate embeddings for protein pairs
    train_embeddings = torch.cat([train_emb1, train_emb2], dim=1)
    val_embeddings = torch.cat([val_emb1, val_emb2], dim=1)
    test_embeddings = torch.cat([test_emb1, test_emb2], dim=1)
    
    # Save embeddings
    embedding_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    
    torch.save(train_embeddings, os.path.join(embedding_dir, "train_embeddings.pt"))
    torch.save(val_embeddings, os.path.join(embedding_dir, "val_embeddings.pt"))
    torch.save(test_embeddings, os.path.join(embedding_dir, "test_embeddings.pt"))
    
    # Train regression model
    print("Training regression model")
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_embeddings.numpy())
    X_val_scaled = scaler.transform(val_embeddings.numpy())
    X_test_scaled = scaler.transform(test_embeddings.numpy())
    
    # Train model
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=200,
        early_stopping=True,
        random_state=args.random_seed
    )
    
    model.fit(X_train_scaled, train_df["Y"].values)
    
    # Evaluate on test set
    test_predictions = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_df["Y"].values, test_predictions))
    pearson = stats.pearsonr(test_df["Y"].values, test_predictions)[0]
    spearman = stats.spearmanr(test_df["Y"].values, test_predictions)[0]
    r2 = r2_score(test_df["Y"].values, test_predictions)
    
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
    
    ci = get_ci(test_df["Y"].values, test_predictions)
    
    # Collect metrics
    metrics = {
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "R²": r2,
        "CI": ci
    }
    
    # Print metrics
    print("\nESM2 Model Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("ESM2 Model Results\n")
        f.write("=================\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    sns.regplot(x=test_df["Y"].values, y=test_predictions)
    plt.title('ESM2: Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add metrics text to plot
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Save figure
    plt.savefig(os.path.join(args.output_dir, "regression_plot.png"))
    plt.close()
    
    # Save predictions
    test_df["predictions"] = test_predictions
    test_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()