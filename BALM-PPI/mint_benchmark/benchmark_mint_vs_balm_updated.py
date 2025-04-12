import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Add mint directory to the python path
mint_dir = "D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts"
if mint_dir not in sys.path:
    sys.path.append(mint_dir)

# Now import from mint (notice we're using absolute imports)
sys.path.append("D:/BALM_Fineclone/BALM-PPI/Benchmark")
import mint

def load_config(path):
    """Load JSON config file"""
    import json
    with open(path) as f:
        cfg = argparse.Namespace()
        cfg.__dict__.update(json.load(f))
    return cfg

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, df, col1, col2):
        super().__init__()
        # If df is a path, load it as a DataFrame
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df
            
        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()

    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        return self.seqs1[index], self.seqs2[index]

class CollateFn:
    def __init__(self, truncation_seq_length=None):
        self.alphabet = mint.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        from torch import tensor
        heavy_chain, light_chain = zip(*batches)
        chains = [self.convert(c) for c in [heavy_chain, light_chain]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        return chains, chain_ids

    def convert(self, seq_str_list):
        import random
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

class MINTWrapper(torch.nn.Module):
    def __init__(
        self,
        cfg,
        checkpoint_path,
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device="cuda:0",
    ):
        super().__init__()
        import re
        from collections import OrderedDict
        from mint.model.esm2 import ESM2
        
        self.cfg = cfg
        self.sep_chains = sep_chains
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=use_multimer,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if use_multimer:
            # remove 'model.' in keys
            new_checkpoint = OrderedDict(
                (key.replace("model.", ""), value)
                for key, value in checkpoint["state_dict"].items()
            )
            self.model.load_state_dict(new_checkpoint)
        else:
            # Upgrade state dict
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            new_checkpoint = {pattern.sub("", name): param for name, param in checkpoint["model"].items()}
            self.model.load_state_dict(new_checkpoint)
            
        self.model.to(device)
        
        # Freeze parameters based on freeze_percent
        import math
        total_layers = cfg.encoder_layers
        for name, param in self.model.named_parameters():
            if "embed_tokens.weight" in name or "_norm_after" in name or "lm_head" in name:
                param.requires_grad = False
            else:
                layer_num = name.split(".")[1]
                if int(layer_num) <= math.floor(total_layers * freeze_percent):
                    param.requires_grad = False

    def get_one_chain(self, chain_out, mask_expanded, mask):
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
        mean_chain_out = sum_masked / mask_counts
        return mean_chain_out

    def forward(self, chains, chain_ids):
        mask = (
            (~chains.eq(self.model.cls_idx))
            & (~chains.eq(self.model.eos_idx))
            & (~chains.eq(self.model.padding_idx))
        )
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]
        if self.sep_chains:
            mask_chain_0 = (chain_ids.eq(0) & mask).unsqueeze(-1).expand_as(chain_out)
            mask_chain_1 = (chain_ids.eq(1) & mask).unsqueeze(-1).expand_as(chain_out)
            mean_chain_out_0 = self.get_one_chain(
                chain_out, mask_chain_0, (chain_ids.eq(0) & mask)
            )
            mean_chain_out_1 = self.get_one_chain(
                chain_out, mask_chain_1, (chain_ids.eq(1) & mask)
            )
            return torch.cat((mean_chain_out_0, mean_chain_out_1), -1)
        else:
            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
            mean_chain_out = sum_masked / mask_counts
            return mean_chain_out

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MINT against BALM-PPI")
    parser.add_argument("--data_path", type=str, required=True, help="Path to BALM-PPI data CSV")
    parser.add_argument("--balm_config", type=str, default="D:/BALM_Fineclone/BALM-PPI/default_configs/balm_peft.yaml", 
                        help="Path to BALM config file")
    parser.add_argument("--mint_checkpoint", type=str, default="D:/BALM_Fineclone/BALM-PPI/Benchmark/models/mint.ckpt", 
                        help="Path to MINT checkpoint")
    parser.add_argument("--mint_config", type=str, default="D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts/data/esm2_t33_650M_UR50D.json", 
                        help="Path to MINT config file")
    parser.add_argument("--output_dir", type=str, default="./mint_results", 
                        help="Directory to save results")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum sequence length for MINT input")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for embedding generation")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--split_ratio", type=float, default=0.2,
                        help="Train/test split ratio (from BALM)")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed for data splitting")
    parser.add_argument("--data_dir", type=str, default="mint_benchmark_results/data",
                        help="Directory with pre-processed data splits")
    return parser.parse_args()

def load_balm_config(config_path):
    """Load BALM config YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(data_path, split_ratio, random_seed):
    """Preprocess data and split according to BALM's approach"""
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} samples")
    
    # Calculate data bounds (same as in BALM)
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Create train/test split using the same approach as BALM
    from sklearn.model_selection import train_test_split
    
    # Split into train/test
    train_val_df, test_df = train_test_split(
        df, 
        train_size=split_ratio,
        random_state=random_seed
    )
    
    # Split train into train/val
    train_df, val_df = train_test_split(
        train_val_df,
        train_size=0.8,
        random_state=random_seed
    )
    
    print(f"Split dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    return train_df, val_df, test_df, (data_min, data_max)

def prepare_mint_dataset(df, max_seq_len, batch_size):
    """Prepare dataset in MINT format"""
    # Ensure column names are as expected by MINT
    df_mint = df.copy()
    if 'Target' not in df_mint.columns and 'proteina' not in df_mint.columns:
        # Rename columns if needed
        protein_col = df_mint.columns[0] if df_mint.columns[0] != 'Y' else df_mint.columns[1]
        proteina_col = df_mint.columns[1] if df_mint.columns[1] != 'Y' else df_mint.columns[2]
        df_mint = df_mint.rename(columns={protein_col: 'Target', proteina_col: 'proteina'})
    
    # Create MINT dataset and loader
    dataset = CSVDataset(df_mint, 'Target', 'proteina')
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=CollateFn(max_seq_len), 
        shuffle=False
    )
    
    return dataloader

def get_mint_embeddings(model, dataloader, device, sep_chains=True):
    """Generate embeddings using MINT"""
    all_embeddings = []
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (chains, chain_ids) in enumerate(tqdm(dataloader, desc="Generating embeddings")):
            # Move data to device
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            
            # Get embeddings
            embeddings = model(chains, chain_ids)
            all_embeddings.append(embeddings.detach().cpu())
            
    # Concatenate all embeddings
    return torch.cat(all_embeddings, dim=0)

def train_regression_model(X_train, y_train, X_val, y_val):
    """Train a simple MLP regression model on the embeddings"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define and train model
    model = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=1234
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    return model, scaler, y_pred

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    # Calculate metrics
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
    
    ci = get_ci(np.array(y_true), np.array(y_pred))
    
    metrics = {
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "R²": r2,
        "CI": ci
    }
    
    return metrics

def plot_results(y_true, y_pred, metrics, output_path):
    """Plot regression results"""
    plt.figure(figsize=(10, 8))
    sns.regplot(x=y_true, y=y_pred)
    plt.title('MINT: Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add metrics text to plot
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def save_results(train_df, val_df, metrics, output_dir):
    """Save all results to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save metrics to TXT
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("MINT Model Results\n")
        f.write("=================\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Save data splits for reproducing
    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    
    print(f"Results saved to {output_dir}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configurations
    balm_config = load_balm_config(args.balm_config)
    mint_config = load_config(args.mint_config)
    
    # Check if preprocessed data exists
    if os.path.exists(os.path.join(args.data_dir, "train.csv")):
        print(f"Loading preprocessed data from {args.data_dir}")
        train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        
        # Calculate data bounds
        data_min = test_df['Y'].min()
        data_max = test_df['Y'].max()
        print(f"Data range: {data_min:.4f} to {data_max:.4f}")
        print(f"Using preprocessed data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    else:
        # Preprocess data
        train_df, val_df, test_df, (data_min, data_max) = preprocess_data(
            args.data_path, 
            args.split_ratio, 
            args.random_seed
        )
    
    # Check if CUDA is available
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("WARNING: CUDA is not available, using CPU instead")
        args.device = "cpu"
    
    # Initialize MINT model
    print(f"Initializing MINT model from {args.mint_checkpoint}")
    wrapper = MINTWrapper(
        mint_config, 
        args.mint_checkpoint, 
        freeze_percent=1.0,  # Use the full pre-trained model
        sep_chains=True,  # Separate chain embeddings
        device=args.device
    )
    
    # Prepare datasets
    train_loader = prepare_mint_dataset(train_df, args.max_seq_len, args.batch_size)
    val_loader = prepare_mint_dataset(val_df, args.max_seq_len, args.batch_size)
    test_loader = prepare_mint_dataset(test_df, args.max_seq_len, args.batch_size)
    
    # Generate embeddings
    print("Generating embeddings for training set")
    train_embeddings = get_mint_embeddings(wrapper, train_loader, args.device)
    print("Generating embeddings for validation set")
    val_embeddings = get_mint_embeddings(wrapper, val_loader, args.device)
    print("Generating embeddings for test set")
    test_embeddings = get_mint_embeddings(wrapper, test_loader, args.device)
    
    # Save embeddings to disk for potential reuse
    embedding_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    
    torch.save(train_embeddings, os.path.join(embedding_dir, "train_embeddings.pt"))
    torch.save(val_embeddings, os.path.join(embedding_dir, "val_embeddings.pt"))
    torch.save(test_embeddings, os.path.join(embedding_dir, "test_embeddings.pt"))
    print(f"Saved embeddings to {embedding_dir}")
    
    # Train regression model
    print("Training regression model on embeddings")
    model, scaler, val_predictions = train_regression_model(
        train_embeddings.numpy(), 
        train_df['Y'].values,
        val_embeddings.numpy(),
        val_df['Y'].values
    )
    
    # Evaluate on test set
    X_test_scaled = scaler.transform(test_embeddings.numpy())
    test_predictions = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = evaluate_model(test_df['Y'].values, test_predictions)
    print("\nMINT Model Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Plot results
    plot_results(
        test_df['Y'].values, 
        test_predictions, 
        metrics, 
        os.path.join(args.output_dir, 'regression_plot.png')
    )
    
    # Save all results
    save_results(train_df, test_df, metrics, args.output_dir)

if __name__ == "__main__":
    main()