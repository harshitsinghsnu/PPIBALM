import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
import esm
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

from balm import common_utils
from balm.configs import Configs
from balm.models import BALM
from balm.models.utils import load_trained_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Gaussian Process models on protein embeddings.")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config YAML file for BALM model")
    parser.add_argument("--balm_checkpoint_path", type=str, required=True, help="Path to BALM checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--embedding_type", type=str, required=True, 
                        choices=["ESM2-protein", "ESM2-proteina", "BALM-protein", 
                                "BALM-proteina", "BALM-concat", "BALM-sum", "BALM-subtract", "BALM-cosine"])
    parser.add_argument("--test_size", type=float, default=0.8, help="Fraction of data to use for testing")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data splitting")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for GP model training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for GP model training")
    parser.add_argument("--no_wandb", action="store_true", help="Disable logging metrics to wandb")
    parser.add_argument("--wandb_project", type=str, default="protein-protein-gp", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--output_dir", type=str, default="outputs/gp", help="Directory to save outputs")
    return parser.parse_args()

def get_esm2_embeddings(sequences: list, batch_size: int = 32) -> np.ndarray:
    """Get ESM-2 embeddings for protein sequences"""
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Getting ESM2 embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
            batch_embeddings = results["representations"][33].mean(axis=1)
            embeddings.append(batch_embeddings.cpu().numpy())
            
    return np.concatenate(embeddings)

def get_balm_embeddings(proteins_a: list, proteins_b: list, 
                       model: BALM, batch_size: int = 32) -> tuple:
    """Get BALM embeddings for protein pairs"""
    protein_embeddings = []
    proteina_embeddings = []
    cosine_similarities = []
    
    for i in tqdm(range(0, len(proteins_a), batch_size), desc="Getting BALM embeddings"):
        batch_a = proteins_a[i:i + batch_size]
        batch_b = proteins_b[i:i + batch_size]
        
        inputs = {
            "protein_sequences": batch_a,
            "proteina_sequences": batch_b,
        }
        
        with torch.no_grad():
            outputs = model(inputs)
            protein_embeddings.append(outputs["protein_embedding"].cpu().numpy())
            proteina_embeddings.append(outputs["proteina_embedding"].cpu().numpy())
            cosine_similarities.append(outputs["cosine_similarity"].cpu().numpy())
    
    return (np.concatenate(protein_embeddings),
            np.concatenate(proteina_embeddings),
            np.concatenate(cosine_similarities))

class ProteinGPModel(gpytorch.models.ExactGP):
    """Gaussian Process model for protein embeddings"""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(X_train, y_train, X_test, y_test, sequences_test, 
                  learning_rate: float = 0.1, epochs: int = 50,
                  use_wandb: bool = True, output_dir: str = "outputs/gp"):
    """Train and evaluate GP model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ProteinGPModel(X_train, y_train, likelihood)
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for i in tqdm(range(epochs), desc="Training GP model"):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        if use_wandb:
            wandb.log({"epoch": i+1, "train/loss": loss.item()})
    
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = likelihood(model(X_test)).mean.numpy()
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
        "spearman": spearmanr(y_test, preds)[0],
        "pearson": pearsonr(y_test, preds)[0]
    }
    
    if use_wandb:
        wandb.log({f"test/{k}": v for k, v in metrics.items()})
    
    # Save predictions to file
    results_df = pd.DataFrame({
        "protein_b": sequences_test,
        "label": y_test,
        "prediction": preds
    })
    
    results_path = os.path.join(output_dir, "test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")
    
    # Create regression plot
    plt.figure(figsize=(10, 8))
    ax = sns.regplot(x=y_test, y=preds)
    ax.set_title(f"GP Model Predictions")
    ax.set_xlabel(r"Experimental $pK_d$")
    ax.set_ylabel(r"Predicted $pK_d$")
    
    # Save the plot
    plot_path = os.path.join(output_dir, "regression_plot.png")
    plt.savefig(plot_path)
    print(f"Saved regression plot to {plot_path}")
    
    # Log the plot to W&B
    if use_wandb:
        wandb.log({"regression_plot": wandb.Image(plot_path)})
    
    return metrics

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Set device
        if torch.cuda.is_available():
            DEVICE = "cuda" 
        else:
            DEVICE = "cpu"
        print(f"Using device: {DEVICE}")
        
        # Set random seed
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize wandb
        args.use_wandb = not args.no_wandb  # Default to using wandb, unless --no_wandb is specified
        if args.use_wandb:
            try:
                wandb.login()
                
                run_name = f"gp_{args.embedding_type}_test{args.test_size}_lr{args.lr}"
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=vars(args))
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Continuing without wandb logging...")
                args.use_wandb = False
                
        # Load data
        try:
            df = pd.read_csv(args.data_path)
            
            # Check if required columns exist
            required_columns = ["Target", "proteina", "Y"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"Error: Column '{col}' not found in data file. Available columns: {df.columns.tolist()}")
                    return
                    
            print(f"Loaded data with {len(df)} samples")
        except Exception as e:
            print(f"Error loading data from {args.data_path}: {e}")
            return
        
        # Extract protein sequences and labels
        proteins_a = df["Target"].tolist()
        proteins_b = df["proteina"].tolist()
        y = df["Y"].to_numpy()
        
        # Get embeddings based on embedding_type
        if args.embedding_type.startswith("ESM2"):
            try:
                print(f"Getting ESM2 embeddings for {'protein' if 'protein' in args.embedding_type else 'proteina'}")
                X = get_esm2_embeddings(proteins_a if "protein" in args.embedding_type else proteins_b)
            except Exception as e:
                print(f"Error getting ESM2 embeddings: {e}")
                return
        else:
            try:
                print(f"Loading BALM model from config {args.config_filepath} and checkpoint {args.balm_checkpoint_path}")
                # Load BALM model
                configs = Configs(**common_utils.load_yaml(args.config_filepath))
                model = BALM(configs.model_configs)
                
                # Use checkpoint path from args
                if hasattr(configs.model_configs, 'checkpoint_path'):
                    configs.model_configs.checkpoint_path = args.balm_checkpoint_path
                
                # Load trained model
                model = load_trained_model(model, configs.model_configs, is_training=False)
                model = model.to(DEVICE)
                
                print(f"Getting BALM embeddings for {args.embedding_type}")
                protein_emb, proteina_emb, cosine_sim = get_balm_embeddings(proteins_a, proteins_b, model)
                
                if args.embedding_type == "BALM-protein":
                    X = protein_emb
                elif args.embedding_type == "BALM-proteina":
                    X = proteina_emb
                elif args.embedding_type == "BALM-concat":
                    X = np.concatenate((protein_emb, proteina_emb), axis=1)
                elif args.embedding_type == "BALM-sum":
                    X = protein_emb + proteina_emb
                elif args.embedding_type == "BALM-subtract":
                    X = protein_emb - proteina_emb
                else:  # BALM-cosine
                    X = cosine_sim.reshape(-1, 1)  # Reshape to 2D array if needed
                
            except Exception as e:
                print(f"Error getting BALM embeddings: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Split data into train and test sets
        print(f"Splitting data with test_size={args.test_size}, random_seed={args.random_seed}")
        X_train, X_test, y_train, y_test, _, sequences_test = train_test_split(
            X, y, proteins_b, test_size=args.test_size, random_state=args.random_seed
        )
        
        # Clear memory before GP training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train GP model
        try:
            print(f"Training GP model with lr={args.lr}, epochs={args.epochs}")
            metrics = train_gp_model(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                sequences_test,
                args.lr,
                args.epochs,
                args.use_wandb,
                args.output_dir
            )
            
            print(f"{args.embedding_type} GP Model Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save metrics to file
            with open(os.path.join(args.output_dir, "metrics.txt"), 'w') as f:
                f.write(f"Embedding Type: {args.embedding_type}\n")
                f.write(f"Test Size: {args.test_size}\n")
                f.write(f"Random Seed: {args.random_seed}\n")
                f.write(f"Learning Rate: {args.lr}\n")
                f.write(f"Epochs: {args.epochs}\n\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            if args.use_wandb:
                wandb.finish()
                
        except Exception as e:
            print(f"Error training GP model: {e}")
            import traceback
            traceback.print_exc()
            return
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()