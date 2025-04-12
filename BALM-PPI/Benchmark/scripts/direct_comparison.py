import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import random
import traceback
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained BALM-PPI directly with MINT")
    parser.add_argument("--data_path", type=str, default="D:/BALM_Fineclone/BALM-PPI/scripts/notebooks/Data.csv", 
                        help="Path to full dataset CSV")
    parser.add_argument("--output_dir", type=str, default="D:/BALM_Fineclone/BALM-PPI/direct_model_comparison",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--train_ratio", type=float, default=0.2, help="Ratio of data for training (same as BALM-PPI used)")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed for data split")
    return parser.parse_args()

def calculate_ci(y_true, y_pred):
    """Calculate concordance index."""
    total_pairs = 0
    concordant_pairs = 0
    
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            if y_true[i] != y_true[j]:  # Only count if there's a difference
                total_pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant_pairs += 1
    
    return concordant_pairs / total_pairs if total_pairs > 0 else 0.5

def evaluate_balm(model, data_loader, data_min, data_max, device):
    """Evaluate BALM-PPI model on test data."""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating BALM-PPI"):
            # Scale targets to cosine similarity range
            cosine_targets = [2 * (y - data_min) / (data_max - data_min) - 1 for y in batch["Y"]]
            
            # Prepare input
            inputs = {
                "protein_sequences": batch["Target"],
                "proteina_sequences": batch["proteina"],
                "labels": torch.tensor(cosine_targets, dtype=torch.float32).to(device)
            }
            
            outputs = model(inputs)
            
            # Get predictions using cosine_similarity_to_pkd
            batch_preds = model.cosine_similarity_to_pkd(
                outputs["cosine_similarity"],
                pkd_upper_bound=data_max,
                pkd_lower_bound=data_min
            ).cpu().numpy()
            
            predictions.extend(batch_preds)
            labels.extend(batch["Y"])
    
    # Calculate metrics
    metrics = {}
    metrics['rmse'] = float(np.sqrt(mean_squared_error(labels, predictions)))
    metrics['pearson'] = float(pearsonr(labels, predictions)[0])
    metrics['spearman'] = float(spearmanr(labels, predictions)[0])
    metrics['ci'] = float(calculate_ci(labels, predictions))
    
    return metrics, np.array(predictions), np.array(labels)

def train_mint(model, train_loader, device):
    """Train MINT model on the training data."""
    from mint.helpers.extract import CollateFn
    import torch.optim as optim
    import torch.nn as nn
    
    model.train()
    collate_fn = CollateFn(truncation_seq_length=2048)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    num_epochs = 5  # You can adjust this
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Training MINT - Epoch {epoch+1}/{num_epochs}"):
            try:
                # Prepare data in MINT format
                batch_data = list(zip(batch["Target"], batch["proteina"]))
                chains, chain_ids = collate_fn(batch_data)
                
                # Move to device
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                
                # Create labels tensor
                labels = torch.tensor(batch["Y"], dtype=torch.float32).to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                # MINT may provide embeddings - we need a prediction head
                embeddings = model(chains, chain_ids)
                
                # Use mean pooling as a simple prediction head
                predictions = embeddings.mean(dim=1)
                
                # Calculate loss
                loss = criterion(predictions, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        # Print epoch results
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("MINT training completed")
    return model

def evaluate_mint(model, test_loader, device):
    """Evaluate MINT model on test data."""
    from mint.helpers.extract import CollateFn
    
    model.eval()
    predictions = []
    labels = []
    
    collate_fn = CollateFn(truncation_seq_length=2048)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating MINT"):
            try:
                # Prepare data in MINT format
                batch_data = list(zip(batch["Target"], batch["proteina"]))
                chains, chain_ids = collate_fn(batch_data)
                
                # Move to device
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                
                # Get embeddings
                embeddings = model(chains, chain_ids)
                
                # Use mean pooling for prediction
                batch_preds = embeddings.mean(dim=1).cpu().numpy()
                
                predictions.extend(batch_preds)
                labels.extend(batch["Y"])
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    if len(predictions) > 0:
        # Calculate metrics
        metrics = {}
        metrics['rmse'] = float(np.sqrt(mean_squared_error(labels, predictions)))
        metrics['pearson'] = float(pearsonr(labels, predictions)[0])
        metrics['spearman'] = float(spearmanr(labels, predictions)[0])
        metrics['ci'] = float(calculate_ci(labels, predictions))
        
        return metrics, np.array(predictions), np.array(labels)
    else:
        return {}, np.array([]), np.array([])

def create_comparison_plots(balm_results, mint_results, output_dir):
    """Create comparison visualizations for both models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Side-by-side regression plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # BALM-PPI plot
    sns.regplot(x=balm_results['true_values'], y=balm_results['predictions'], ax=axes[0])
    axes[0].set_title("BALM-PPI Direct Prediction")
    axes[0].set_xlabel(r"Experimental $pK_d$")
    axes[0].set_ylabel(r"Predicted $pK_d$")
    
    # Add metrics to BALM plot
    balm_metrics_text = f"RMSE: {balm_results['metrics']['rmse']:.4f}\n" \
                      f"Pearson: {balm_results['metrics']['pearson']:.4f}\n" \
                      f"Spearman: {balm_results['metrics']['spearman']:.4f}\n" \
                      f"CI: {balm_results['metrics']['ci']:.4f}"
    
    axes[0].text(0.05, 0.95, balm_metrics_text, transform=axes[0].transAxes, 
              verticalalignment='top', horizontalalignment='left',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # MINT plot
    if len(mint_results['true_values']) > 0:
        sns.regplot(x=mint_results['true_values'], y=mint_results['predictions'], ax=axes[1])
        axes[1].set_title("MINT Prediction")
        axes[1].set_xlabel(r"Experimental $pK_d$")
        axes[1].set_ylabel(r"Predicted $pK_d$")
        
        # Add metrics to MINT plot
        mint_metrics_text = f"RMSE: {mint_results['metrics']['rmse']:.4f}\n" \
                          f"Pearson: {mint_results['metrics']['pearson']:.4f}\n" \
                          f"Spearman: {mint_results['metrics']['spearman']:.4f}\n" \
                          f"CI: {mint_results['metrics']['ci']:.4f}"
        
        axes[1].text(0.05, 0.95, mint_metrics_text, transform=axes[1].transAxes, 
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1].text(0.5, 0.5, "No MINT predictions available", 
                     ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Only create additional plots if MINT has results
    if len(mint_results['true_values']) > 0:
        # Metrics comparison bar chart
        metrics = ['rmse', 'pearson', 'spearman', 'ci']
        balm_values = [balm_results['metrics'][m] for m in metrics]
        mint_values = [mint_results['metrics'][m] for m in metrics]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, balm_values, width, label='BALM-PPI')
        plt.bar(x + width/2, mint_values, width, label='MINT')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Model Performance Comparison')
        plt.xticks(x, [m.capitalize() for m in metrics])
        
        # Add value labels
        for i, v in enumerate(balm_values):
            plt.text(i - width/2, v + 0.02, f"{v:.4f}", ha='center')
        
        for i, v in enumerate(mint_values):
            plt.text(i + width/2, v + 0.02, f"{v:.4f}", ha='center')
        
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary report
        with open(os.path.join(output_dir, "comparison_summary.txt"), 'w') as f:
            f.write("Model Comparison Summary\n")
            f.write("======================\n\n")
            
            f.write("BALM-PPI Metrics:\n")
            for metric, value in balm_results['metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            
            f.write("\nMINT Metrics:\n")
            for metric, value in mint_results['metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            
            f.write("\nPerformance Difference (BALM - MINT):\n")
            for metric in metrics:
                diff = balm_results['metrics'][metric] - mint_results['metrics'][metric]
                f.write(f"  {metric.capitalize()}: {diff:.4f}\n")
            
            # Calculate percentage improvement for relevant metrics
            f.write("\nPercentage Improvement:\n")
            
            # For RMSE, lower is better
            rmse_improvement = (mint_results['metrics']['rmse'] - balm_results['metrics']['rmse']) / mint_results['metrics']['rmse'] * 100
            f.write(f"  RMSE: {rmse_improvement:.2f}% {'better' if rmse_improvement > 0 else 'worse'}\n")
            
            # For correlation metrics, higher is better
            for metric in ['pearson', 'spearman', 'ci']:
                improvement = (balm_results['metrics'][metric] - mint_results['metrics'][metric]) / mint_results['metrics'][metric] * 100
                f.write(f"  {metric.capitalize()}: {improvement:.2f}% {'better' if improvement > 0 else 'worse'}\n")
    else:
        with open(os.path.join(output_dir, "balm_results.txt"), 'w') as f:
            f.write("BALM-PPI Model Results\n")
            f.write("=====================\n\n")
            
            f.write("Metrics:\n")
            for metric, value in balm_results['metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")

def main():
    args = parse_args()
    
    # Set the path to BALM-PPI explicitly
    balm_path = "D:/BALM_Fineclone/BALM-PPI"
    if balm_path not in sys.path:
        sys.path.insert(0, balm_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Calculate data bounds
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Split data using the same ratio and seed as in BALM-PPI training
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        df, train_size=args.train_ratio, random_state=args.random_seed
    )
    
    print(f"Split data: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Save splits for reference
    train_path = os.path.join(args.output_dir, "train_split.csv")
    test_path = os.path.join(args.output_dir, "test_split.csv")
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Create batch loaders
    def batch_loader(df, batch_size):
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            yield {
                "Target": batch_df["Target"].tolist(),
                "proteina": batch_df["proteina"].tolist(),
                "Y": batch_df["Y"].tolist()
            }
    
    train_loader = list(batch_loader(train_data, args.batch_size))
    test_loader = list(batch_loader(test_data, args.batch_size))
    
    # Results dictionary
    results = {}
    
    # 1. Load and evaluate BALM-PPI
    print("\n==== Loading Pre-trained BALM-PPI Model ====")
    
    try:
        from balm import common_utils
        from balm.configs import Configs
        from balm.models import BALM
        
        config_path = "D:/BALM_Fineclone/BALM-PPI/default_configs/balm_peft.yaml"
        checkpoint_path = "D:/BALM_Fineclone/BALM-PPI/outputs/latest_checkpoint.pth"
        
        # Load configuration and initialize model
        print(f"Loading configuration from {config_path}")
        configs = Configs(**common_utils.load_yaml(config_path))
        
        balm_model = BALM(configs.model_configs)
        balm_model = balm_model.to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            balm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            balm_model.load_state_dict(checkpoint)
        
        balm_model.eval()
        print("BALM-PPI Model loaded successfully")
        
        # Evaluate BALM-PPI on test data
        print("\n==== Evaluating BALM-PPI on Test Data ====")
        balm_metrics, balm_predictions, balm_labels = evaluate_balm(
            balm_model, test_loader, data_min, data_max, device
        )
        
        print("\nBALM-PPI Direct Prediction Results:")
        for metric, value in balm_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        # Save results
        results['balm'] = {
            'metrics': balm_metrics,
            'predictions': balm_predictions,
            'true_values': balm_labels
        }
        
    except Exception as e:
        print(f"Error processing BALM-PPI: {e}")
        traceback.print_exc()
    
    # 2. Load, train and evaluate MINT
    print("\n==== Loading MINT Model ====")
    
    # Add MINT to path
    mint_dir = "D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts/mint"
    if mint_dir not in sys.path:
        sys.path.append(mint_dir)
    
    try:
        import mint
        from mint.helpers.extract import MINTWrapper, load_config
        
        # Add random to mint.helpers.extract
        mint.helpers.extract.random = random
        
        config_path = "D:/BALM_Fineclone/BALM-PPI/Benchmark/scripts/mint/data/esm2_t33_650M_UR50D.json"
        checkpoint_path = "D:/BALM_Fineclone/BALM-PPI/Benchmark/models/mint.ckpt"
        
        # Load configuration
        cfg = load_config(config_path)
        
        # Initialize model
        mint_model = MINTWrapper(
            cfg, 
            checkpoint_path, 
            freeze_percent=1.0, 
            use_multimer=True, 
            sep_chains=True,
            device=device
        )
        print("MINT Model loaded successfully")
        
        # Train MINT on the same training data
        print("\n==== Training MINT on Training Data ====")
        trained_mint_model = train_mint(mint_model, train_loader, device)
        
        # Evaluate trained MINT on test data
        print("\n==== Evaluating MINT on Test Data ====")
        mint_metrics, mint_predictions, mint_labels = evaluate_mint(
            trained_mint_model, test_loader, device
        )
        
        if len(mint_predictions) > 0:
            print("\nMINT Prediction Results:")
            for metric, value in mint_metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
            
            # Save results
            results['mint'] = {
                'metrics': mint_metrics,
                'predictions': mint_predictions,
                'true_values': mint_labels
            }
        else:
            print("No valid MINT predictions generated")
            results['mint'] = {
                'metrics': {},
                'predictions': np.array([]),
                'true_values': np.array([])
            }
        
    except Exception as e:
        print(f"Error processing MINT: {e}")
        traceback.print_exc()
        results['mint'] = {
            'metrics': {},
            'predictions': np.array([]),
            'true_values': np.array([])
        }
    
    # Create comparison visualizations
    print("\n==== Creating Comparison Visualizations ====")
    create_comparison_plots(results['balm'], results['mint'], args.output_dir)
    
    print(f"\nComparison results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()