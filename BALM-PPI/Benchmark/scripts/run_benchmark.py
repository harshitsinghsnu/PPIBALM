#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple benchmark script for comparing BALM-PPI and MINT.
"""

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
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import json

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark BALM-PPI and MINT on binding affinity prediction")
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to the dataset CSV file")
    parser.add_argument("--balm_root", type=str, required=True,
                      help="Path to BALM-PPI root directory")
    parser.add_argument("--balm_config", type=str, default="default_configs/balm_peft.yaml",
                      help="Path to BALM-PPI config file (relative to balm_root)")
    parser.add_argument("--balm_checkpoint", type=str, default="outputs/latest_checkpoint.pth",
                      help="Path to BALM-PPI checkpoint file (relative to balm_root)")
    parser.add_argument("--mint_dir", type=str, default="mint",
                      help="Path to MINT directory")
    parser.add_argument("--mint_checkpoint", type=str, default="../models/mint.ckpt",
                      help="Path to MINT checkpoint file")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--test_size", type=float, default=0.8,
                      help="Fraction of data to use for testing")
    parser.add_argument("--random_seed", type=int, default=123,
                      help="Random seed for data splitting and model initialization")
    parser.add_argument("--no_balm", action="store_true", 
                      help="Skip BALM-PPI evaluation")
    parser.add_argument("--no_mint", action="store_true",
                      help="Skip MINT evaluation")
    return parser.parse_args()

def main():
    """Main function to run the benchmark."""
    # Parse arguments
    args = parse_args()
    
    # Add BALM-PPI to path
    balm_root = os.path.abspath(args.balm_root)
    if balm_root not in sys.path:
        sys.path.insert(0, balm_root)
    print(f"Added BALM-PPI path: {balm_root}")
    
    # Import BALM modules
    try:
        from balm import common_utils
        from balm.configs import Configs
        from balm.models import BALM
        from balm.metrics import get_pearson, get_rmse, get_spearman, get_ci
        print("Successfully imported BALM-PPI modules")
    except ImportError as e:
        print(f"Failed to import BALM-PPI modules: {e}")
        if not args.no_balm:
            print("Cannot proceed with BALM-PPI evaluation. Use --no_balm to skip.")
            return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "BALM-PPI"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "MINT"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)
    
    # Load and prepare data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Calculate bounds
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Split data
    train_data, test_data = train_test_split(
        df, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Save splits
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = os.path.join(args.output_dir, f'train_split_{timestamp}.csv')
    test_path = os.path.join(args.output_dir, f'test_split_{timestamp}.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Data prepared and saved:")
    print(f"  - {len(train_data)} training samples saved to {train_path}")
    print(f"  - {len(test_data)} test samples saved to {test_path}")
    
    # Results dictionary
    results = {
        'timestamp': timestamp,
        'models': {}
    }
    
    # Evaluate BALM-PPI
    if not args.no_balm:
        print("\n" + "="*40)
        print("Evaluating BALM-PPI")
        print("="*40)
        
        balm_config_path = os.path.join(balm_root, args.balm_config)
        balm_checkpoint_path = os.path.join(balm_root, args.balm_checkpoint)
        
        try:
            # Load configuration
            print(f"Loading configuration from {balm_config_path}")
            configs = Configs(**common_utils.load_yaml(balm_config_path))
            
            # Initialize model
            model = BALM(configs.model_configs)
            model = model.to(device)
            
            # Load checkpoint
            print(f"Loading checkpoint from {balm_checkpoint_path}")
            checkpoint = torch.load(balm_checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully")
            
            # Direct evaluation on test data
            model.eval()
            predictions = []
            labels = []
            
            # Process in batches
            batch_size = 8
            total_batches = (len(test_data) + batch_size - 1) // batch_size
            
            print("Evaluating BALM-PPI directly...")
            with torch.no_grad():
                for i in tqdm(range(0, len(test_data), batch_size), total=total_batches, desc="BALM-PPI Prediction"):
                    batch_df = test_data.iloc[i:i+batch_size]
                    
                    # Scale targets to cosine similarity range
                    cosine_targets = [2 * (y - data_min) / (data_max - data_min) - 1 for y in batch_df['Y']]
                    
                    inputs = {
                        "protein_sequences": batch_df["Target"].tolist(),
                        "proteina_sequences": batch_df["proteina"].tolist(),
                        "labels": torch.tensor(cosine_targets, dtype=torch.float32).to(device)
                    }
                    
                    outputs = model(inputs)
                    
                    # Convert predictions
                    if "cosine_similarity" in outputs:
                        batch_preds = model.cosine_similarity_to_pkd(
                            outputs["cosine_similarity"],
                            pkd_upper_bound=data_max,
                            pkd_lower_bound=data_min
                        )
                    else:
                        batch_preds = outputs["logits"]
                    
                    # Store results
                    batch_preds = batch_preds.cpu().numpy()
                    batch_labels = batch_df["Y"].values
                    
                    predictions.extend(batch_preds)
                    labels.extend(batch_labels)
            
            # Calculate metrics
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            rmse = np.sqrt(mean_squared_error(labels, predictions))
            pearson = pearsonr(labels, predictions)[0]
            spearman = spearmanr(labels, predictions)[0]
            
            # Calculate CI
            total_pairs = 0
            concordant_pairs = 0
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if labels[i] != labels[j]:  # Only count if there's a difference
                        total_pairs += 1
                        if (labels[i] > labels[j] and predictions[i] > predictions[j]) or \
                           (labels[i] < labels[j] and predictions[i] < predictions[j]):
                            concordant_pairs += 1
            ci = concordant_pairs / total_pairs if total_pairs > 0 else 0.5
            
            # Print metrics
            print("\nBALM-PPI Direct Evaluation Results:")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - Pearson: {pearson:.4f}")
            print(f"  - Spearman: {spearman:.4f}")
            print(f"  - CI: {ci:.4f}")
            
            # Create regression plot
            plt.figure(figsize=(10, 8))
            ax = sns.regplot(x=labels, y=predictions, scatter_kws={'alpha': 0.5})
            min_val = min(min(labels), min(predictions))
            max_val = max(max(labels), max(predictions))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line')
            
            # Add metrics to plot
            metrics_text = f"RMSE: {rmse:.4f}\n" \
                          f"Pearson: {pearson:.4f}\n" \
                          f"Spearman: {spearman:.4f}\n" \
                          f"CI: {ci:.4f}"
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Experimental pKd')
            ax.set_ylabel('Predicted pKd')
            ax.set_title('BALM-PPI Direct Evaluation')
            
            # Save plot
            plot_path = os.path.join(args.output_dir, "BALM-PPI", f"direct_regression_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add results
            results['models']['BALM-PPI'] = {
                'direct': {
                    'rmse': float(rmse),
                    'pearson': float(pearson),
                    'spearman': float(spearman),
                    'ci': float(ci),
                    'plot': plot_path
                }
            }
            
            # Save predictions
            results_df = pd.DataFrame({
                'True_pKd': labels,
                'Predicted_pKd': predictions,
                'Error': predictions - labels
            })
            
            pred_path = os.path.join(args.output_dir, "BALM-PPI", f"predictions_{timestamp}.csv")
            results_df.to_csv(pred_path, index=False)
            
        except Exception as e:
            print(f"Error evaluating BALM-PPI: {e}")
            import traceback
            traceback.print_exc()
    
    # Skip MINT for now as it requires more setup
    if not args.no_mint:
        print("\n" + "="*40)
        print("MINT evaluation requires additional setup and is skipped in this simplified script.")
        print("Please see the comprehensive benchmark for MINT evaluation.")
        print("="*40)
    
    # Save final results
    results_path = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("Benchmark completed successfully!")

if __name__ == "__main__":
    main()