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

from balm import common_utils
from balm.configs import Configs
from balm.models import BALM, BaselineModel
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman

def parse_args():
    parser = argparse.ArgumentParser(description="Test BALM model with different PEFT methods")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test CSV data file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to load checkpoint from")
    parser.add_argument("--no_wandb", action="store_true", help="Disable logging metrics to wandb")
    parser.add_argument("--wandb_project", type=str, default="ppiATTPFT", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs (defaults to checkpoint directory)")
    parser.add_argument("--data_min", type=float, default=None, help="Minimum value of Y for normalization (default: min value in the test data)")
    parser.add_argument("--data_max", type=float, default=None, help="Maximum value of Y for normalization (default: max value in the test data)")
    parser.add_argument("--pkd_lower_bound", type=float, default=None, help="Lower bound of pKd values (default: min value in the test data)")
    parser.add_argument("--pkd_upper_bound", type=float, default=None, help="Upper bound of pKd values (default: max value in the test data)")
    return parser.parse_args()

def load_checkpoint(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: Checkpoint does not contain 'model_state_dict'. Trying to load directly...")
            model.load_state_dict(checkpoint)
        
        epoch = checkpoint.get('epoch', 0)
        return epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def main():
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Set device
        if torch.cuda.is_available():
            DEVICE = "cuda" 
        else:
            DEVICE = "cpu"
        print(f"Using device: {DEVICE}")
        
        # Set random seed
        seed = 42
        torch.cuda.manual_seed(seed)
        
        # Load config
        try:
            configs = Configs(**common_utils.load_yaml(args.config_filepath))
        except Exception as e:
            print(f"Error loading config from {args.config_filepath}: {e}")
            return
        
        # Create output directory if it doesn't exist
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model based on configuration
        try:
            if hasattr(configs.model_configs, 'model_type') and configs.model_configs.model_type.upper() == 'BASELINE':
                model = BaselineModel(configs.model_configs)
            else:
                model = BALM(configs.model_configs)
            
            model = model.to(DEVICE)
        except Exception as e:
            print(f"Error initializing model: {e}")
            return
        
        # Load trained model
        try:
            epoch = load_checkpoint(model, args.checkpoint_path)
            print(f"Loaded model from checkpoint at epoch {epoch}")
        except Exception as e:
            print(f"Error loading checkpoint from {args.checkpoint_path}: {e}")
            return
        
        # Check if PEFT is enabled
        peft_method = "none"
        if hasattr(configs.model_configs, 'peft_configs') and hasattr(configs.model_configs.peft_configs, 'enabled') and configs.model_configs.peft_configs.enabled:
            if hasattr(configs.model_configs.peft_configs, 'protein') and hasattr(configs.model_configs.peft_configs.protein, 'method'):
                peft_method = configs.model_configs.peft_configs.protein.method
            print(f"PEFT method: {peft_method}")
        
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
        
        # Calculate data bounds
        data_min = args.data_min if args.data_min is not None else df['Y'].min()
        data_max = args.data_max if args.data_max is not None else df['Y'].max()
        print(f"Data range for normalization: {data_min:.4f} to {data_max:.4f}")
        
        # Calculate bounds for conversion
        pkd_lower_bound = args.pkd_lower_bound if args.pkd_lower_bound is not None else df['Y'].min()
        pkd_upper_bound = args.pkd_upper_bound if args.pkd_upper_bound is not None else df['Y'].max()
        print(f"pKd bounds for conversion: {pkd_lower_bound:.4f} to {pkd_upper_bound:.4f}")
        
        # Initialize wandb
        args.use_wandb = not args.no_wandb  # Default to using wandb, unless --no_wandb is specified
        if args.use_wandb:
            try:
                wandb.login()
                
                # Extract model type and PEFT method for run name
                model_type = "BALM"
                if hasattr(configs.model_configs, 'model_type'):
                    model_type = configs.model_configs.model_type
                
                run_name = f"{model_type}_{peft_method}_testing_epoch_{epoch}"
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Continuing without wandb logging...")
                args.use_wandb = False
        
        # Testing phase
        model = model.eval()
        predictions = []
        labels = []
        start = time.time()

        print("Starting evaluation on test set...")
        test_pbar = tqdm(df.iterrows(), total=len(df), desc="Testing")
        
        for _, sample in test_pbar:
            try:
                # Scale target to cosine similarity range
                cosine_target = 2 * (sample['Y'] - data_min) / (data_max - data_min) - 1
                
                # Prepare input
                inputs = {
                    "protein_sequences": [sample["Target"]],
                    "proteina_sequences": [sample["proteina"]],
                    "labels": torch.tensor([cosine_target], dtype=torch.float32).to(DEVICE)
                }
                
                with torch.no_grad():
                    output = model(inputs)
                    
                    # Handle different model types
                    if hasattr(model, 'cosine_similarity_to_pkd') and "cosine_similarity" in output:
                        prediction = model.cosine_similarity_to_pkd(output["cosine_similarity"], 
                                                                   pkd_upper_bound=pkd_upper_bound, 
                                                                   pkd_lower_bound=pkd_lower_bound)
                    else:
                        # Assume BaselineModel with direct pKd prediction (logits)
                        prediction = output["logits"]
                    
                label = torch.tensor([sample["Y"]])
                
                predictions.append(prediction.item())
                labels.append(label.item())
                
                # Occasionally print predictions
                if len(predictions) % 500 == 0:
                    print(f"\nPredicted pKd: {prediction.item():.4f} | True pKd: {label.item():.4f}")
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        if not predictions:
            print("No valid predictions were made. Exiting.")
            return
            
        # Log the total time taken
        test_time = time.time() - start
        if args.use_wandb:
            wandb.log({"test_time": test_time})

        print(f"Time taken for {len(df)} protein-proteina pairs: {test_time:.2f} seconds")
        
        # Calculate metrics
        tensor_labels = torch.tensor(labels)
        tensor_predictions = torch.tensor(predictions)
        
        rmse = get_rmse(tensor_labels, tensor_predictions)
        pearson = get_pearson(tensor_labels, tensor_predictions)
        spearman = get_spearman(tensor_labels, tensor_predictions)
        ci = get_ci(tensor_labels, tensor_predictions)

        print(f"RMSE: {rmse}")
        print(f"Pearson: {pearson}")
        print(f"Spearman: {spearman}")
        print(f"CI: {ci}")

        # Log metrics to W&B
        if args.use_wandb:
            wandb.log({
                "RMSE": rmse.item(),
                "Pearson": pearson.item(),
                "Spearman": spearman.item(),
                "CI": ci.item()
            })

        # Create regression plot
        plt.figure(figsize=(10, 8))
        ax = sns.regplot(x=labels, y=predictions)
        ax.set_title(f"{run_name}")
        ax.set_xlabel(r"Experimental $pK_d$")
        ax.set_ylabel(r"Predicted $pK_d$")

        # Save the plot
        plot_path = os.path.join(output_dir, "regression_plot.png")
        plt.savefig(plot_path)

        # Log the plot to W&B
        if args.use_wandb:
            wandb.log({"regression_plot": wandb.Image(plot_path)})

        # Save results to CSV
        results_df = pd.DataFrame({
            'True_pKd': labels,
            'Predicted_pKd': predictions
        })
        results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

        # Final report
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            f.write(f"Model: {run_name}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"RMSE: {rmse.item()}\n")
            f.write(f"Pearson: {pearson.item()}\n")
            f.write(f"Spearman: {spearman.item()}\n")
            f.write(f"CI: {ci.item()}\n")
            f.write(f"Test time: {test_time:.2f} seconds\n")
        
        print("Results saved to:", output_dir)
        
        if args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()