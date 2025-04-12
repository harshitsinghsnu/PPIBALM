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

# Add the parent directory to the path (if needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
            print(f"Loading configuration from {args.config_filepath}")
            configs = Configs(**common_utils.load_yaml(args.config_filepath))
            print("Configuration loaded successfully")
        except Exception as e:
            print(f"Error loading config from {args.config_filepath}: {e}")
            return
        
        # Create output directory if it doesn't exist
        output_dir = args.output_dir if args.output_dir else os.path.dirname(args.checkpoint_path)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create figures directory structure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_figures_dir = os.path.join(output_dir, "figures", timestamp)
        model_type = getattr(configs.model_configs, 'model_type', "BALM")
        peft_method = "none"
        
        if hasattr(configs.model_configs, 'peft_configs') and hasattr(configs.model_configs.peft_configs, 'enabled') and configs.model_configs.peft_configs.enabled:
            if hasattr(configs.model_configs.peft_configs, 'protein') and hasattr(configs.model_configs.peft_configs.protein, 'method'):
                peft_method = configs.model_configs.peft_configs.protein.method
        
        figures_dirs = {
            "evaluation": os.path.join(base_figures_dir, "evaluation", f"{model_type}_{peft_method}"),
            "predictions": os.path.join(base_figures_dir, "predictions")
        }

        for dir_path in figures_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
        
        # Initialize model based on configuration
        try:
            if hasattr(configs.model_configs, 'model_type') and configs.model_configs.model_type.upper() == 'BASELINE':
                model = BaselineModel(configs.model_configs)
                print("Initialized BaselineModel")
            else:
                model = BALM(configs.model_configs)
                print("Initialized BALM model")
            
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
                
                run_name = f"{model_type}_{peft_method}_testing_epoch_{epoch}_{timestamp}"
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
                
                # Log test configuration
                wandb.config.update({
                    "config_file": args.config_filepath,
                    "checkpoint_path": args.checkpoint_path,
                    "data_path": args.data_path,
                    "data_min": data_min,
                    "data_max": data_max,
                    "pkd_lower_bound": pkd_lower_bound,
                    "pkd_upper_bound": pkd_upper_bound
                })
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                print("Continuing without wandb logging...")
                args.use_wandb = False
        
        # Testing phase
        model = model.eval()
        predictions = []
        labels = []
        protein_seqs = []
        proteina_seqs = []
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
                protein_seqs.append(sample["Target"])
                proteina_seqs.append(sample["proteina"])
                
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
        ax.set_title(f"{model_type}_{peft_method} Evaluation (Epoch {epoch})")
        ax.set_xlabel(r"Experimental $pK_d$")
        ax.set_ylabel(r"Predicted $pK_d$")
        
        # Add metrics to plot
        metrics_text = f"RMSE: {rmse.item():.4f}\n" \
                       f"Pearson: {pearson.item():.4f}\n" \
                       f"Spearman: {spearman.item():.4f}\n" \
                       f"CI: {ci.item():.4f}"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save the plot
        regression_path = os.path.join(figures_dirs["evaluation"], f"regression_plot_{timestamp}.png")
        plt.savefig(regression_path)
        plt.close()
        print(f"Saved regression plot to {regression_path}")

        # Log the plot to W&B
        if args.use_wandb:
            wandb.log({"regression_plot": wandb.Image(regression_path)})

        # Create histogram of errors
        errors = np.array(predictions) - np.array(labels)
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f"{model_type}_{peft_method} Prediction Errors")
        plt.xlabel("Prediction Error (Predicted - True)")
        plt.ylabel("Count")
        
        # Save the error histogram
        error_hist_path = os.path.join(figures_dirs["evaluation"], f"error_histogram_{timestamp}.png")
        plt.savefig(error_hist_path)
        plt.close()

        # Create scatter plot of errors vs true values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=labels, y=errors)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f"{model_type}_{peft_method} Error vs True Value")
        plt.xlabel("True pKd")
        plt.ylabel("Prediction Error")
        
        # Save the error scatter plot
        error_scatter_path = os.path.join(figures_dirs["evaluation"], f"error_scatter_{timestamp}.png")
        plt.savefig(error_scatter_path)
        plt.close()

        # Save results to CSV with all data
        results_df = pd.DataFrame({
            'Target_Sequence': protein_seqs,
            'ProteinA_Sequence': proteina_seqs,
            'True_pKd': labels,
            'Predicted_pKd': predictions,
            'Error': np.array(predictions) - np.array(labels)
        })
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f"test_results_detailed_{timestamp}.csv")
        results_df.to_csv(detailed_path, index=False)
        
        # Save summary results (without sequences to save space)
        summary_df = pd.DataFrame({
            'True_pKd': labels,
            'Predicted_pKd': predictions,
            'Error': np.array(predictions) - np.array(labels)
        })
        summary_path = os.path.join(output_dir, f"test_results_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)

        # Save metrics to text file
        metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {model_type}_{peft_method}\n")
            f.write(f"Config: {args.config_filepath}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Test data: {args.data_path}\n")
            f.write(f"Number of samples: {len(labels)}\n")
            f.write(f"Data range: {data_min:.4f} to {data_max:.4f}\n")
            f.write(f"pKd bounds: {pkd_lower_bound:.4f} to {pkd_upper_bound:.4f}\n")
            f.write(f"RMSE: {rmse.item():.4f}\n")
            f.write(f"Pearson: {pearson.item():.4f}\n")
            f.write(f"Spearman: {spearman.item():.4f}\n")
            f.write(f"CI: {ci.item():.4f}\n")
            f.write(f"Mean absolute error: {np.mean(np.abs(errors)):.4f}\n")
            f.write(f"Max absolute error: {np.max(np.abs(errors)):.4f}\n")
            f.write(f"Test time: {test_time:.2f} seconds\n")
        
        # Save metrics as JSON for easier programmatic access
        json_metrics = {
            "model_type": model_type,
            "peft_method": peft_method,
            "epoch": epoch,
            "test_size": len(labels),
            "rmse": float(rmse.item()),
            "pearson": float(pearson.item()),
            "spearman": float(spearman.item()),
            "ci": float(ci.item()),
            "mean_abs_error": float(np.mean(np.abs(errors))),
            "max_abs_error": float(np.max(np.abs(errors))),
            "test_time": float(test_time),
            "timestamp": timestamp
        }
        
        json_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
        with open(json_path, 'w') as f:
            import json
            json.dump(json_metrics, f, indent=2)
        
        print(f"Results and metrics saved to {output_dir}")
        print(f"Detailed results: {detailed_path}")
        print(f"Summary results: {summary_path}")
        print(f"Metrics (text): {metrics_path}")
        print(f"Metrics (JSON): {json_path}")
        
        if args.use_wandb:
            # Upload results to wandb
            wandb.log({
                "error_histogram": wandb.Image(error_hist_path),
                "error_scatter": wandb.Image(error_scatter_path),
                "detailed_results": wandb.Table(dataframe=results_df),
                "summary_results": wandb.Table(dataframe=summary_df),
                "metrics_json": json_metrics
            })
            wandb.finish()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()