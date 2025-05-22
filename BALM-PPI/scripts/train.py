import os
import sys
import time
import random # Added import
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import BALM modules
from balm import common_utils
from balm.configs import Configs, FineTuningType # Added FineTuningType
from balm.models import BALM, BaselineModel
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman

def parse_args():
    parser = argparse.ArgumentParser(description="Train BALM model with different PEFT methods")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--wandb_project", type=str, default="ppiATTPFT", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")

def load_checkpoint(model, optimizer, file_path):
    try:
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {file_path}, resuming from epoch {start_epoch}")
        return start_epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set random seed for reproducibility
    seed = 42 # Or args.seed if you add a seed argument to argparse
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    # For deterministic operations
    # torch.backends.cudnn.deterministic = True # Can impact performance
    # torch.backends.cudnn.benchmark = False   # Can impact performance
    
    print(f"Loading config from {args.config_filepath}")
    
    # Set device
    if torch.cuda.is_available():
        DEVICE = "cuda" 
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load config
    try:
        configs = Configs(**common_utils.load_yaml(args.config_filepath))
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Override epochs if specified in arguments
    if args.epochs is not None:
        configs.training_configs.epochs = args.epochs
        print(f"Overriding epochs to {args.epochs}")
    
    # Create output directory if it doesn't exist
    os.makedirs(configs.training_configs.outputs_dir, exist_ok=True)
    
    # Create figures directory
    figures_dir = os.path.join(configs.training_configs.outputs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize model based on configuration
    if hasattr(configs.model_configs, 'model_type') and configs.model_configs.model_type.upper() == 'BASELINE':
        model = BaselineModel(configs.model_configs)
    else:
        model = BALM(configs.model_configs)
    
    model = model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        params=[param for name, param in model.named_parameters() if param.requires_grad],
        lr=configs.model_configs.model_hyperparameters.learning_rate,
    )
    
    print(f"Starting training with learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded data with {len(df)} samples")
    
    # Calculate data bounds
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Split data
    train_ratio = configs.dataset_configs.train_ratio if hasattr(configs.dataset_configs, 'train_ratio') else 0.2
    # Use the unified seed for train_test_split
    random_state_seed = seed 
    
    train_data, test_data = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_state_seed # Updated to use unified seed
    )
    print(f"Number of train data: {len(train_data)}")
    print(f"Number of test data: {len(test_data)}")
    
    # Initialize wandb if not disabled
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            wandb.login()
            
            # Determine model type string
            model_type_str = "BALM" # Default
            if hasattr(configs.model_configs, 'model_type') and configs.model_configs.model_type:
                model_type_str = str(configs.model_configs.model_type.value).upper()


            # Determine PEFT description for run name
            active_peft_methods = []
            if hasattr(configs.model_configs, 'protein_fine_tuning_type') and \
               configs.model_configs.protein_fine_tuning_type not in [None, FineTuningType.BASELINE]:
                   active_peft_methods.append(str(configs.model_configs.protein_fine_tuning_type.value))
            
            if hasattr(configs.model_configs, 'proteina_fine_tuning_type') and \
               configs.model_configs.proteina_fine_tuning_type not in [None, FineTuningType.BASELINE]:
                   active_peft_methods.append(str(configs.model_configs.proteina_fine_tuning_type.value))

            peft_description = "_".join(sorted(list(set(active_peft_methods))))

            if peft_description:
                run_name = f"{model_type_str}_{peft_description}_training"
            else:
                run_name = f"{model_type_str}_training"
            
            print(f"Wandb run name: {run_name}") # Log the generated run name

            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
            
            # Log hyperparameters
            wandb.config.learning_rate = configs.model_configs.model_hyperparameters.learning_rate
            wandb.config.num_epochs = configs.training_configs.epochs
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            use_wandb = False
    
    # Check for checkpoint
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(configs.training_configs.outputs_dir, "latest_checkpoint.pth")
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Update checkpoint path for saving
    save_checkpoint_path = os.path.join(configs.training_configs.outputs_dir, "latest_checkpoint.pth")
    best_model_path = os.path.join(configs.training_configs.outputs_dir, "best_model.pth")
    
    # Training loop
    num_epochs = configs.training_configs.epochs
    start = time.time()
    best_loss = float('inf')
    
    # Track losses for plotting
    train_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        
        # Create progress bar for each epoch
        pbar = tqdm(train_data.iterrows(), total=len(train_data), 
                    desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        batch_losses = []  # Track individual batch losses
        
        for idx, sample in pbar:
            # Scale to cosine similarity range
            cosine_target = 2 * (sample['Y'] - data_min) / (data_max - data_min) - 1
            
            inputs = {
                "protein_sequences": [sample["Target"]],
                "proteina_sequences": [sample["proteina"]],
                "labels": torch.tensor([cosine_target], dtype=torch.float32).to(DEVICE)
            }

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = outputs["loss"]
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            current_loss = loss.item()
            batch_losses.append(current_loss)
            total_loss += current_loss
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'batch_loss': f'{current_loss:.4f}',
                'avg_loss': f'{total_loss/(idx+1):.4f}'
            })

        avg_loss = total_loss / len(train_data)
        train_losses.append(avg_loss)
        
        # Print detailed epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        if batch_losses:
            print(f"Min Batch Loss: {min(batch_losses):.4f}")
            print(f"Max Batch Loss: {max(batch_losses):.4f}")
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss achieved: {best_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, best_model_path)

        # Print memory usage if using CUDA
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch+1, save_checkpoint_path)

    training_time = time.time() - start
    print("\nTraining complete!")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Average time per epoch: {training_time/num_epochs/60:.2f} minutes")
    print(f"Best loss achieved: {best_loss:.4f}")
    
    # Testing phase
    model.eval()
    predictions = []
    labels = []
    start = time.time()

    print("Starting evaluation on test set...")
    test_pbar = tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")
    
    for _, sample in test_pbar:
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
                                                          pkd_upper_bound=data_max, 
                                                          pkd_lower_bound=data_min)
            else:
                # Assume BaselineModel, output["logits"] are in [-1, 1] range
                scaled_prediction = output["logits"]
                # Clamp to ensure it's within the expected range
                clamped_scaled_prediction = torch.clamp(scaled_prediction, -1.0, 1.0)
                # Scale back to original pKd range
                pkd_range = data_max - data_min
                prediction = ((clamped_scaled_prediction + 1.0) / 2.0) * pkd_range + data_min
                # Final clamp to ensure it's within the original data bounds
                prediction = torch.clamp(prediction, data_min, data_max)
            
        label = torch.tensor([sample["Y"]])
        
        predictions.append(prediction.item())
        labels.append(label.item())
        
        # Occasionally print predictions
        if len(predictions) % 500 == 0:
            print(f"\nPredicted pKd: {prediction.item():.4f} | True pKd: {label.item():.4f}")
    
    # Log the total time taken
    test_time = time.time() - start
    if use_wandb:
        wandb.log({"test_time": test_time})

    print(f"Time taken for {len(test_data)} protein-proteina pairs: {test_time:.2f} seconds")
    
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
    if use_wandb:
        wandb.log({
            "RMSE": rmse.item(),
            "Pearson": pearson.item(),
            "Spearman": spearman.item(),
            "CI": ci.item()
        })

    # Create regression plot
    # Use model_type_str for consistency in plot titles
    plot_title_model_part = model_type_str 
    if peft_description: # peft_description was defined during wandb setup
        plot_title_peft_part = peft_description
    else: # Fallback if wandb was not used, recalculate peft_description
        active_peft_methods_plot = []
        if hasattr(configs.model_configs, 'protein_fine_tuning_type') and \
           configs.model_configs.protein_fine_tuning_type not in [None, FineTuningType.BASELINE]:
               active_peft_methods_plot.append(str(configs.model_configs.protein_fine_tuning_type.value))
        if hasattr(configs.model_configs, 'proteina_fine_tuning_type') and \
           configs.model_configs.proteina_fine_tuning_type not in [None, FineTuningType.BASELINE]:
               active_peft_methods_plot.append(str(configs.model_configs.proteina_fine_tuning_type.value))
        plot_title_peft_part = "_".join(sorted(list(set(active_peft_methods_plot))))

    final_plot_title = f"{plot_title_model_part}_{plot_title_peft_part}_Evaluation" if plot_title_peft_part else f"{plot_title_model_part}_Evaluation"

    plt.figure(figsize=(10, 8))
    ax = sns.regplot(x=labels, y=predictions)
    ax.set_title(final_plot_title)
    ax.set_xlabel(r"Experimental $pK_d$")
    ax.set_ylabel(r"Predicted $pK_d$")

    # Save the plot
    plot_path = os.path.join(figures_dir, "regression_plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Also save training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{final_plot_title} - Loss Curve") # Use the same title structure
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "loss_curve.png"))
    plt.close()

    # Log the plots to W&B
    if use_wandb:
        wandb.log({
            "regression_plot": wandb.Image(plot_path),
            "loss_curve": wandb.Image(os.path.join(figures_dir, "loss_curve.png"))
        })
        wandb.finish()
    
    print(f"Results and plots saved to {figures_dir}")
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()