import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

from balm import common_utils
from balm.configs import Configs
from balm.models import BALM, BaselineModel
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman

def parse_args():
    parser = argparse.ArgumentParser(description="Train BALM model with different PEFT methods")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--wandb_project", type=str, default="ppiATTPFT", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to load checkpoint from")
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'peft_config': model.model_configs.peft_configs if hasattr(model.model_configs, 'peft_configs') else None
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch} to {file_path}")

def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {file_path}, resuming from epoch {start_epoch}")
    return start_epoch

def main():
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
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    
    # Create output directory if it doesn't exist
    os.makedirs(configs.training_configs.outputs_dir, exist_ok=True)
    
    # Initialize model based on configuration
    if hasattr(configs.model_configs, 'model_type') and configs.model_configs.model_type.upper() == 'BASELINE':
        model = BaselineModel(configs.model_configs)
    else:
        model = BALM(configs.model_configs)
    
    model = model.to(DEVICE)
    
    # Check if PEFT is enabled
    peft_method = "none"
    if hasattr(configs.model_configs, 'peft_configs') and hasattr(configs.model_configs.peft_configs, 'enabled') and configs.model_configs.peft_configs.enabled:
        if hasattr(configs.model_configs.peft_configs, 'protein') and hasattr(configs.model_configs.peft_configs.protein, 'method'):
            peft_method = configs.model_configs.peft_configs.protein.method
        print(f"PEFT method: {peft_method}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        params=[param for name, param in model.named_parameters() if param.requires_grad],
        lr=configs.model_configs.model_hyperparameters.learning_rate,
    )
    
    print(f"Starting training with learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Training on device: {DEVICE}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded data with {len(df)} samples")
    
    # Calculate data bounds
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Split data
    train_data, test_data = train_test_split(
        df, 
        train_size=configs.dataset_configs.train_ratio if hasattr(configs.dataset_configs, 'train_ratio') else 0.2, 
        random_state=configs.training_configs.random_seed if hasattr(configs.training_configs, 'random_seed') else 1234
    )
    print(f"Number of train data: {len(train_data)}")
    print(f"Number of test data: {len(test_data)}")
    
    # Calculate bounds for normalization
    pkd_lower_bound = df['Y'].min()
    pkd_upper_bound = df['Y'].max()
    print(f"pkd_lower_bound: {pkd_lower_bound}, pkd_upper_bound: {pkd_upper_bound}")
    
    # Initialize wandb
    wandb.login()
    
    # Extract model type and PEFT method for run name
    model_type = "BALM"
    if hasattr(configs.model_configs, 'model_type'):
        model_type = configs.model_configs.model_type
    
    run_name = f"{model_type}_{peft_method}_training"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
    
    # Log hyperparameters
    wandb.config.learning_rate = configs.model_configs.model_hyperparameters.learning_rate
    wandb.config.num_epochs = configs.training_configs.epochs
    
    # Check for checkpoint
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(configs.training_configs.outputs_dir, "latest_checkpoint.pth")
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # Update checkpoint path for saving
    save_checkpoint_path = os.path.join(configs.training_configs.outputs_dir, "latest_checkpoint.pth")
    
    # Training loop
    num_epochs = configs.training_configs.epochs
    start = time.time()
    best_loss = float('inf')
    
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

            # Print sequence lengths occasionally
            if idx % 100 == 0:
                print(f"\nSequence lengths - Target: {len(sample['Target'])}, "
                      f"ProteinA: {len(sample['proteina'])}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = outputs["loss"]
            loss.backward()
            
            # Gradient norm debugging
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if idx % 100 == 0:
                print(f"Gradient norm: {grad_norm:.4f}")

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
        
        # Print detailed epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Min Batch Loss: {min(batch_losses):.4f}")
        print(f"Max Batch Loss: {max(batch_losses):.4f}")
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss achieved: {best_loss:.4f}")

        # Print memory usage if using CUDA
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Log metrics to wandb
        wandb.log({'epoch': epoch + 1, 'loss': avg_loss})
        
        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, epoch+1, save_checkpoint_path)

    training_time = time.time() - start
    print("\nTraining complete!")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Average time per epoch: {training_time/num_epochs/60:.2f} minutes")
    print(f"Best loss achieved: {best_loss:.4f}")
    
    # Testing phase
    model = model.eval()
    predictions = []
    labels = []
    start = time.time()

    print("Starting evaluation on test set...")
    test_pbar = tqdm(test_data.iterrows(), total=len(test_data), desc="Testing")
    
    for _, sample in test_pbar:
        # Scale target to cosine similarity range (same as training)
        cosine_target = 2 * (sample['Y'] - data_min) / (data_max - data_min) - 1
        
        # Prepare input (exactly matching training format)
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
    
    # Log the total time taken
    test_time = time.time() - start
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
    plot_path = os.path.join(configs.training_configs.outputs_dir, "regression_plot.png")
    plt.savefig(plot_path)

    # Log the plot to W&B
    wandb.log({"regression_plot": wandb.Image(plot_path)})

    wandb.finish()

if __name__ == "__main__":
    main()