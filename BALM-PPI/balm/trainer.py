import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from balm.models import BALM
from balm.metrics import get_ci, get_pearson, get_rmse, get_spearman

class ProteinPairDataset(Dataset):
    """Dataset for protein-protein pairs with binding affinity labels."""
    
    def __init__(self, dataframe, pkd_lower=None, pkd_upper=None):
        """
        Initialize dataset from a pandas DataFrame.
        
        Args:
            dataframe: DataFrame with protein, proteina, and Y columns
            pkd_lower: Lower bound for pKd values (for normalization)
            pkd_upper: Upper bound for pKd values (for normalization)
        """
        self.data = dataframe
        self.pkd_lower = pkd_lower if pkd_lower is not None else self.data['Y'].min()
        self.pkd_upper = pkd_upper if pkd_upper is not None else self.data['Y'].max()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Scale Y to cosine similarity range [-1, 1]
        cosine_target = 2 * (row['Y'] - self.pkd_lower) / (self.pkd_upper - self.pkd_lower) - 1
        
        return {
            "protein_sequences": row["Target"],
            "proteina_sequences": row["proteina"],
            "labels": torch.tensor([cosine_target], dtype=torch.float32)
        }

def collate_fn(batch):
    """Collate function for dataloader that handles variable length sequences."""
    protein_seqs = [item["protein_sequences"] for item in batch]
    proteina_seqs = [item["proteina_sequences"] for item in batch]
    labels = torch.cat([item["labels"] for item in batch])
    
    return {
        "protein_sequences": protein_seqs,
        "proteina_sequences": proteina_seqs,
        "labels": labels
    }

class Trainer:
    """Trainer class for BALM Protein-Protein Interaction models."""
    
    def __init__(self, config, wandb_entity=None, wandb_project=None, output_dir="outputs"):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object with model and training parameters
            wandb_entity: Weights & Biases entity name
            wandb_project: Weights & Biases project name
            output_dir: Directory to save outputs
        """
        self.config = config
        self.model_config = config.model_configs
        self.dataset_config = config.dataset_configs
        self.training_config = config.training_configs
        
        self.device = torch.device(f"cuda:{self.training_config.device}" 
                                  if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = BALM(self.model_config)
        self.model = self.model.to(self.device)
        
        # Tracking variables
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.best_model_path = None
        self.pkd_lower_bound = None
        self.pkd_upper_bound = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup_dataset(self, csv_path=None):
        """
        Load and prepare the dataset.
        
        Args:
            csv_path: Path to CSV file with protein data (optional)
        """
        # Try to find the dataset
        if csv_path is None:
            # Check common locations
            potential_paths = [
                f"data/{self.dataset_config.dataset_name}.csv",
                f"scripts/notebooks/{self.dataset_config.dataset_name}.csv",
                "scripts/notebooks/Data.csv"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if csv_path is None:
                raise FileNotFoundError(f"Could not find dataset for {self.dataset_config.dataset_name}")
        
        # Load the data
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset with {len(df)} samples from {csv_path}")
        
        # Calculate bounds for normalization
        self.pkd_lower_bound = df['Y'].min()
        self.pkd_upper_bound = df['Y'].max()
        print(f"pKd range: {self.pkd_lower_bound:.4f} to {self.pkd_upper_bound:.4f}")
        
        # Split the data
        if self.dataset_config.split_method == "random":
            train_ratio = self.dataset_config.train_ratio
            
            if train_ratio > 0:
                # Split into train/test
                train_val_df, test_df = train_test_split(
                    df, 
                    train_size=train_ratio,
                    random_state=self.training_config.random_seed
                )
                
                # Split train into train/val
                train_df, val_df = train_test_split(
                    train_val_df,
                    train_size=0.8,
                    random_state=self.training_config.random_seed
                )
                
                print(f"Split dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
            else:
                # Zero-shot: everything is test
                train_df, val_df = None, None
                test_df = df
                print(f"Zero-shot mode: all {len(test_df)} samples for testing")
        else:
            raise ValueError(f"Unsupported split method: {self.dataset_config.split_method}")
        
        # Create datasets and loaders
        if train_df is not None:
            train_dataset = ProteinPairDataset(train_df, self.pkd_lower_bound, self.pkd_upper_bound)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
        
        if val_df is not None:
            val_dataset = ProteinPairDataset(val_df, self.pkd_lower_bound, self.pkd_upper_bound)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        test_dataset = ProteinPairDataset(test_df, self.pkd_lower_bound, self.pkd_upper_bound)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    def save_checkpoint(self, epoch, optimizer, checkpoint_path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'pkd_lower_bound': self.pkd_lower_bound,
            'pkd_upper_bound': self.pkd_upper_bound
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, optimizer=None):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load pKd bounds if available
        if 'pkd_lower_bound' in checkpoint:
            self.pkd_lower_bound = checkpoint['pkd_lower_bound']
            self.pkd_upper_bound = checkpoint['pkd_upper_bound']
            print(f"Loaded pKd bounds: {self.pkd_lower_bound:.4f} to {self.pkd_upper_bound:.4f}")
        
        epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
        return epoch
    
    def train(self):
        """Execute training process."""
        # Check if we have training data
        if self.train_loader is None:
            print("No training data available. Skipping training.")
            return
        
        # Initialize wandb if available
        if self.wandb_entity and self.wandb_project:
            wandb.login()
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=self.config
            )
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.model_config.model_hyperparameters.learning_rate
        )
        
        # Load checkpoint if provided
        start_epoch = 0
        if self.model_config.checkpoint_path:
            checkpoint_path = self.model_config.checkpoint_path
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path, optimizer)
        
        # Training loop
        num_epochs = self.training_config.epochs
        start_time = time.time()
        best_loss = float('inf')
        patience = 10  # Early stopping patience
        no_improve_epochs = 0
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.model.train()
            total_loss = 0.0
            
            # Training progress bar
            pbar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}",
                leave=False
            )
            
            for batch in pbar:
                # Prepare inputs
                batch_labels = batch["labels"].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
            
            # Calculate average loss
            avg_loss = total_loss / len(self.train_loader)
            
            # Validation
            if self.val_loader:
                val_metrics = self.evaluate(self.val_loader, "val")
                val_loss = val_metrics["val/loss"]
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Pearson: {val_metrics['val/pearson']:.4f}")
                
                # Log to wandb
                if self.wandb_entity and self.wandb_project:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": avg_loss,
                        **val_metrics
                    })
                
                # Check for improvement
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_epochs = 0
                    
                    # Save best model
                    best_checkpoint_path = os.path.join(self.output_dir, "best_model.pth")
                    self.save_checkpoint(epoch + 1, optimizer, best_checkpoint_path)
                    self.best_model_path = best_checkpoint_path
                else:
                    no_improve_epochs += 1
                    print(f"No improvement for {no_improve_epochs} epochs")
                
                # Early stopping
                if no_improve_epochs >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                # No validation data, just log training loss
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {avg_loss:.4f}")
                
                if self.wandb_entity and self.wandb_project:
                    wandb.log({"epoch": epoch + 1, "train/loss": avg_loss})
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(self.output_dir, "latest_checkpoint.pth")
            self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation loss: {best_loss:.4f}")
        
        # Load best model for testing
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.load_checkpoint(self.best_model_path)
        
        # Run evaluation on test set
        if self.test_loader:
            print("\nEvaluating on test set:")
            test_metrics = self.evaluate(self.test_loader, "test", save_predictions=True)
            
            if self.wandb_entity and self.wandb_project:
                wandb.log(test_metrics)
        
        return self.best_model_path
    
    def evaluate(self, data_loader, split="test", save_predictions=False):
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            split: Dataset split name (e.g., "train", "val", "test")
            save_predictions: Whether to save predictions to CSV
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_protein_seqs = []
        all_proteina_seqs = []
        total_loss = 0.0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split}"):
                # Get inputs and compute predictions
                batch_labels = batch["labels"].to(self.device)
                outputs = self.model(batch)
                
                # Get loss
                if "loss" in outputs:
                    loss = outputs["loss"]
                    total_loss += loss.item()
                
                # Get predictions
                if self.model_config.loss_function == "cosine_mse":
                    predictions = outputs["cosine_similarity"]
                else:
                    predictions = outputs["logits"]
                
                # Store results
                all_labels.append(batch_labels.cpu())
                all_predictions.append(predictions.cpu())
                all_protein_seqs.extend(batch["protein_sequences"])
                all_proteina_seqs.extend(batch["proteina_sequences"])
        
        # Concatenate batched results
        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)
        
        # Calculate metrics
        metrics = {}
        
        # Add loss if available
        if len(data_loader) > 0:
            metrics[f"{split}/loss"] = total_loss / len(data_loader)
        
        # Convert from cosine similarity to pKd for metrics
        if self.model_config.loss_function == "cosine_mse":
            # Convert to original scale for metrics
            pkd_range = self.pkd_upper_bound - self.pkd_lower_bound
            labels_pkd = (all_labels + 1) / 2 * pkd_range + self.pkd_lower_bound
            predictions_pkd = (all_predictions + 1) / 2 * pkd_range + self.pkd_lower_bound
        else:
            labels_pkd = all_labels
            predictions_pkd = all_predictions
        
        # Calculate metrics
        metrics[f"{split}/rmse"] = get_rmse(labels_pkd, predictions_pkd)
        metrics[f"{split}/pearson"] = get_pearson(labels_pkd, predictions_pkd)
        metrics[f"{split}/spearman"] = get_spearman(labels_pkd, predictions_pkd)
        metrics[f"{split}/ci"] = get_ci(labels_pkd, predictions_pkd)
        
        # Print metrics
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  {name}: {value:.4f}")
        
        # Save predictions if requested
        if save_predictions:
            # Convert to numpy for saving
            labels_np = labels_pkd.numpy()
            preds_np = predictions_pkd.numpy()
            
            # Create dataframe
            results_df = pd.DataFrame({
                "protein": all_protein_seqs,
                "proteina": all_proteina_seqs,
                "label": labels_np,
                "prediction": preds_np
            })
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, f"{split}_predictions.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
            
            # Log to wandb if available
            if self.wandb_entity and self.wandb_project:
                # Create and log a scatter plot
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Create figures directory
                figures_dir = os.path.join(self.output_dir, "figures", "evaluation")
                os.makedirs(figures_dir, exist_ok=True)
                
                plt.figure(figsize=(8, 6))
                sns.regplot(x=labels_np, y=preds_np)
                plt.title(f"{split.capitalize()} Predictions")
                plt.xlabel("Experimental pKd")
                plt.ylabel("Predicted pKd")
                
                plot_path = os.path.join(figures_dir, f"{split}_plot.png")
                plt.savefig(plot_path)
                plt.close()
                
                wandb.log({f"{split}_plot": wandb.Image(plot_path)})
        
        return metrics