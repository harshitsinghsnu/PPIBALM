import sys
import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random  # Add this import for the MINT code
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run the benchmark pipeline for protein-protein binding models")
    parser.add_argument('--config', type=str, required=True, help='Path to the benchmark configuration JSON file')
    return parser.parse_args()

# Load configuration
def load_benchmark_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded configuration from {config_path}")
    
    # Add timestamp if not present
    if 'timestamp' not in config:
        config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    return config

def create_directory_structure(config):
    """Create the necessary directory structure for benchmarking."""
    dirs = [
        os.path.join(config['output_dir'], 'config'),
        os.path.join(config['output_dir'], 'models'),
        os.path.join(config['output_dir'], 'results/BALM-PPI'),
        os.path.join(config['output_dir'], 'results/MINT'),
        os.path.join(config['output_dir'], 'results/comparison'),
        os.path.join(config['output_dir'], 'embeddings/BALM-PPI'),
        os.path.join(config['output_dir'], 'embeddings/MINT')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Directory structure created")

def save_config(config):
    """Save the benchmark configuration."""
    config_path = os.path.join(config['output_dir'], 'config', f"benchmark_config_{config['timestamp']}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")
    return config_path

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

def prepare_data(data_path, test_size, random_seed, output_dir, timestamp):
    """Prepare the dataset for benchmarking."""
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} samples")
    
    # Calculate bounds
    data_min = df['Y'].min()
    data_max = df['Y'].max()
    print(f"Data range: {data_min:.4f} to {data_max:.4f}")
    
    # Set seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)  # Add this for MINT
    
    # Split data - single split for consistent comparison
    train_data, test_data = train_test_split(
        df, test_size=test_size, random_state=random_seed
    )
    
    # Save splits
    train_path = os.path.join(output_dir, f'train_split_{timestamp}.csv')
    test_path = os.path.join(output_dir, f'test_split_{timestamp}.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"✅ Data prepared and saved:")
    print(f"  - {len(train_data)} training samples saved to {train_path}")
    print(f"  - {len(test_data)} test samples saved to {test_path}")
    
    return {
        'train_path': train_path,
        'test_path': test_path,
        'data_min': data_min,
        'data_max': data_max,
        'data_stats': {
            'min': data_min,
            'max': data_max,
            'mean': df['Y'].mean(),
            'std': df['Y'].std()
        }
    }

def initialize_balm(config_path, checkpoint_path, device="cuda"):
    """Initialize BALM-PPI model."""
    print("Initializing BALM-PPI model...")
    
    # Import BALM modules 
    from balm import common_utils
    from balm.configs import Configs
    from balm.models import BALM
    
    # Load configurations
    print(f"Loading configuration from {config_path}")
    configs = Configs(**common_utils.load_yaml(config_path))
    
    # Initialize model
    model = BALM(configs.model_configs)
    model = model.to(device)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                # Only load model parameters, ignore metadata
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading direct state dict
                model.load_state_dict(checkpoint)
            print(f"✅ BALM-PPI model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("Continuing with untrained model...")
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} not found, using untrained model")
    
    model.eval()
    return model, configs

def initialize_mint(config_path, checkpoint_path, device="cuda"):
    """Initialize MINT model."""
    print("Initializing MINT model...")
    
    # Make sure random module is in __builtins__
    if 'random' not in dir(__builtins__):
        import random
        __builtins__.random = random
    
    # Add MINT directory to path
    mint_dir = os.path.dirname(os.path.dirname(config_path))
    if mint_dir not in sys.path:
        sys.path.append(mint_dir)
        print(f"Added MINT path to sys.path: {mint_dir}")
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        return None, None
        
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
        return None, None
    
    try:
        # Add random to globals in case MINT uses it without importing
        globals()['random'] = random
        
        # Import mint modules
        import mint
        from mint.helpers.extract import MINTWrapper, load_config, CollateFn
        
        # Add the random module to mint.helpers.extract
        mint.helpers.extract.random = random
        
        print(f"Successfully imported MINT from {mint.__file__}")
        
        # Load configurations
        cfg = load_config(config_path)
        
        # Initialize model wrapper
        model = MINTWrapper(
            cfg, 
            checkpoint_path, 
            freeze_percent=1.0, 
            use_multimer=True, 
            sep_chains=True,
            device=device
        )
        model.eval()
        print(f"✅ MINT model loaded from {checkpoint_path}")
        
        return model, cfg
    except Exception as e:
        print(f"❌ Failed to initialize MINT model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def process_balm(model, data_path, data_min, data_max, device):
    """Process data with BALM-PPI model."""
    df = pd.read_csv(data_path)
    embeddings = []
    
    batch_size = 8  # Adjust based on GPU memory
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="BALM-PPI Processing"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Scale targets to cosine similarity range
            cosine_targets = [2 * (y - data_min) / (data_max - data_min) - 1 for y in batch_df['Y']]
            
            inputs = {
                "protein_sequences": batch_df["Target"].tolist(),
                "proteina_sequences": batch_df["proteina"].tolist(),
                "labels": torch.tensor(cosine_targets, dtype=torch.float32).to(device)
            }
            
            outputs = model(inputs)
            
            # Get protein embeddings - concatenate both protein embeddings
            protein_emb = outputs["protein_embedding"]
            proteina_emb = outputs["proteina_embedding"]
            combined_emb = torch.cat([protein_emb, proteina_emb], dim=1)
            
            embeddings.append(combined_emb.cpu())
    
    return torch.cat(embeddings, dim=0)

def process_mint(model, data_path, device):
    """Process data with MINT model."""
    df = pd.read_csv(data_path)
    embeddings = []
    
    # Import mint modules and make sure random is available
    import random
    from mint.helpers.extract import CollateFn
    
    # Add random to the module in case it's not imported there
    sys.modules['mint.helpers.extract'].random = random
    
    batch_size = 4  # MINT may require smaller batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Monkey patch the convert method if needed
    collate_fn = CollateFn(truncation_seq_length=2048)
    
    # Add safety for random if not present
    if not hasattr(collate_fn, 'random'):
        collate_fn.random = random
    
    try:
        # Patch the convert method directly
        original_convert = collate_fn.convert
        
        def safe_convert(seq):
            # Make sure random is available
            import random
            sys.modules['mint.helpers.extract'].random = random
            return original_convert(seq)
        
        collate_fn.convert = safe_convert
    except Exception as e:
        print(f"Warning: Could not patch convert method: {e}")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="MINT Processing"):
            try:
                batch_df = df.iloc[i:i+batch_size]
                
                # Prepare input in MINT format
                batch_data = list(zip(batch_df["Target"].tolist(), batch_df["proteina"].tolist()))
                chains, chain_ids = collate_fn(batch_data)
                
                # Move to device
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                
                # Get embeddings
                emb = model(chains, chain_ids)
                embeddings.append(emb.cpu())
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next batch instead of failing
                continue
    
    return torch.cat(embeddings, dim=0) if embeddings else torch.zeros(1)



def generate_embeddings(model_name, model_type, model, data, output_dir, timestamp, device="cuda"):
    """Generate embeddings for a model."""
    print(f"Generating embeddings for {model_name}...")
    embeddings_dir = os.path.join(output_dir, 'embeddings', model_name)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    if model_type == 'balm':
        # Process with BALM-PPI
        train_embeddings = process_balm(model, data['train_path'], data['data_min'], data['data_max'], device)
        test_embeddings = process_balm(model, data['test_path'], data['data_min'], data['data_max'], device)
    elif model_type == 'mint':
        # Process with MINT
        train_embeddings = process_mint(model, data['train_path'], device)
        test_embeddings = process_mint(model, data['test_path'], device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save embeddings
    train_output = os.path.join(embeddings_dir, f"train_embeddings_{timestamp}.pt")
    test_output = os.path.join(embeddings_dir, f"test_embeddings_{timestamp}.pt")
    
    torch.save(train_embeddings, train_output)
    torch.save(test_embeddings, test_output)
    
    print(f"✅ Embeddings generated and saved for {model_name}")
    return {
        'train_embeddings_path': train_output,
        'test_embeddings_path': test_output
    }

def train_linear_model(embeddings_path, labels_path, output_dir, model_name, timestamp, random_seed):
    """Train a linear model on the embeddings."""
    print(f"Training linear model for {model_name}...")
    
    # Load embeddings and labels
    embeddings = torch.load(embeddings_path)
    df = pd.read_csv(labels_path)
    labels = df['Y'].values
    
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    # Standardize embeddings
    embeddings_mean = embeddings.mean(axis=0, keepdims=True)
    embeddings_std = embeddings.std(axis=0, keepdims=True)
    embeddings = (embeddings - embeddings_mean) / (embeddings_std + 1e-8)
    
    # Train Ridge regression model
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    
    # Setup model with hyperparameter search
    model = Ridge(random_state=random_seed)
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False]
    }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    # Train model
    grid_search.fit(embeddings, labels)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Save model and metadata
    model_dir = os.path.join(output_dir, 'results', model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model_output = os.path.join(model_dir, f"ridge_model_{timestamp}.pkl")
    metadata_output = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
    
    import joblib
    joblib.dump(best_model, model_output)
    
    with open(metadata_output, 'w') as f:
        json.dump({
            'best_params': {k: float(v) if isinstance(v, np.float64) else v for k, v in best_params.items()},
            'embeddings_path': embeddings_path,
            'labels_path': labels_path,
            'embeddings_mean': embeddings_mean.tolist(),
            'embeddings_std': embeddings_std.tolist(),
            'random_seed': random_seed
        }, f, indent=2)
    
    print(f"✅ Linear model trained and saved for {model_name}")
    print(f"  - Best parameters: {best_params}")
    
    return {
        'model_path': model_output,
        'metadata_path': metadata_output,
        'best_model': best_model,
        'best_params': best_params,
        'embeddings_mean': embeddings_mean,
        'embeddings_std': embeddings_std
    }

def create_regression_plot(y_true, y_pred, metrics, output_dir, model_name, timestamp):
    """Create and save regression plot."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with regression line
    ax = sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.5})
    
    # Add identity line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line')
    
    # Add metrics to plot
    metrics_text = f"RMSE: {metrics['rmse']:.4f}\n" \
                   f"Pearson: {metrics['pearson']:.4f}\n" \
                   f"Spearman: {metrics['spearman']:.4f}\n" \
                   f"CI: {metrics['ci']:.4f}"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Experimental pKd')
    ax.set_ylabel('Predicted pKd')
    ax.set_title(f'{model_name} Regression Analysis')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"regression_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Regression plot saved to {plot_path}")

def evaluate_model(model_info, embeddings_path, labels_path, output_dir, model_name, timestamp):
    """Evaluate a trained model on test data."""
    print(f"Evaluating {model_name} model...")
    
    # Load model and metadata
    best_model = model_info['best_model']
    embeddings_mean = model_info['embeddings_mean']
    embeddings_std = model_info['embeddings_std']
    
    # Load test embeddings and labels
    embeddings = torch.load(embeddings_path)
    df = pd.read_csv(labels_path)
    labels = df['Y'].values
    
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()
    
    # Standardize embeddings using training stats
    embeddings = (embeddings - embeddings_mean) / (embeddings_std + 1e-8)
    
    # Generate predictions
    predictions = best_model.predict(embeddings)
    
    # Calculate metrics
    metrics = {}
    metrics['rmse'] = float(np.sqrt(mean_squared_error(labels, predictions)))
    metrics['pearson'] = float(pearsonr(labels, predictions)[0])
    metrics['spearman'] = float(spearmanr(labels, predictions)[0])
    metrics['ci'] = float(calculate_ci(labels, predictions))
    
    # Save results
    results_dir = os.path.join(output_dir, 'results', model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    results_output = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
    predictions_output = os.path.join(results_dir, f"predictions_{timestamp}.csv")
    
    with open(results_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_df = df.copy()
    predictions_df['prediction'] = predictions
    predictions_df['error'] = predictions - labels
    predictions_df.to_csv(predictions_output, index=False)
    
    print(f"✅ Evaluation complete for {model_name}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - Pearson: {metrics['pearson']:.4f}")
    print(f"  - Spearman: {metrics['spearman']:.4f}")
    print(f"  - CI: {metrics['ci']:.4f}")
    
    # Create and save regression plot
    create_regression_plot(labels, predictions, metrics, results_dir, model_name, timestamp)
    
    return {
        'metrics': metrics,
        'results_path': results_output,
        'predictions_path': predictions_output
    }

def evaluate_model_directly(model, model_name, model_type, data_path, output_dir, data_min, data_max, timestamp, device="cuda"):
    """Evaluate model directly on test data without using embeddings + linear regression."""
    print(f"Directly evaluating {model_name} model...")
    
    # Create output directory
    direct_dir = os.path.join(output_dir, 'results', model_name, "direct_eval")
    os.makedirs(direct_dir, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(data_path)
    
    # Track predictions and labels
    predictions = []
    labels = []
    
    # Batch processing
    batch_size = 8  # Adjust based on model and GPU memory
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Process data based on model type
    with torch.no_grad():
        model.eval()  # Ensure model is in evaluation mode
        
        if model_type == 'balm':
            for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc=f"Direct {model_name} Prediction"):
                batch_df = df.iloc[i:i+batch_size]
                
                # Scale targets to cosine similarity range
                cosine_targets = [2 * (y - data_min) / (data_max - data_min) - 1 for y in batch_df['Y']]
                
                # Prepare input
                inputs = {
                    "protein_sequences": batch_df["Target"].tolist(),
                    "proteina_sequences": batch_df["proteina"].tolist(),
                    "labels": torch.tensor(cosine_targets, dtype=torch.float32).to(device)
                }
                
                # Get model outputs
                outputs = model(inputs)
                
                # Convert predictions to pKd
                if "cosine_similarity" in outputs:
                    batch_preds = model.cosine_similarity_to_pkd(
                        outputs["cosine_similarity"],
                        pkd_upper_bound=data_max,
                        pkd_lower_bound=data_min
                    ).cpu().numpy()
                else:
                    batch_preds = outputs["logits"].cpu().numpy()
                
                # Store predictions and true values
                predictions.extend(batch_preds)
                labels.extend(batch_df["Y"].values)
                
        elif model_type == 'mint':
            # Import MINT processing utilities if needed
            try:
                from mint.helpers.extract import CollateFn
                collate_fn = CollateFn(truncation_seq_length=2048)
                
                for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc=f"Direct {model_name} Prediction"):
                    batch_df = df.iloc[i:i+batch_size]
                    
                    # Prepare data in MINT format
                    batch_data = list(zip(batch_df["Target"].tolist(), batch_df["proteina"].tolist()))
                    chains, chain_ids = collate_fn(batch_data)
                    
                    # Move to device
                    chains = chains.to(device)
                    chain_ids = chain_ids.to(device)
                    
                    # Get predictions from MINT
                    batch_preds = model.predict(chains, chain_ids).cpu().numpy()
                    
                    predictions.extend(batch_preds)
                    labels.extend(batch_df["Y"].values)
            except Exception as e:
                print(f"Error processing MINT model: {e}")
                return {
                    'metrics': {'rmse': 0, 'pearson': 0, 'spearman': 0, 'ci': 0},
                    'results_path': '',
                    'plot_path': ''
                }
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    metrics = {}
    metrics['rmse'] = float(np.sqrt(mean_squared_error(labels, predictions)))
    metrics['pearson'] = float(pearsonr(labels, predictions)[0])
    metrics['spearman'] = float(spearmanr(labels, predictions)[0])
    metrics['ci'] = float(calculate_ci(labels, predictions))
    
    print(f"✅ Direct evaluation complete for {model_name}")
    print(f"  - RMSE: {metrics['rmse']:.4f}")
    print(f"  - Pearson: {metrics['pearson']:.4f}")
    print(f"  - Spearman: {metrics['spearman']:.4f}")
    print(f"  - CI: {metrics['ci']:.4f}")
    
    # Create regression plot
    plt.figure(figsize=(10, 8))
    ax = sns.regplot(x=labels, y=predictions)
    ax.set_title(f"{model_name} Direct Evaluation")
    ax.set_xlabel(r"Experimental $pK_d$")
    ax.set_ylabel(r"Predicted $pK_d$")
    
    # Add metrics to plot
    metrics_text = f"RMSE: {metrics['rmse']:.4f}\n" \
                   f"Pearson: {metrics['pearson']:.4f}\n" \
                   f"Spearman: {metrics['spearman']:.4f}\n" \
                   f"CI: {metrics['ci']:.4f}"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plot_path = os.path.join(direct_dir, f"direct_regression_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results as CSV
    results_df = pd.DataFrame({
        'True_pKd': labels,
        'Predicted_pKd': predictions,
        'Error': predictions - labels
    })
    results_path = os.path.join(direct_dir, f"direct_predictions_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    return {
        'metrics': metrics,
        'results_path': results_path,
        'plot_path': plot_path
    }

def create_comparison_visualizations(model_results, output_dir, timestamp):
    """Create comparison visualizations between models."""
    print("Creating comparison visualizations...")
    
    # Create separate DataFrames for direct and embedding-based evaluations
    direct_metrics = []
    embedding_metrics = []
    
    for model_name, results in model_results.items():
        # Direct evaluation metrics
        if 'direct_evaluation' in results:
            direct_metrics.append({
                'model': f"{model_name} (Direct)",
                **results['direct_evaluation']['metrics']
            })
        
        # Embedding-based metrics
        if 'evaluation' in results:
            embedding_metrics.append({
                'model': f"{model_name} (Embedding)",
                **results['evaluation']['metrics']
            })
    
    # Combine both types of metrics
    all_metrics = direct_metrics + embedding_metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save comparison table
    comparison_dir = os.path.join(output_dir, 'results', 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    table_path = os.path.join(comparison_dir, f"comparison_table_{timestamp}.csv")
    metrics_df.to_csv(table_path, index=False)
    
    # Create bar charts for each metric
    metrics_to_plot = ['rmse', 'pearson', 'spearman', 'ci']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        
        # Create bar chart
        ax = sns.barplot(x='model', y=metric, data=metrics_df)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom')
        
        # Set title and labels
        metric_names = {'rmse': 'RMSE (lower is better)', 
                       'pearson': 'Pearson Correlation', 
                       'spearman': 'Spearman Correlation',
                       'ci': 'Concordance Index'}
        
        plt.title(metric_names.get(metric, metric))
        plt.ylabel(metric_names.get(metric, metric))
        plt.xlabel('')
    
    plt.tight_layout()
    
    # Save visualization
    comparison_path = os.path.join(comparison_dir, f"metrics_comparison_{timestamp}.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison visualizations saved to {comparison_dir}")
    
    # Create comprehensive results dictionary
    results_table = {
        'timestamp': timestamp,
        'models': {},
        'comparison': {
            'table_path': table_path,
            'visualization_path': comparison_path
        }
    }
    
    for model_name, results in model_results.items():
        results_table['models'][model_name] = {
            'direct_metrics': results.get('direct_evaluation', {}).get('metrics', {}),
            'embedding_metrics': results.get('evaluation', {}).get('metrics', {}),
            'paths': {
                'direct_results': results.get('direct_evaluation', {}).get('results_path', ''),
                'embedding_results': results.get('evaluation', {}).get('results_path', '')
            }
        }
    
    # Save comprehensive results
    results_json_path = os.path.join(comparison_dir, f"benchmark_results_{timestamp}.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_table, f, indent=2)
    
    print(f"✅ Complete benchmark results saved to {results_json_path}")
    
    return results_table

def main():
    """Main function to run the benchmark."""
    # Parse command line arguments
    args = parse_args()
    
    # Load benchmark configuration
    config = load_benchmark_config(args.config)
    timestamp = config['timestamp']
    
    print(f"Starting benchmark pipeline at {timestamp}")
    
    # Create directory structure
    create_directory_structure(config)
    
    # Save config
    config_path = save_config(config)
    
    # Prepare data
    data_info = prepare_data(
        config['dataset_path'],
        config['test_size'],
        config['random_seeds'][0],
        config['output_dir'],
        timestamp
    )
    
    # Track results for each model
    model_results = {}
    
    # Process each model
    for model_name, model_config in config['models'].items():
        print(f"\n{'=' * 40}\nProcessing {model_name}\n{'=' * 40}")
        
        # Initialize model
        if model_config['type'] == 'balm':
            model, configs = initialize_balm(
                model_config['config_path'],
                model_config['checkpoint_path']
            )
        elif model_config['type'] == 'mint':
            model, configs = initialize_mint(
                model_config['config_path'],
                model_config['checkpoint_path']
            )
        else:
            print(f"⚠️ Unknown model type: {model_config['type']}")
            continue
        
        if model is None:
            print(f"⚠️ Failed to initialize model {model_name}, skipping")
            continue
        
        # Initialize results dictionary for this model
        model_results[model_name] = {}
        
        # Direct evaluation if enabled
        if config.get('use_direct_evaluation', False):
            direct_eval_results = evaluate_model_directly(
                model,
                model_name,
                model_config['type'],
                data_info['test_path'],
                config['output_dir'],
                data_info['data_min'],
                data_info['data_max'],
                timestamp
            )
            model_results[model_name]['direct_evaluation'] = direct_eval_results
        
        # Generate embeddings for embedding-based evaluation
        embedding_paths = generate_embeddings(
            model_name,
            model_config['type'],
            model,
            data_info,
            config['output_dir'],
            timestamp
        )
        model_results[model_name]['embeddings'] = embedding_paths
        
        # Train linear model on embeddings
        training_info = train_linear_model(
            embedding_paths['train_embeddings_path'],
            data_info['train_path'],
            config['output_dir'],
            model_name,
            timestamp,
            config['random_seeds'][0]
        )
        model_results[model_name]['training'] = training_info
        
        # Evaluate embedding-based model
        evaluation_info = evaluate_model(
            training_info,
            embedding_paths['test_embeddings_path'],
            data_info['test_path'],
            config['output_dir'],
            model_name,
            timestamp
        )
        model_results[model_name]['evaluation'] = evaluation_info
    
    # Create comparison visualizations
    comparison_results = create_comparison_visualizations(
        model_results,
        config['output_dir'],
        timestamp
    )
    
    print("\n✅ Benchmark pipeline completed successfully!")
    return model_results

if __name__ == "__main__":
    # Clear out any paths that might cause conflicts
    sys.path = [p for p in sys.path if 'balm - cursor' not in p]

    # Add the project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    sys.path.insert(0, project_root)

    try:
        # Start the benchmark
        main()
    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)