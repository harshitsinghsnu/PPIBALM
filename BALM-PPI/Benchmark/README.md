# BALM-PPI Benchmarking Framework

This directory contains tools and scripts for benchmarking BALM-PPI against other protein-protein binding affinity prediction models, including MINT.

## Directory Structure

```
Benchmark/
├── config/                # Configuration files for benchmarking
├── models/                # Store downloaded model weights
├── results/               # Store all benchmark results
│   ├── BALM-PPI/          # Your model results
│   ├── MINT/              # MINT model results
│   └── comparison/        # Comparison visualizations
├── embeddings/            # Store embeddings for each model
└── scripts/               # Benchmarking scripts
```

## Quick Start

1. **Setup the environment**:
   Make sure you have all dependencies installed from the main BALM-PPI environment.

2. **Download models**:
   ```bash
   cd scripts
   python download_mint.py
   ```

3. **Create a benchmark configuration**:
   ```bash
   python create_benchmark_config.py --include_balm_variants
   ```

4. **Run the benchmark**:
   ```bash
   python benchmark_pipeline.py
   ```

5. **Run multiple benchmarks**:
   ```bash
   python run_all_benchmarks.py
   ```

## Detailed Instructions

### Creating Configurations

You can create customized benchmark configurations using `create_benchmark_config.py`:

```bash
# Include all BALM-PPI variants
python create_benchmark_config.py --include_balm_variants --output ../config/all_variants.json

# Compare only with MINT
python create_benchmark_config.py --no_balm_variants --output ../config/mint_only.json

# Use specific model checkpoints
python create_benchmark_config.py --balm_checkpoint "../../outputs/best_checkpoint.pth" --output ../config/custom_checkpoint.json
```

### Running Benchmarks

The `benchmark_pipeline.py` script runs a complete benchmark workflow:

1. Prepares the dataset, splitting into train and test sets
2. Initializes each model
3. Generates embeddings
4. Trains linear models on the embeddings
5. Evaluates on test data
6. Creates comparison visualizations

```bash
python benchmark_pipeline.py --config ../config/benchmark_config.json
```

### Viewing Results

Results are saved in the `results/` directory:

- Individual model results in `results/MODEL_NAME/`
- Comparison visualizations in `results/comparison/`
- Complete benchmark results in `results/benchmark_results_TIMESTAMP.json`

## Understanding Results

### Metrics

The following metrics are calculated for each model:

- **RMSE (Root Mean Square Error)**: Lower is better
- **Pearson Correlation**: Higher is better (range: -1 to 1)
- **Spearman Correlation**: Higher is better (range: -1 to 1)
- **Concordance Index (CI)**: Higher is better (range: 0 to 1)

### Visualizations

- **Regression Plots**: Scatter plots with trend lines showing predicted vs. actual values
- **Metric Comparisons**: Bar charts comparing metrics across models
- **Error Distributions**: Histograms showing the distribution of prediction errors

## Adding New Models

To add a new model to the benchmark, modify the configuration file to include the new model:

```json
"NEW_MODEL_NAME": {
    "type": "custom",
    "config_path": "path/to/config",
    "checkpoint_path": "path/to/checkpoint"
}
```

Then implement the corresponding model initialization and processing functions in `benchmark_pipeline.py`.

## Troubleshooting

- **Missing checkpoints**: Ensure all model checkpoints are available at the specified paths
- **GPU memory issues**: Reduce batch sizes in embedding generation
- **Missing dependencies**: Make sure all required packages are installed

## Citation

If you use this benchmarking framework in your research, please cite the BALM-PPI paper: [citation]