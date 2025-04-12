import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BALM-PPI data to MINT format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./mint_data", help="Output directory")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Train-test split ratio")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed for splitting")
    return parser.parse_args()

def convert_data(input_file, output_dir, split_ratio, random_seed):
    """
    Convert BALM-PPI data format to MINT format
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")
    
    # Check if required columns exist
    if 'Target' not in df.columns or 'proteina' not in df.columns or 'Y' not in df.columns:
        print("Warning: Expected columns 'Target', 'proteina', and 'Y' not found")
        
        # Look for protein sequence columns and binding affinity column
        sequence_cols = []
        target_col = None
        
        for col in df.columns:
            # Look for columns that might contain protein sequences
            if df[col].dtype == object and df[col].str.contains('[A-Z]{10,}').any():
                sequence_cols.append(col)
            # Look for numeric columns that might be binding affinities
            elif df[col].dtype in [np.float64, np.int64] and col != 'Unnamed: 0':
                target_col = col
        
        if len(sequence_cols) >= 2 and target_col:
            # Rename columns to match MINT format
            df = df.rename(columns={
                sequence_cols[0]: 'Target',
                sequence_cols[1]: 'proteina',
                target_col: 'Y'
            })
            print(f"Renamed columns: {sequence_cols[0]} -> Target, {sequence_cols[1]} -> proteina, {target_col} -> Y")
        else:
            print("Error: Could not identify appropriate columns for protein sequences and binding affinity")
            sys.exit(1)
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, train_size=split_ratio, random_state=random_seed
    )
    
    train_df, val_df = train_test_split(
        train_val_df, train_size=0.8, random_state=random_seed
    )
    
    print(f"Split data into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Saved splits to {output_dir}")
    
    # Return paths to split files
    return {
        'train': os.path.join(output_dir, "train.csv"),
        'val': os.path.join(output_dir, "val.csv"),
        'test': os.path.join(output_dir, "test.csv")
    }

def main():
    args = parse_args()
    convert_data(args.input_file, args.output_dir, args.split_ratio, args.random_seed)

if __name__ == "__main__":
    main()