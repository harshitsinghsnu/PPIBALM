# convert_balm_data.py
import pandas as pd
import argparse
import os

def convert_data(input_path, output_path):
    """Convert data from BALM format to a standard format compatible with both models"""
    print(f"Converting data from {input_path} to {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded data with {len(df)} rows")
    
    # Check which columns exist
    required_columns = ["Target", "proteina", "Y"]
    alternative_columns = {
        "Target": ["protein", "sequence_1", "seq1"],
        "proteina": ["protein_a", "sequence_2", "seq2"],
        "Y": ["target", "label", "binding_affinity"]
    }
    
    # Map columns to required format
    for req_col in required_columns:
        if req_col not in df.columns:
            # Try alternative column names
            found = False
            for alt_col in alternative_columns[req_col]:
                if alt_col in df.columns:
                    print(f"Mapping column {alt_col} to {req_col}")
                    df[req_col] = df[alt_col]
                    found = True
                    break
            
            if not found:
                print(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Could not find column {req_col} or any alternatives in the input data")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save only necessary columns
    df[required_columns].to_csv(output_path, index=False)
    print(f"Converted data saved to {output_path} with {len(df)} rows")

def main():
    parser = argparse.ArgumentParser(description="Convert BALM data format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    args = parser.parse_args()
    
    convert_data(args.input, args.output)

if __name__ == "__main__":
    main()