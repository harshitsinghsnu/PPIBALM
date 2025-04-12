import os
import sys
import argparse
import requests
from tqdm import tqdm

def download_file(url, target_path):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(target_path)) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return target_path

def main():
    parser = argparse.ArgumentParser(description="Download MINT model checkpoint for benchmarking")
    parser.add_argument("--output_dir", type=str, default="../models", help="Directory to save model")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # MINT model URL on HuggingFace
    model_url = "https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt"
    target_path = os.path.join(args.output_dir, "mint.ckpt")
    
    print(f"Downloading MINT model checkpoint to {target_path}...")
    download_file(model_url, target_path)
    print("✅ MINT model downloaded successfully")

if __name__ == "__main__":
    main()