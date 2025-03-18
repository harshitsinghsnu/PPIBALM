import os
from tdc.multi_pred import DTI
import pandas as pd

# Create a custom dataset class for PPB-Affinity
class PPBAffinity:
    def __init__(self, *args, **kwargs):
        # Extract parameters
        data_path = kwargs.get("data_path", "D:\\BALM Fineclone\\BALM-PPI\\scripts\\notebooks\\Data.csv")
        self.train_ratio = kwargs.get("train_ratio")
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data directly from CSV instead of HuggingFace
        self.data = pd.read_csv(data_path)
        
        # Extract target values
        self.y = self.data['Y'].tolist() if 'Y' in self.data.columns else []
        self.dataset_name = "PPB-Affinity"

# Simplified DATASET_MAPPING with only your custom dataset
DATASET_MAPPING = {
    "PPB-Affinity": PPBAffinity
}

def get_dataset(dataset_name, harmonize_affinities_mode=None, *args, **kwargs):
    """
    Get dataset for protein-protein interaction studies.
    """
    if dataset_name.startswith("DTI_"):
        dti_dataset_name = dataset_name.replace("DTI_", "")
        dataset = DTI(name=dti_dataset_name)
        if harmonize_affinities_mode:
            dataset.harmonize_affinities(mode=harmonize_affinities_mode)
            # Convert $K_d$ to $pKd$
            dataset.convert_to_log(form="binding")
    elif dataset_name in DATASET_MAPPING:
        dataset = DATASET_MAPPING[dataset_name](*args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Ensure protein sequences are strings
    if hasattr(dataset, 'data'):
        dataset.data['Target'] = dataset.data['Target'].astype(str)
        dataset.data['proteina'] = dataset.data['proteina'].astype(str)

    return dataset