import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import os

try:
    # Try relative imports first (when imported as a package)
    from .utils import *
    from .params import *
except ImportError:
    # Fall back to absolute imports (when running directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import *
    from src.params import *

class LabelNormalization():
    def __init__(self, label):
        self.mean = label.mean()
        self.std = label.std()

        self.min = label.min()
        self.max = label.max()

    def normalize(self, label):
        return 2*(label - self.min) / (self.max - self.min) - 1

    def unnormalize(self, normalized_label):
        return (normalized_label + 1)*(self.max - self.min)/2 + self.min

def clean_json_file(file_path):
    """
    Remove 'control_points_array' key from JSON file to clean up unused data.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return False
    
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Remove control_points_array if it exists
        if 'control_points_array' in data:
            del data['control_points_array']
            print(f"Removed 'control_points_array' from {file_path}")
            
            # Write back the cleaned data
            with open(file_path, 'w') as f:
                json.dump(data, f)
            print(f"Successfully cleaned and saved {file_path}")
            return True
        else:
            print(f"'control_points_array' not found in {file_path}")
            return False
            
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False

def tpms_dataset(file_path, return_normalizer = False, multi_objective_cond = False): 

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            tokens = np.array(data['tokens'])
            control_points_stress = np.array(data['control_points_stress'])
            label_array = np.array(data['label_array'])
            if multi_objective_cond:
                stiffness = np.array(data['stiffness'])
                moduli = np.array(data['moduli'])
    else:
        raise ValueError(f"File {file_path} does not exist")

    if np.unique(tokens[:,seq_len:].flatten()).shape[0] > 1:
        print('Error in tokenization')
        exit()
    else:
        tokens = tokens[:,:seq_len]

    label_normalizer = LabelNormalization(control_points_stress)
    control_points_stress = label_normalizer.normalize(control_points_stress)

    # Helper function to get multi-condition data
    def get_multi_objective_cond_data():
        if multi_objective_cond_type == 'stiffness':
            return np.array(data['stiffness'])
        else:
            moduli_data = np.array(data['moduli'])
            if multi_label_dim == 3:
                return moduli_data[:, multi_objective_cond_col]
            elif multi_label_dim == 12:
                return moduli_data
            else:
                raise ValueError(f"Unsupported multi_label_dim: {multi_label_dim}")
    
    # Convert numpy arrays to tensors
    tokens_tensor = torch.from_numpy(np.array(tokens)).long()
    control_points_tensor = torch.from_numpy(control_points_stress).float()
    label_array_tensor = torch.from_numpy(label_array).float()
    
    # Create dataset based on conditions
    if multi_objective_cond:
        multi_objective_cond_data = get_multi_objective_cond_data()
        multi_objective_cond_tensor = torch.from_numpy(multi_objective_cond_data).float()
        dataset = TensorDataset(tokens_tensor, control_points_tensor, label_array_tensor, multi_objective_cond_tensor)
    else:
        dataset = TensorDataset(tokens_tensor, control_points_tensor, label_array_tensor)
    
    # Return dataset with or without normalizer
    if return_normalizer:
        return dataset, label_normalizer
    else:
        return dataset

def get_dataset(data_path, return_normalizer = False, multi_objective_cond = False):

    return tpms_dataset(data_path, return_normalizer = return_normalizer, multi_objective_cond = multi_objective_cond)

# Example usage of dataset functions
if __name__ == "__main__":

    data_path = '../data/dataset/dataset_multi_objective_cond.json'
    print("\n=== Dataset Loading ===")
    print("Loading dataset")
    try:
        dataset_basic = get_dataset(data_path, return_normalizer=False, multi_objective_cond=False)
        print(f"Basic dataset loaded with {len(dataset_basic)} samples")
        
        # Show structure of a sample
        sample = dataset_basic[0]
        print(f"Sample structure: {len(sample)} tensors")
        print(f"  - Tokens shape: {sample[0].shape}")
        print(f"  - Control points shape: {sample[1].shape}")
        print(f"  - Label array shape: {sample[2].shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
    