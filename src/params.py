"""
Parameters Configuration for Diffusion Model Training

This module contains all the global parameters and configuration settings for the diffusion model training pipeline. 

"""

# ========================================================================================
# DATA AND SEQUENCE PARAMETERS
# ========================================================================================

# Number of data points sampled per equation for material property evaluation
num_datapoints = 40

# Maximum sequence length for equation tokenization
seq_len = 22

# Dimension of material property labels (number of material properties)
label_dim = 11

# ========================================================================================
# MATERIAL PROPERTY SPECIFICATIONS
# ========================================================================================

# Material property names corresponding to elastic constants and Poisson's ratios
# E11, E22, E33: Young's moduli in principal directions
# G23, G31, G12: Shear moduli in principal planes  
# v21, v31, v32, v12, v13, v23: Poisson's ratios
s_name = [
    'E11', 'E22', 'E33',           # Young's moduli
    'G23', 'G31', 'G12',           # Shear moduli
    'v21', 'v31', 'v32',           # Poisson's ratios (first set)
    'v12', 'v13', 'v23'            # Poisson's ratios (second set)
]

# ========================================================================================
# MULTI-OBJECTIVE CONDITIONING CONFIGURATION
# ========================================================================================

# Whether to enable multi-objective conditioning (conditioning on multiple properties)
multi_objective_cond = False

# Type of multi-objective conditioning to use
# Options: 'stiffness', 'moduli', 'poissons_ratio'
multi_objective_cond_type = 'poissons_ratio'

# Configure multi-objective parameters based on conditioning type
if multi_objective_cond_type == 'stiffness':
    # Full stiffness matrix conditioning (21 independent components)
    multi_label_dim = 21
    multi_objective_cond_col = list(range(0, 21))
else:
    multi_label_dim = 3
    
    if multi_objective_cond_type == 'moduli':
        # Condition on Young's moduli (E11, E22, E33)
        multi_objective_cond_col = list(range(0, 3))
    elif multi_objective_cond_type == 'poissons_ratio':
        # Condition on specific Poisson's ratios (v21, v31, v32)
        multi_objective_cond_col = list(range(6, 9))
    else:
        raise ValueError(f"Unknown multi_objective_cond_type: {multi_objective_cond_type}")

# ========================================================================================
# DATASET PATH CONFIGURATION
# ========================================================================================


# Select dataset path based on conditioning type
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if multi_objective_cond:
    # Dataset with multi-objective conditioning data
    data_path = os.path.join(project_root, 'data', 'dataset', 'dataset_multi_objective_cond.json')
else:
    # Standard single-objective dataset
    data_path = os.path.join(project_root, 'data', 'dataset', 'dataset.json')

# ========================================================================================
# DIRECTORY AND TRAINING CONFIGURATION
# ========================================================================================

# Directory for wandb experiment tracking logs
wandb_run_dir = os.path.join(project_root, 'wandb')

# Directory for training runs and model outputs
run_dir = os.path.join(project_root, 'runs')

# Fraction of dataset to use for validation (0.1 = 10%)
valid_size = 0.1

def get_param_summary():
    """
    Get a summary of current parameter configuration.
    
    Returns:
        dict: Dictionary containing key parameter information
    """
    return {
        'data_config': {
            'num_datapoints': num_datapoints,
            'seq_len': seq_len,
            'label_dim': label_dim,
            'data_path': data_path
        },
        'conditioning_config': {
            'multi_objective_cond': multi_objective_cond,
            'multi_objective_cond_type': multi_objective_cond_type,
            'multi_label_dim': multi_label_dim if multi_objective_cond else None,
            'conditioning_columns': multi_objective_cond_col if multi_objective_cond else None
        },
        'training_config': {
            'valid_size': valid_size,
            'run_dir': run_dir,
            'wandb_run_dir': wandb_run_dir
        },
        'material_properties': s_name
    }


# ========================================================================================
# CONSTANTS AND DERIVED PARAMETERS
# ========================================================================================

# Indices for different material property groups
YOUNGS_MODULI_INDICES = [0, 1, 2]          # E11, E22, E33
SHEAR_MODULI_INDICES = [3, 4, 5]           # G23, G31, G12  
POISSONS_RATIO_INDICES = [6, 7, 8, 9, 10, 11]  # All Poisson's ratios

# Material property group names for easy reference
PROPERTY_GROUPS = {
    'youngs_moduli': [s_name[i] for i in YOUNGS_MODULI_INDICES],
    'shear_moduli': [s_name[i] for i in SHEAR_MODULI_INDICES],
    'poissons_ratios': [s_name[i] for i in POISSONS_RATIO_INDICES]
}

# Total number of material properties
NUM_MATERIAL_PROPERTIES = len(s_name)

# Conditioning dimension (depends on whether multi-objective is enabled)
CONDITIONING_DIM = multi_label_dim if multi_objective_cond else label_dim