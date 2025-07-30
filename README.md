
<h1 align="center">DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials via Diffusion Transformers</h1>
<h4 align="center">
</h4>
<div align="center">
  <span class="author-block">
    <a>Li Zheng</a><sup>1</sup>,</span>
  <span class="author-block">
    <a>Siddhant Kumar</a><sup>2</sup>, and</span>
    <span class="author-block">
    <a>Dennis M. Kochmann</a><sup>1</sup></span>  
</div>
<div align="center">
  <span class="author-block"><sup>1</sup>ETH Zurich, </span>
  <span class="author-block"><sup>2</sup>TU Delft</span>
</div>

$~$
<p align="center"><img src="figures/DiffuMeta.png#gh-light-mode-only"\></p>

## 🎯 Overview

This project proposes a generative framework integrating diffusion transformers with a novel algebraic language representation, encoding 3D shell metamaterial geometries as mathematical sentences for inverse design with precisely targeted mechanical properties.

## ✨ Highlights

- **Novel Algebraic Language**: Introduces a unique mathematical representation that encodes 3D shell metamaterial geometries as interpretable algebraic sentences
- **Diffusion Transformers**: Leverages state-of-the-art diffusion models combined with transformer architectures for high-quality generation
- **Inverse Design Capability**: Enables precise targeting of specific mechanical properties for metamaterial design

<p align="center"><img src="figures/sequence_generation.png#gh-light-mode-only" width="600"\></p>


## 📁 Project Structure

```
.
├── main.py                    # Main training script
├── sample.py                  # Inference/sampling script
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── params.py                  # Global parameters and constants
│
├── src/                       # Core source code package
│   ├── __init__.py            # Package initialization
│   ├── params.py              # Global parameters and constants
│   ├── model.py               # Transformer model architecture
│   ├── gaussian_diffusion.py # Diffusion process implementation
│   ├── datasets.py            # Dataset loading and preprocessing
│   ├── utils.py               # Utility functions and tokenizer
│   └── transformer_utils.py  # Transformer-specific utilities
│
├── data/                      # Data directory
│   ├── dataset/               # Training datasets
│   ├── eq_vocab.pickle        # Equation vocabulary
│   └── inv_design_target/     # Target properties for inverse design
│
├── script/                    # Shell scripts for cluster execution
│   ├── train.sh               # Training job script
│   └── sample.sh              # Sampling job script
│
├── dataGeneration/            # Data generation utilities
│   ├── __init__.py            # Package initialization
│   ├── data_generator.py      # Generate equation datasets
│   └── validity_check.py      # Validate generated equations
│
├── model_checkpoints/         # Saved model checkpoints
└── runs/                      # Training outputs and checkpoints
```

## 🛠️ Installation

### Prerequisites

The framework was developed and tested on Python 3.11.6 using CUDA 12.0. The required dependencies can be installed by running:

```bash
pip install --user -r requirements.txt
```

## ⚙️ Configuration

The project uses YAML configuration files for easy parameter management. Edit `config.yaml` to customize:

### Key Configuration Sections

```yaml
# Wandb tracking
wandb:
  project: "diffusion-training"
  entity: "your-username"

# Training parameters
training:
  train_batch_size: 512
  num_epochs: 200001
  lr: 0.0001
  optimizer_type: "adamw"

# Model architecture
model:
  input_dims: 128
  output_dims: 256
  transformer_hidden_size: 512
  transformer_num_hidden_layers: 2

# Diffusion settings
diffusion:
  noise_schedule: "sqrt"
  diffusion_steps: 2000
  predict_xstart: true

# Classifier-free guidance
cfg:
  cfg_scale: 4.0
  cfg_dropout_prob: 0.1
```

### Global Parameters

Edit `params.py` to modify global settings:

- `num_datapoints`: Data points per equation (default: 40)
- `seq_len`: Maximum sequence length (default: 22)  
- `multi_objective_cond`: Enable multi-objective conditioning
- `multi_objective_cond_type`: Type of conditioning ('stiffness', 'moduli', 'poissons_ratio')

## 🚀 Usage
### Setup and Data Preparation

To conduct similar studies as those presented in the publication, start by cloning this repository via
```
https://github.com/li-zhengz/DiffuMeta.git
```

Next, download the data and model checkpoints provided in the ETHZ Research Collection. Unzip the dataset in the dataset folder and the pre-trained model in the model_checkpoint.zip, as shown in the project structure. 

### Training

```bash
python main.py --config config.yaml
```

### Inference/Sampling

#### Generate Equations with Default Settings
```bash
python sample.py
```

#### Custom Sampling Parameters
```bash
python sample.py \
  --cfg_scale 10.0 \
  --num_samples 200 \
  --model_checkpoint model_checkpoints/best_model_checkpoint.pth
```

### Monitoring Training

The project integrates with Weights & Biases for experiment tracking:

1. **View training progress:** Visit your wandb dashboard
2. **Monitor metrics:** Loss curves, validation metrics, embedding statistics
3. **Compare experiments:** Different configurations and hyperparameters

## 🧪 Data Generation

### Generate New Implicit Equations
```bash
python dataGeneration/data_generator.py
```

### Check The Validity of Generated Implicit Surfaces
```bash
python dataGeneration/validity_check.py
```

## 📁 Output Files

### Training Outputs
- `runs/{run_name}/best_model_checkpoint.pth`: Best model based on validation loss
- `runs/{run_name}/check_point/epoch_{N}_model_checkpoint.pth`: Periodic checkpoints
- `runs/{run_name}/train_dataset.pth`: Training dataset split

### Sampling Outputs  
- `generation_results/`: Generated equation samples
- `generation_results/sample_output.txt`: Sampling logs and statistics
- `generation_results/valid_samples.csv`: Successfully generated valid algebraic equations
- `generation_results/invalid_samples.csv`: Generated equations that failed validation (cannot be converted to connected shell geometries)


## 📧 Contact

For questions or support, please contact:
- **Author**: [Li Zheng](https://scholar.google.com/citations?user=dLCJjh4AAAAJ&hl=en)
- **Email**: li.zheng@mavt.ethz.ch
- **Institution**: [Mechanics and Materials Laboratory, ETH Zurich](https://mm.ethz.ch/)


