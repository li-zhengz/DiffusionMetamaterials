# Diffusion Model Training with YAML Configuration

This script automatically creates a Wandb project when you run the training script, with all configuration managed through a YAML file for easier parameter management.


## Usage

### Basic Usage (with default config.yaml)
```bash
python main.py
```

This will use the default `config.yaml` file and create a wandb project called "diffusion-training".

### Custom Configuration File
```bash
python main.py --config my_custom_config.yaml
```

## Key Configuration Sections

### Wandb Configuration
- `wandb.project`: Name of the wandb project
- `wandb.entity`: Your wandb username/entity

### Training Parameters
- `training.train_batch_size`: Batch size for training
- `training.num_epochs`: Number of training epochs
- `training.lr`: Learning rate
- `training.optimizer_type`: Optimizer type ['adam', 'nadam', 'adamw', 'sgd']
- `training.max_grad_norm`: Gradient clipping threshold
- `training.save_model_step`: How often to save model checkpoints

### Model Architecture
- `model.input_dims`: Input dimensions
- `model.output_dims`: Output dimensions
- `model.transformer_hidden_size`: Transformer hidden size
- `model.transformer_num_hidden_layers`: Number of transformer layers
- `model.transformer_num_attention_heads`: Number of attention heads

### Diffusion Parameters
- `diffusion.diffusion_steps`: Number of diffusion steps
- `diffusion.noise_schedule`: Noise schedule ['sqrt', 'cosine']
- `diffusion.section_counts`: Sampling method configuration

### Classifier-Free Guidance
- `cfg.cfg_scale`: Classifier-free guidance scale
- `cfg.cfg_dropout_prob`: CFG dropout probability

## Creating Custom Configurations

1. Copy the default `config.yaml` file
2. Modify the parameters you want to change
3. Save with a new name (e.g., `experiment_1.yaml`)
4. Run with: `python main.py --config experiment_1.yaml`

## Prerequisites

Make sure you have the required packages:

```bash
pip install wandb pyyaml
wandb login
```
