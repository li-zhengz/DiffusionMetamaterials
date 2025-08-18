# DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials

## Table of Contents

1. [Model Overview](#model-overview)
2. [Model Architecture](#model-architecture)
3. [Use](#use)
4. [Training Details](#training-details)
5. [Evaluation](#evaluation)
6. [Computational Requirements](#computational-requirements)
7. [Usage Examples](#usage-examples)
8. [Model Outputs](#model-outputs)
9. [Citation](#citation)
10. [Contact Information](#contact-information)
11. [Model Card Metadata](#model-card-metadata)

## Model Overview

**Model Name:** DiffuMeta  
**Model Version:** 1.0  
**Model Type:** Diffusion Transformer  
**License:** MIT License  
**Paper:** [DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials via Diffusion Transformers](https://arxiv.org/abs/2507.15753)  
**Repository:** https://github.com/li-zhengz/DiffusionMetamaterials  
**Institution:** Mechanics and Materials Laboratory, ETH Zurich  
**Authors:** Li Zheng and collaborators  

## Model Architecture

### Core Components
- **Transformer Architecture:**
  - Hidden size: 512 (configurable)
  - Number of layers: 2 (configurable)
  - Input dimensions: 128
  - Output dimensions: 256

- **Diffusion Process:**
  - Diffusion steps: 2000
  - Noise schedule: Square root
  - Prediction target: Direct prediction of `x_start`
  - Classifier-free guidance scale: 4.0 (configurable)

- **Sequence Parameters:**
  - Maximum sequence length: 22
  - Data points per equation: 40
  - Vocabulary: Algebraic equation vocabulary (stored in `eq_vocab.pickle`)

### Training Configuration
- **Optimizer:** AdamW
- **Learning Rate:** 0.0001
- **Batch Size:** 512
- **Training Epochs:** 200,001
- **Guidance Dropout:** 0.1

##  Use

Inverse design of 3D shell metamaterials with targeted mechanical properties, including:
- Stiffness properties
- Elastic moduli
- Poisson's ratio
- Multi-objective mechanical property optimization

## Training Details

### Training Data
- Custom-generated dataset of 3D shell metamaterial geometries
- Algebraic representations of metamaterial structures and their corresponding mechanical properties
- Dataset available through [ETHZ Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/746797)

### Data Preprocessing
- Conversion of 3D geometries to algebraic sentence representations
- Tokenization using custom vocabulary
- Property normalization for conditioning
- Sequence padding and truncation to a maximum length of 22 tokens

## Evaluation

### Testing Data
- Dataset available through [ETHZ Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/746797)

### Metrics
- **Generation Quality:** Validity of generated algebraic equations
- **Mechanical Property Accuracy:** Deviation from target mechanical properties
- **Geometric Validity:** Ability to convert generated equations to valid 3D shell geometries
- **Diversity:** Variation in generated metamaterial designs

## Computational Requirements

### Dependencies
- Python 3.11.6+
- PyTorch with CUDA 12.0 support
- Additional dependencies in `requirements.txt`

### Hardware Requirements
- CUDA-compatible GPU (recommended for training)
- Sufficient memory for transformer models and diffusion processes
- Storage for datasets and model checkpoints

## Usage Examples

### Basic Training
```bash
python main.py --config config.yaml
```

### Inference/Sampling
```bash
python sample.py --cfg_scale 10.0 --num_samples 200 --model_checkpoint model_checkpoints/best_model_checkpoint.pth
```

### Data Generation
```bash
python dataGeneration/data_generator.py
python dataGeneration/validity_check.py
```

## Model Outputs

### Generated Files
- **Valid Samples:** `generation_results/valid_samples.csv`
- **Invalid Samples:** `generation_results/invalid_samples.csv`
- **Sampling Logs:** `generation_results/sample_output.txt`
- **Model Checkpoints:** Best and periodic checkpoints saved during training

### Output Format
- Algebraic equations representing 3D shell metamaterial geometries
- Corresponding mechanical property predictions
- Validity status and conversion success rates

## Citation

If you use this model in your research, please cite:

```bibtex
@article{zheng2025diffumeta,
  title={DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials via Diffusion Transformers},
  author={Zheng, Li and [other authors]},
  journal={arXiv preprint arXiv:2507.15753},
  year={2025}
}
```

## Contact Information

**Primary Contact:** [Li Zheng](https://scholar.google.com/citations?user=dLCJjh4AAAAJ&hl=en)
**Email:** li.zheng@mavt.ethz.ch  
**Institution:** Mechanics and Materials Laboratory, ETH Zurich  

## Model Card Metadata

- **Model Card Version:** 1.0
- **Last Updated:** August 2025
- **Compliance:** Academic research use, subject to institutional guidelines

---

*This model card follows best practices for ML model documentation and transparency. For technical support or questions about the model, please refer to the repository issues page or contact the authors directly.*
