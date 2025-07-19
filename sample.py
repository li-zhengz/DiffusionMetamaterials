import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import csv
import multiprocessing
from multiprocessing import Pool
import logging
import time
import os
import argparse
from torch.nn.functional import log_softmax
from functools import partial
from src.utils import *
from src.model import *
from dataGeneration.validity_check import *
from src.datasets import *
from src.gaussian_diffusion import *
from src.params import *

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Sample equations using diffusion model')
parser.add_argument('--cfg_scale', type=float, default=0.0, 
                    help='Classifier-free guidance scale (default: 0.0 for unconditional sampling)')
parser.add_argument('--num_samples', type=int, default=200,
                    help='Number of samples to generate (default: 200)')
parser.add_argument('--model_checkpoint', type=str, default='model_checkpoints/model_checkpoint.pth',
                    help='Path to model checkpoint (default: model_checkpoints/model_checkpoint.pth)')
args = parser.parse_args()

model_checkpoint_path = args.model_checkpoint
inv_target_file = f'data/inv_design_target/inv_target.csv'
train_dataset_path = f'data/dataset/train_dataset.pth'
outdir = f"generation_results"
if os.path.exists(outdir):
    os.system(f'rm -r {outdir}')
os.makedirs(outdir, exist_ok=True)
log_output_file = f'{outdir}/sample_output.txt'

num_samples = args.num_samples

sequence_length = seq_len
num_beams = 1
cfg_scale = args.cfg_scale

# Diffusion parameters
noise_scheduler_type = 'sqrt'
diffusion_steps = 2000
predict_xstart = True
sigma_small = False
training_mode = 'e2e'
rescale_timesteps = True
learn_sigmas = False
use_kl = False
rescale_learned_sigmas = False

# Load training dataset
dataset, label_normalizer = get_dataset(data_path, return_normalizer=True)

def setup_logger(log_file):
    logging.basicConfig(filename = log_file,
                        level = logging.INFO,
                        filemode = 'w',
                        format = '%(message)s')

def evaluate_validity_generated_samples(model_checkpoint_path, save_idx = None, num_samples = 100):
    start = time.time()

    setup_logger(log_output_file)
    logging.info(f"Processing model: {model_checkpoint_path}")
    
    tokenizer = EquationTokenizer() 

    target_c = np.genfromtxt(inv_target_file, delimiter=',').reshape(1,-1)
    target_c = np.tile(target_c, (num_samples, 1))
    target_c = torch.from_numpy(target_c).float().to(device)

    if len(target_c) < num_samples:
        num_samples = len(target_c)
    else:
        target_c = target_c[:num_samples,:]
    
    target_c = label_normalizer.normalize(target_c)

    train_dataset = torch.load(train_dataset_path)
    full_dataset = train_dataset.dataset
    subset_indices = train_dataset.indices

    # Get the tokens from the full dataset and index with the subset
    training_tokens = full_dataset.tensors[0][subset_indices]

    # Convert to NumPy
    training_tokens = training_tokens.cpu().numpy()
    
    if not os.path.exists(model_checkpoint_path):
        raise ValueError(f"Model checkpoint path {model_checkpoint_path} does not exist")
    else:
        # Fix module path for models saved before src reorganization
        import sys
        import src.model as model_module
        import src.transformer_utils as transformer_utils
        import src.params as params
        import src.gaussian_diffusion as gaussian_diffusion
        import src.utils as utils
        import src.datasets as datasets
        
        # Map old module names to new src locations
        sys.modules['model'] = model_module
        sys.modules['transformer_utils'] = transformer_utils  
        sys.modules['params'] = params
        sys.modules['gaussian_diffusion'] = gaussian_diffusion
        sys.modules['utils'] = utils
        sys.modules['datasets'] = datasets
        
        model = torch.load(model_checkpoint_path, map_location=torch.device('cpu')).to(device)
        model.eval()
        model.cfg_scale = cfg_scale

        valid_output_filename = f"{outdir}/valid_sample_equations.csv"
        invalid_output_filename = f"{outdir}/invalid_sample_equations.csv"
        
        # Initialize diffusion model
        betas = get_named_beta_schedule(noise_scheduler_type, diffusion_steps)
        diffusion = SpacedDiffusion(
            use_timesteps = space_timesteps(diffusion_steps, section_counts = 'ddim'+str(diffusion_steps)),
            betas = betas,
            rescale_timesteps = rescale_timesteps,
            training_mode = training_mode,
            predict_xstart = predict_xstart,
            learn_sigmas = learn_sigmas,
            use_kl = use_kl,
            rescale_learned_sigmas = rescale_learned_sigmas,
        )
        # Prepare to store intermediate results
        intermediate_results = []
        sample_shape = (num_samples, seq_len, model.word_embedding.weight.shape[-1])

        # Modify the sampling loop to return all intermediate samples
        all_samples = diffusion.p_sample_loop(model, sample_shape, cond=target_c, langevin_fn=None)
        # all_samples is a list or tensor of shape (timesteps, num_samples, seq_len, embed_dim)

        # Save intermediate results at desired time steps
        save_timesteps_percentage = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        save_timesteps = [round(len(all_samples) * p) for p in save_timesteps_percentage]
        save_timesteps[-1] = -1
        for t in save_timesteps:
            sample_t = all_samples[t]
            logits_t = model.get_logits(sample_t)
            log_probs_t = log_softmax(logits_t, dim=-1)
            cands_t = torch.topk(log_probs_t, k=num_beams, dim=-1)
            decoded_sentences_t = []
            for seq_list in cands_t.indices:
                for k in range(num_beams):
                    seq = seq_list[:, k]
                    _, decoded_sentence = tokenizer.decode(seq.tolist())
                    decoded_sentences_t.append(decoded_sentence)
            intermediate_results.append({
                'timestep': t,
                'decoded_sentences': decoded_sentences_t
            })
            # Optionally, save to file
            os.makedirs(f"{outdir}/intermediate_timestep_samples", exist_ok=True)
            with open(f"{outdir}/intermediate_timestep_samples/sample_timestep_{t}.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for eq in decoded_sentences_t:
                    writer.writerow([eq])

        sample = diffusion.p_sample_loop(model, sample_shape, cond=target_c, langevin_fn=None)
        sample = sample[-1] # extract the last timestep

        logits = model.get_logits(sample)
        cands = torch.topk(logits, k = 1, dim = -1)
        log_probs = log_softmax(logits, dim = -1)
        cands = torch.topk(log_probs, k = num_beams, dim = -1)

        decoded_sentences = []
        decoded_sequences = []

        for seq_list in cands.indices:
            for k in range(num_beams):
                seq = seq_list[:,k]
                decoded_sequence, decoded_sentence = tokenizer.decode(seq.tolist())
                decoded_sentences.append(decoded_sentence)
                decoded_sequences.append(decoded_sequence)
        
        sample = torch.repeat_interleave(sample, num_beams, dim = 0)
        target_c = torch.repeat_interleave(target_c, num_beams, dim = 0)

        valid_eq = []
        invalid_eq = []
        valid_eq_target_c = []

        for i in range(len(decoded_sentences)):
            eq = decoded_sentences[i].lstrip('+')
            try:
                is_dependent = is_equation_dependent_on_xyz(eq)
                is_valid = is_mesh_valid(eq)
            except:
                is_dependent = False
                is_valid = False

            if is_dependent and is_valid:
                valid_eq.append(eq)
                valid_eq_target_c.append(label_normalizer.unnormalize(target_c[i].cpu().detach().numpy()))
            else:
                invalid_eq.append(eq)
        
        with open(invalid_output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for eq in invalid_eq:
                writer.writerow([eq])
        if len(valid_eq) == 0:
            logging.info("*"*70)
            logging.info(f"Model loaded from {model_checkpoint_path}")   
            logging.info("No valid samples")
        else:
                
            generated_tokens = tokenizer.encode(valid_eq)

            unique_ratio = len(np.unique(generated_tokens, axis = 0))/len(generated_tokens)
            training_set = set(tuple(token) if isinstance(token, (list, np.ndarray)) else token for token in training_tokens)
            generated_set = set(tuple(token) if isinstance(token, list) else token for token in generated_tokens)

            novelty_score = len(generated_set.difference(training_set))/len(generated_tokens)
            novel_idx = [idx for idx, token in enumerate(generated_tokens) if tuple(token) not in training_set]

            unique_idx = np.unique(generated_tokens, axis = 0, return_index = True)[1]
            np.savetxt(f'{outdir}/valid_target_c.csv', np.array(valid_eq_target_c)[unique_idx,:], delimiter = ",")

            unique_valid_eq = np.array(valid_eq)[unique_idx]

            with open(valid_output_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for eq in unique_valid_eq:
                    writer.writerow([eq])
            
            end = time.time()
            
            logging.info("*"*70)
            logging.info(f"Model loaded from: {model_checkpoint_path}")   
            logging.info(f"Number of samples = {num_samples}")
            logging.info(f"Validity score = {len(valid_eq)/num_samples*100: .2f}%")
            logging.info(f"Unique score = {unique_ratio*100: .2f}%")
            logging.info(f"Novelty score = {novelty_score*100: .2f}%")
            logging.info(f"Generation time = {end - start: .2f} s")
            logging.info("*"*70)

evaluate_validity_generated_samples(model_checkpoint_path, num_samples = num_samples)