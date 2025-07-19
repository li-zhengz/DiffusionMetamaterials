import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import einops
import numpy as np
import math
import datasets
from transformers import AutoConfig
try:
    # Try relative imports first (when imported as a package)
    from .transformer_utils import *
    from .params import *
except ImportError:
    # Fall back to absolute imports (when running directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.transformer_utils import *
    from src.params import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        self.null_token = nn.Parameter(torch.randn(1, hidden_size))
        self.hidden_size = hidden_size

    def forward(self, labels, dropout_prob, force_drop_ids = None):
        use_dropout = dropout_prob > 0

        if use_dropout or force_drop_ids is not None:
            
            label_embed = torch.zeros((labels.shape[0], self.hidden_size))

            batch_drop_mask = torch.rand(labels.shape[0], device=labels.device) < dropout_prob  # Shape: (batch_size,)

            # Apply the MLP transformation to all labels
            label_transformed = self.mlp(labels)  # Shape: (batch_size, hidden_size)
            # Use batch_drop_mask to either assign transformed labels or the null_token
            label_embed = torch.where(
            batch_drop_mask,            # Mask expanded to (batch_size, 1, 1)
            self.null_token,                      # Assign null_token where mask is True
            label_transformed                          # Otherwise, assign transformed labels
            )

        else:
            label_embed = self.mlp(labels)

        return label_embed

class AttentionLabelEmbedder(nn.Module):
    def __init__(self, label_dim, hidden_size):
        super().__init__()

        self.mlp = nn.Sequential(
            # nn.Linear(1, hidden_size, bias=False)
            nn.Linear(label_dim, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

        self.null_token = nn.Parameter(torch.randn(1, hidden_size))
        self.hidden_size = hidden_size

    
    def forward(self, labels, dropout_prob, force_drop_ids = None):
        use_dropout = dropout_prob > 0

        if use_dropout or force_drop_ids is not None:
            
            label_embed = torch.zeros((labels.shape[0], self.hidden_size))

            batch_drop_mask = torch.rand(labels.shape[0], device=labels.device) < dropout_prob  # Shape: (batch_size,)

            # Apply the MLP transformation to all labels
            label_transformed = self.mlp(labels)  # Shape: (batch_size, label_dim, hidden_size)
            # Use batch_drop_mask to either assign transformed labels or the null_token
            label_embed = torch.where(
            batch_drop_mask.view(-1,1),                    # Mask expanded to (batch_size, 1, 1)
            self.null_token,                      # Assign null_token where mask is True
            label_transformed                          # Otherwise, assign transformed labels
            )

        else:
            label_embed = self.mlp(labels)

        return label_embed

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x.transpose(0,1))

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio = 2, cross_attn = 0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)

        self.attn = Attention(hidden_size, num_heads, qkv_bias=False, qk_norm = True, **block_kwargs)

        if cross_attn > 0:
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
            self.cross_attn = CrossAttention(hidden_size, cross_attn, num_heads, qkv_bias=False, **block_kwargs)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        # self.factor = 9 if cross_attn > 0 else 6
        self.factor = 6
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * self.factor, bias=False),
        )

    def forward(self, x, c, y = None, pad_mask = None):

        if y is not None:
            y = y.reshape(x.shape[0], -1, x.shape[2]).to(x.dtype)
            x = torch.cat((y, x), dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x.float()).to(x.dtype), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp( modulate(self.norm2(x.float()).to(x.dtype), shift_mlp, scale_mlp))
        if y is not None:
            x = x[:,y.shape[1]:,:]
        
        return x

class FinalLayer(nn.Module):

    def __init__(self, hidden_size, seq_length):
    
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
        self.linear = nn.Linear(hidden_size, seq_length, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
        )

    def forward(self, x, c, y = None):
        if y is not None:
            y = y.reshape(x.shape[0], -1, x.shape[2]).to(x.dtype)
            x = torch.cat((y, x), dim = 1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim = -1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        if y is not None:
            x = x[:,y.shape[1]:,:]
        return x

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        transformer_num_hidden_layers,
        transformer_num_attention_heads,
        transformer_hidden_size,
        proj_activation_func,
        mlp_ratio,
        depth,
        cfg_scale,
        dropout=0.1,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        logits_mode=1,
        dropout_prob=0.1,
        cross_attn = 0,
        latent_model=None,
        embedding_scale=1.,
        learn_embedding_scale=False,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.is_decoder = True
            config.add_cross_attention = True
            config.hidden_dropout_prob = dropout
            config.num_hidden_layers = transformer_num_hidden_layers
            config.vocab_size = vocab_size
            config.num_attention_heads = transformer_num_attention_heads
            config.hidden_size = transformer_hidden_size

        self.input_dims = input_dims
        self.output_dims = self.input_dims
        self.hidden_t_dim = hidden_t_dim
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.cfg_scale = cfg_scale
        self.cfg_drop_prob = dropout_prob
        if cross_attn:
            self.cross_attn = config.hidden_size
        else:
            self.cross_attn = 0
        self.latent_model = latent_model

        pad_idx = 0
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims, padding_idx = pad_idx)
        if learn_embedding_scale:
            self.embedding_scale = nn.Parameter(torch.tensor(3.0))
        else:
            self.embedding_scale = embedding_scale

        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight 


        self.label_embed = AttentionLabelEmbedder(label_dim, config.hidden_size)
        self.multi_label_embed = AttentionLabelEmbedder(multi_label_dim, config.hidden_size)

        self.multi_cond_embed = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.t_embedder = TimestepEmbedder(config.hidden_size)

        if self.input_dims != config.hidden_size:
            if proj_activation_func == 'tanh':
                self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
            elif proj_activation_func == 'relu':
                self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.ReLU(), nn.Linear(config.hidden_size, config.hidden_size))
        
    
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.positional_embedding = PositionalEncoding(config.hidden_size, max_len=seq_len)

        self.blocks = nn.ModuleList([DiTBlock(config.hidden_size, config.num_attention_heads, mlp_ratio = mlp_ratio, cross_attn=self.cross_attn) for _ in range(depth)])

        self.final_layer = FinalLayer(config.hidden_size, self.output_dims)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if self.output_dims != config.hidden_size:
            if proj_activation_func == 'tanh':
                self.output_down_proj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Tanh(), 
                    nn.Linear(config.hidden_size, self.output_dims)
                    )
            elif proj_activation_func == 'relu':
                self.output_down_proj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(), 
                    nn.Linear(config.hidden_size, self.output_dims)
                    )
                
    def freeze_embedding(self):
        self.word_embedding.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Embedding):
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            if block.adaLN_modulation[-1].bias is not None:
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        if self.final_layer.adaLN_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_embeds(self, input_ids, mask = None):
        latent = self.word_embedding(input_ids)
        return latent * self.embedding_scale

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr) * self.embedding_scale
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, cond = None, mask = None, null_cond_prob = 0.1):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.t_embedder(timesteps)

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x
    
        emb_x = self.positional_embedding(emb_x)

        if cond is not None:
            if null_cond_prob != 0. and null_cond_prob != 1.:
                null_cond_prob = self.cfg_drop_prob

            if multi_objective_cond:
                cond_1, cond_2 = cond[0], cond[1]
                emb_y = self.label_embed(cond_1, null_cond_prob)
                emb_c = self.multi_label_embed(cond_2, null_cond_prob)

                label_emb = torch.cat([emb_y, emb_c], dim = 1)
                label_emb = self.multi_cond_embed(label_emb)
            else:
                emb_y = self.label_embed(cond, null_cond_prob)
                label_emb = emb_y

        else:
            # label_emb_seq = None
            label_emb = None
        
        # c = emb_t + emb_y if self.cross_attn == 0 else emb_t
        c = emb_t if cond is None else emb_t + emb_y

        emb_x = self.dropout(self.layer_norm(emb_x)) # shape: (N, seq_len, hidden_dim)
        # label_emb shape: (N, hidden_dim)
        # emb_x = self.dropout(emb_x)

        for block in self.blocks:
            x_out = block(emb_x, c, label_emb, pad_mask = mask)
        x_out = self.final_layer(x_out, c, label_emb)
        
        return x_out


    def forward_with_cfg(self, *args, **kwargs):

        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        # return logits

        return null_logits + (logits - null_logits) * self.cfg_scale
        