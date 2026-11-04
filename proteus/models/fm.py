# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import copy
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

from proteus.utils.timm.registry import register_model
from huggingface_hub import PyTorchModelHubMixin

from .fm_utils import Block, DecoderBlock, LayerNorm, MutationRegressor
from .fusion import create_fusion_module
from proteus.data.modality_info import MODALITY_INFO


# Model definitions
__all__ = [
    # GELU models
    'fm_tiny_6e_6d_gelu',
    'fm_small_8e_8d_gelu',
    'fm_base_12e_12d_gelu',
    'fm_large_24e_24d_gelu',
    'fm_xlarge_24e_24d_gelu',
    # SwiGLU models
    'fm_tiny_6e_6d_swiglu_nobias',
    'fm_small_8e_8d_swiglu_nobias',
    'fm_base_12e_12d_swiglu_nobias',
    'fm_large_24e_24d_swiglu_nobias',
    'fm_xlarge_24e_24d_swiglu_nobias',
    # SwiGLU + QKNorm models
    'fm_base_12e_12d_swiglu_qknorm_nobias',
    'fm_large_24e_24d_swiglu_qknorm_nobias',
    'fm_xlarge_24e_24d_swiglu_qknorm_nobias',
]



class FourM(nn.Module):
    """4M model.

    Args:
        encoder_embeddings: Dict of encoder embedding modules.
        decoder_embeddings: Dict of decoder embedding modules.
        modality_info: Dict containing modality information.
        dim: Embedding dimension.
        encoder_depth: Number of encoder blocks.
        decoder_depth: Number of decoder blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        proj_bias: If True, add a learnable bias to the last projection of the attention block.
        mlp_bias: If True, add a learnable bias to linear layers in the MLP / feed-forward.
        drop_path_rate_encoder: Stochastic depth rate for encoder.
        drop_path_rate_decoder: Stochastic depth rate for decoder.
        shared_drop_path: If True, shares drop path between encoder and decoder.
        act_layer: Activation layer to be used.
        norm_layer: Normalization layer to be used.
        gated_mlp: If True, make the feedforward gated (e.g., SwiGLU).
        qk_norm: If True, applies normalization to queries and keys (QKNorm).
        decoder_causal_mask: If True, decoder will use a causal mask for all tokens.
        decoder_sep_mask: If True, decoder attention is restricted to within each modality only.
        num_register_tokens: Number of register tokens.
        use_act_checkpoint: If True, use activation checkpoint for each block.
    """
    def __init__(self,
                 encoder_embeddings: Dict[str, nn.Module],
                 decoder_embeddings: Dict[str, nn.Module],
                 modality_info: Dict[str, Any],
                 dim: int = 768,
                 encoder_depth: int = 12,
                 decoder_depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 mlp_bias: bool = True,
                 drop_path_rate_encoder: float = 0.0,
                 drop_path_rate_decoder: float = 0.0,
                 shared_drop_path: bool = False,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: Union[partial, nn.Module] = partial(LayerNorm, eps=1e-6),
                 gated_mlp: bool = False, # Make the feedforward gated for e.g. SwiGLU
                 qk_norm: bool = False,
                 decoder_causal_mask: bool = False,
                 decoder_sep_mask: bool = True,
                 num_register_tokens: int = 0,
                 use_act_checkpoint: bool = False,
                 share_modality_embeddings: bool = True,
                 fusion_method: str = 'add',
                 ):
        super().__init__()

        self.modality_info = modality_info
        self.dim = dim
        self.init_std = 0.02
        self.use_act_checkpoint = use_act_checkpoint
        self.num_register_tokens = num_register_tokens
        self.fusion_method = fusion_method

        # Initialize fusion module for multi-modal encoder tokens
        num_encoder_modalities = len(encoder_embeddings)
        self.fusion_module = create_fusion_module(fusion_method, dim, num_encoder_modalities)

        # Encoder embeddings & init
        self.encoder_modalities = set(encoder_embeddings.keys())
        for emb in encoder_embeddings.values():
            emb.init(dim_tokens=dim, init_std=self.init_std)
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)
        
        self.regression_head = MutationRegressor(dim)

        ## Transformer encoder
        if shared_drop_path:
            dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth + decoder_depth)][:encoder_depth]
        else:
            dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth)] # stochastic depth decay rule

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, mlp_bias=mlp_bias,
                 drop_path=dpr_encoder[i], act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm, mutation_bias_value=10.0)
            for i in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=self.init_std)

        # Additional register tokens that can be used by the encoder during fine-tuning
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
            nn.init.normal_(self.register_tokens, std=self.init_std)
        else:
            self.register_tokens = None
        
        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            elif "pretrained_model" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_num_layers_encoder(self):
        return len(self.encoder)

    def get_num_layers(self):
        return self.get_num_layers_encoder() + self.get_num_layers_decoder()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip


        return no_wd_set

    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Aggregate encoder tokens and embeddings across modalities.

        Args:
            mod_dict: mapping modality name → dict with keys:
                - 'x'           Tensor of shape (B, L, D): modality tokens
                - 'emb'         Tensor of shape (B, L, D): modality embeddings
                - 'input_mask'  BoolTensor of shape (B, L): 0=valid, 1=masked

        Returns:
            encoder_tokens_all: Tensor[B, L, D] fused modality tokens (method depends on fusion_method)
            emb_all:            Tensor[B, L, D] sum of all modality embeddings
            encoder_mask_all:   BoolTensor[B, L] zeros (all positions valid)
            mod_mask_all:       LongTensor[B, L] zeros (placeholder)
        """
        # 1) collect per-modality tensors in sorted order for consistency
        modality_names = sorted(mod_dict.keys())
        token_list = [mod_dict[name]['x'] for name in modality_names]
        emb_list = [mod_dict[name]['emb'] for name in modality_names]

        # 2) fuse tokens across modalities
        if self.fusion_module is not None:
            # Use the configured fusion method
            encoder_tokens_all = self.fusion_module(token_list)
        else:
            # Default: simple element-wise addition (backward compatible)
            encoder_tokens_all = token_list[0].clone()
            for t in token_list[1:]:
                encoder_tokens_all = encoder_tokens_all + t

        # 3) embeddings: always use simple sum (no learnable fusion)
        emb_all = emb_list[0].clone()
        for e in emb_list[1:]:
            emb_all = emb_all + e

        # 4) build placeholder masks
        B, L, D = emb_all.shape
        device = emb_all.device

        seq_input_mask = mod_dict['sequence']['input_mask']   # BoolTensor[B, 514]

        encoder_mask_all = seq_input_mask[:, 1:-1]            # BoolTensor[B, 512]
        mod_mask_all = torch.zeros((B, L), dtype=torch.long, device=device)

        return encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all

    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_encoder_tokens: int) -> Tuple[torch.Tensor]:
        """Concatenates and mask encoder tensors based on provided modality information.

        This function consolidates encoder tokens from multiple modalities, then selects a specified number of them based on modality information (i.e. masking).

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities. 
                            It is expected to have keys for each modality and values 
                            containing the modalities' associated tensors.
            num_encoder_tokens (int): Number of encoder tokens to retain after masking.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): Selected encoder tokens from all modalities. Shape (B, N, D) where N is the number of selected encoder tokens. 
                - encoder_emb (torch.Tensor): Corresponding embeddings for encoder tokens. Shape (B, N, D)
                - encoder_mask (torch.Tensor): A boolean mask indicating which encoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each encoder token (with -1 indicating unassigned pad tokens). Shape (B, N)

        Notes:
            - If `num_register_tokens` is set and greater than 0, register tokens are added at the beginning of the sequence.
        """
        B = list(mod_dict.values())[0]['tensor'].shape[0]

        encoder_tokens, encoder_emb, encoder_mask, mod_mask = self.cat_encoder_tensors(mod_dict)

        if self.num_register_tokens > 0:
            register_tokens = repeat(self.register_tokens, '() n d -> b n d', b=B)
            # We add register tokens at the beginning of the sequence
            encoder_tokens = torch.cat([register_tokens, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(register_tokens), encoder_emb], dim=1)
            encoder_mask = torch.cat([torch.zeros((B, register_tokens.shape[1]), dtype=torch.bool, device=encoder_mask.device), encoder_mask], dim=1)
            mod_mask = torch.cat([torch.full((B, register_tokens.shape[1]), -1, dtype=torch.int16, device=mod_mask.device), mod_mask], dim=1)
            mut_mask = mod_dict['sequence']['mutation_mask']
            zero_col = torch.zeros(B, 1, dtype=mut_mask.dtype, device=mut_mask.device)
            mut_mask = torch.cat([zero_col, mut_mask], dim=1)   # shape [B, 513]
            mod_dict['sequence']['mutation_mask'] = mut_mask

        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.
        mod_mask[encoder_mask] = -1
        # Mask could be of shape 'b n1 n2' but not needed for masked_fill
        # This means this mask can then be re-used for decoder cross-attention
        encoder_mask = rearrange(encoder_mask, 'b n2 -> b 1 n2')

        return encoder_tokens, encoder_emb, encoder_mask, mod_mask

    def forward_encoder(self, 
                        x: torch.Tensor, 
                        encoder_mask: torch.Tensor,
                        mut_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Encoder input tokens. Shape (B, N, D) where N is the number of encoder tokens.
            encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
            
        Returns:
            torch.Tensor: Encoder output. Shape (B, N, D)
        """

        for blk in self.encoder:
            x = blk(x, mask=encoder_mask, mut_mask=mut_mask)
            
        x = self.encoder_norm(x)

        return x

    def forward_logits(self, 
                       y: torch.Tensor, 
                       decoder_mod_dict: Dict[str, Dict[str, torch.Tensor]], 
                       decoder_mod_mask: torch.Tensor,
                       return_all_logits: bool = False) -> Dict[str, torch.Tensor]:
        """Forward computation of logits for each modality.

        Args:
            y (torch.Tensor): Decoder output. Shape (B, M, D).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Integer mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of logits for each modality.
        """

        mod_logits = {}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            if return_all_logits:
                logits = self.decoder_embeddings[mod].forward_logits(y)
            else:
                logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            mod_logits[mod] = logits
        return mod_logits

    def forward_loss(self, 
                     score_pred: torch.Tensor,
                     score_label: torch.Tensor,
                     loss_type: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the loss based on the specified loss type.

        Args:
            y (torch.Tensor): Decoder output. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Integer mask indicating which tokens belong to which modality. Shape (B, M).
            loss_type (str): The type of loss to compute. Either 'mod' or 'token'.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and dictionary of loss for each modality.
        """
        loss_dict = {}
        if loss_type == 'mse':
            loss = self.forward_mutation_loss(score_pred, score_label)
        else:
            raise ValueError("Invalid loss type")

        loss_dict = {
            'loss': loss,
        }
        return loss_dict

    def forward_mod_loss(self, 
                         y: torch.Tensor, 
                         target_ids: torch.Tensor, 
                         decoder_mod_dict: Dict[str, Any], 
                         decoder_mod_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the modality-wise loss.

        Args:
            y (torch.Tensor): Decoder tokens. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total modality loss and dictionary of loss for each modality.
        """       
        mod_loss = {}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])

            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=logits.device)
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss

        loss = sum(mod_loss.values()) / len(mod_loss)

        return loss, mod_loss
    
    def forward_multitask_loss(self, 
                        y: torch.Tensor, 
                        target_ids: torch.Tensor, 
                        decoder_mod_dict: Dict[str, Any], 
                        decoder_mod_mask: torch.Tensor,
                        score_pred: torch.Tensor,
                        score_label: torch.Tensor,
                        ) -> Tuple[torch.Tensor, float, float, Dict[str, torch.Tensor], Dict[str, float]]:
        """Computes the automatically weighted multi-task loss with logging of scaled components."""
        
        mod_loss = {}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if logits.numel() == 0:
                mod_loss[mod] = torch.zeros(1, device=logits.device)
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss

        generation_loss = sum(mod_loss.values()) / len(mod_loss)

        score_label = score_label[:, 0].float()  
        regression_loss = nn.MSELoss()(score_pred, score_label)
        regression_loss = regression_loss* 100  # Scale regression loss by 100
        
        total_loss = generation_loss + regression_loss
     
        return total_loss, generation_loss, regression_loss, mod_loss


    def forward_token_loss(self, 
                           y: torch.Tensor, 
                           target_ids: torch.Tensor, 
                           decoder_mod_dict: Dict[str, Any], 
                           decoder_mod_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the token-wise loss.

        Args:
            y (torch.Tensor): Decoder tokens. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total token loss and dictionary of loss for each modality.
        """        
        mod_loss = {}
        mod_count = {}

        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=logits.device)
                mod_count[mod] = 0
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss
                mod_count[mod] = logits.numel()

        loss = sum([mod_loss[mod] * mod_count[mod] for mod in mod_loss.keys()]) / sum(mod_count.values())

        return loss, mod_loss
    

    def forward_mutation_loss(self, 
                        score_pred: torch.Tensor,
                        score_label: torch.Tensor,
                        ) -> Tuple[torch.Tensor, float, float, Dict[str, torch.Tensor], Dict[str, float]]:
        
        score_label = score_label[:, 0].float()  
        regression_loss = nn.MSELoss()(score_pred, score_label)

        return regression_loss
    

    def forward(self, 
            mod_dict: Dict[str, Dict[str, torch.Tensor]], 
            num_encoder_tokens: int, 
            num_decoder_tokens: int, 
            loss_type: str = 'mod', 
            return_logits: bool = False) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for the model.

        Args:
            mod_dict (Dict[str, Dict[str, torch.Tensor]]): Dictionary containing the tensors, masks, and other info for each modality.
                - mod_dict[modality_name]["tensor_name"]: Shape can vary based on tensor_name and modality.
            num_encoder_tokens (int): Number of tokens to keep for the encoder.
            num_decoder_tokens (int): Number of tokens to keep for the decoder.
            loss_type (str, optional): The type of loss to compute. Can be 'mod' (average of loss per modality) or 'token' (average loss per token). Default is 'mod'.
            return_logits (bool, optional): If True, return the logits. Default is False.

        Returns:
            Union[dict, tuple]: 
                - If return_logits is True: Dictionary of logits for each modality.
                - Otherwise: Tuple containing the total loss and dictionary of loss for each modality.
        """

        # Mod dicts
        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        if loss_type == 'multi_task' or loss_type == 'mse':
            score_data = mod_dict.get("scores") or mod_dict.get("scores_normalized") or mod_dict.get("scores_normalized_5fold")
            if score_data is None:
                raise KeyError("Neither 'scores' nor 'scores_normalized' found in mod_dict")
            score_label = score_data["tensor"]
        else:
            score_label = None
        
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)


        # Encoder
        x = encoder_tokens + encoder_emb

        x = self.forward_encoder(x, encoder_mask=encoder_mask, mut_mask=encoder_mod_dict['sequence']['mutation_mask'])
        score_pred = self.regression_head(x)

        
        # Loss
        loss = self.forward_loss(score_pred, score_label, loss_type)

        # Logits
        if return_logits:
            return (score_pred, score_label), loss
        else:
            return loss


    def freeze_encoder(self, freeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_encoder_except_specific_embeddings(self, frozen_embedding_domain):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.encoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.encoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_decoder(self, freeze_embeddings=True):
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_decoder_except_specific_embeddings(self, frozen_embedding_domain):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.decoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def freeze_shared_params(self):
        self.freeze_encoder(freeze_embeddings=False)

    def freeze_params_except_specific_embeddings(self, frozen_embedding_domain):
        self.freeze_encoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)
        self.freeze_decoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)

    def unfreeze_shared_params(self):
        self.unfreeze_encoder(unfreeze_embeddings=False)

    def unfreeze_all(self):
        self.unfreeze_encoder(unfreeze_embeddings=True)


################################################

# Wrapper for easy loading with Huggingface Hub

class FM(FourM, PyTorchModelHubMixin):
    """Wrapper around FourM for easy loading with Huggingface Hub.

    Args:
        config (dict): Dictionary containing the model and modality configuration, 
            used for loading from Huggingface Hub.
    """
    def __init__(self, config: dict):

        config = copy.deepcopy(config)

        all_domains = sorted(list(set(config['domains_in']) | set(config['domains_out'])))
        modality_info = {mod: MODALITY_INFO[mod] for mod in all_domains}

        encoder_embeddings = {}
        for mod in config['domains_in']:
            info = modality_info[mod]
            if info.get("encoder_embedding", None) is not None:
                if info["type"] == "img":
                    image_size, patch_size = info.get('input_size', config['image_size']), info.get('patch_size', config['patch_size'])
                    encoder_embeddings[mod] = info["encoder_embedding"](patch_size=patch_size, image_size=image_size)
                else:
                    encoder_embeddings[mod] = info["encoder_embedding"]()
    
        decoder_embeddings = {}
        for mod in config['domains_out']:
            info = modality_info[mod]
            if info.get("decoder_embedding", None) is not None:
                if info["type"] == "img":
                    image_size, patch_size = info.get('input_size', config['image_size']), info.get('patch_size', config['patch_size'])
                    decoder_embeddings[mod] = info["decoder_embedding"](patch_size=patch_size, image_size=image_size, share_embedding=False)
                else:
                    decoder_embeddings[mod] = info["decoder_embedding"](share_embedding=False)

        config['norm_layer'] = partial(LayerNorm, eps=1e-6, bias=config['norm_bias'])
        config['act_layer'] = getattr(torch.nn, config['act_layer'])

        del config['norm_bias']
        del config['domains_in']
        del config['domains_out']
        del config['image_size']
        del config['patch_size']
        
        super().__init__(
            encoder_embeddings=encoder_embeddings,
            decoder_embeddings=decoder_embeddings,
            modality_info=modality_info,
            **config
        )   


################################################

# Model definitions
        
# GELU variants
@register_model
def fm_tiny_6e_6d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=6,
        decoder_depth=6,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_small_8e_8d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=8,
        decoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_base_12e_12d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_large_24e_24d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def fm_xlarge_24e_24d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# SwiGLU variants
@register_model
def fm_tiny_6e_6d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=6,
        decoder_depth=6,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model


@register_model
def fm_small_8e_8d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=8,
        decoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model


@register_model
def fm_base_12e_12d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model

@register_model
def fm_large_24e_24d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model

@register_model
def fm_xlarge_24e_24d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model

# SwiGLU + QKNorm variants


@register_model
def fm_base_12e_12d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model


@register_model
def fm_large_24e_24d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model

@register_model
def fm_xlarge_24e_24d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model
