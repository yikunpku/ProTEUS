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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
# ESM2 / ESM-MSA-1b live in the separate `fair_esm` namespace in this env so
# the two packages don't clash on sys.modules['esm']. Do NOT switch this to
# `torch.hub.load("facebookresearch/esm:main", ...)` — hubconf.py does
# `from esm.pretrained import esm1_*` which then resolves to esm v3.x and
# explodes.
import fair_esm.pretrained as _fair_esm_pretrained
from huggingface_hub import login
from .fm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair

# For anyone tempted to put `ssl._create_default_https_context =
# ssl._create_unverified_context` here to force a weight download through: at
# module scope that disables certificate verification process-wide, for every
# stdlib HTTPS client (urllib, http.client, torch.hub, smtplib, ...) in
# anything that imports this package. A CERTIFICATE_VERIFY_FAILED here is the
# environment's trust store; `proteus-doctor` prints how to fix it.


class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens,
                                     padding_idx=self.padding_idx)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        ids = d['tensor']
        B = ids.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d['input_mask']
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
    
class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 vocab_size: int,
                 patch_size: Union[int, Tuple[int,int]] = 16,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224,
                 **kwargs):

        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image tokens for each batch. Shape (B, H, W) where B is the batch size, and H, W are height and width of the tokenized image.                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dictionary with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
        """
        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb

        return d


class ImageEncoderEmbedding(nn.Module):
    """Embedding module for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    This adapter / embedding differs from the one of MultiMAE by taking as input a dict and
     separating positional embeddings and modality embeddings from the input projection
     Input projection is 'x', posemb + modemb is 'emb'

    Args:
        num_channels: Number of input channels of the image/feature map
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 patch_size: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Image -> tokens projection
        # No bias term here, so modality embedding fully comes from self.mod_emb
        self.proj = nn.Linear(self.num_channels * self.patch_size[0] * self.patch_size[1], self.dim_tokens, bias=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image to sequence of tokens.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image for each batch. Shape (B, C, H, W) where B is the batch size, C is the number of channels, and H, W are height and width of the image.

                
        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, (H / PH) * (W / PW), D), where PH and PW are the patch sizes
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, (H / PH) * (W / PW), D)
        """
        x = d['tensor']
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.patch_size[0] == 0) and (W % self.patch_size[1] == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.patch_size[0]}x{self.patch_size[1]}'

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = self.proj(rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)', ph=self.patch_size[0], pw=self.patch_size[1]))

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x_patch
        d['emb'] = x_emb

        return d


class SequenceEmbEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 ):
        super().__init__()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        orig_emb = d['tensor']
        orig_emb = orig_emb.squeeze(1) if orig_emb.dim() == 4 else orig_emb
        B = orig_emb.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        # Map to embedding
        x = self.emb_proj(orig_emb)

        x_emb = self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
    
class MSAEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 ):
        super().__init__()
        # Use fair_esm namespace directly (esm 3.x is already in sys.modules
        # from the ESM3 import above; torch.hub.load would hit a broken
        # hubconf.py).
        self.msa_transformer, _ = _fair_esm_pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_transformer.eval()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        msa_token = d['tensor'].squeeze(1) if d['tensor'].dim() == 4 else d['tensor']  # (B, N_msa, L)
        with torch.no_grad():
            msa_embeddding = self.msa_transformer(msa_token, repr_layers=[12], return_contacts=False)["representations"][12][:,0,1:,:]  # Exclude the first token (BOS)
        
        B = msa_embeddding.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        # Map to embedding
        x = self.emb_proj(msa_embeddding)
        B, L, D = x.shape  
        mod_emb_expanded = self.mod_emb.expand(B, L, D)
        x_emb = mod_emb_expanded

        d['x'] = x
        d['emb'] = x_emb
        return d
    
class MSAEmbEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 ):
        super().__init__()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        msa_embeddding = d['tensor']
       
        B = msa_embeddding.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        # Map to embedding
        x = self.emb_proj(msa_embeddding)
        B, L, D = x.shape  
        mod_emb_expanded = self.mod_emb.expand(B, L, D)
        x_emb = mod_emb_expanded

        d['x'] = x
        d['emb'] = x_emb
        return d
    
    
class ProteinSequenceEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 base_model: str = "esm1v_t33_650M_UR90S_1"
                 ):
        super().__init__()
        self.base_model = base_model
        # ESMC (EvolutionaryScale `esm` pkg) is loaded differently from ESM2
        # (fair_esm). Stored sequence tokens are ESM2 alphabet ids; ESMC shares
        # the same 20-AA vocab ordering, so we remap ids via a small LUT
        # (validated: remapped ids == ESMC-native tokens, emb max|Δ|=0).
        self.is_esmc = str(base_model).lower().startswith('esmc')
        # DPLM is a discrete-diffusion LM trained from an ESM2 checkpoint, so it
        # keeps ESM2's architecture AND its alphabet - measured id for id against
        # fair_esm's ESM-1b alphabet, 0 differences - which is why the stored
        # sequence tokens feed it unchanged and no LUT is needed here. It loads
        # with plain transformers (config says model_type 'esm'); verified
        # missing_keys=0, unexpected_keys=0, so nothing is left at its random init.
        # Diffusion needs no timestep argument: the noise level of a discrete
        # diffusion LM is how much of the input is <mask>, and a clean sequence is
        # t=0, which is the deterministic representation a frozen encoder wants.
        self.is_dplm = str(base_model).lower().startswith('dplm')
        if self.is_dplm:
            from transformers import EsmForMaskedLM
            _repo = {'dplm_150m': 'airkingbd/dplm_150m',
                     'dplm_650m': 'airkingbd/dplm_650m',
                     'dplm_3b': 'airkingbd/dplm_3b'}.get(str(base_model).lower(),
                                                         str(base_model))
            self.pretrained_model = EsmForMaskedLM.from_pretrained(_repo)
            self.pretrained_model.eval()
        elif self.is_esmc:
            from esm.models.esmc import ESMC
            _bm = str(base_model).lower()
            if _bm in ('esmc_6b', 'esmc-6b', 'esmc6b'):
                # ESMC-6B: no official local entrypoint (esm pkg routes 6B via
                # forge API). Load biohub's HF-format weights into esm's native
                # ESMC(2560d/40h/80L) via a state_dict remap (TE-fused ->
                # esm's split layout). Validated: cos 0.9996 vs reference emb.
                import glob as _glob
                from safetensors.torch import load_file as _lf
                from esm.tokenization import get_esmc_model_tokenizers as _get_tok
                # Same cache location as before, resolved through the HF
                # environment instead of a literal home directory so this works
                # on a machine other than the one it was written on.
                import os as _os
                _hub = _os.environ.get(
                    'HF_HUB_CACHE',
                    _os.path.join(_os.environ.get(
                        'HF_HOME', _os.path.expanduser('~/.cache/huggingface')),
                        'hub'))
                _cands = _glob.glob(_os.path.join(
                    _hub, 'models--biohub--ESMC-6B', 'snapshots', '*') + _os.sep)
                if not _cands:
                    raise FileNotFoundError(
                        f"ESMC-6B snapshot not found under {_hub}. Download "
                        f"biohub/ESMC-6B first, or set HF_HOME/HF_HUB_CACHE.")
                _snap = _cands[0]
                _sd = {}
                for _f in sorted(_glob.glob(_snap + 'model-*.safetensors')):
                    _sd.update(_lf(_f))

                def _rm6b(k):
                    if k.endswith('_extra_state'):
                        return None
                    k = k.replace('esmc.', '', 1) if k.startswith('esmc.') else k
                    k = k.replace('lm_head.', 'sequence_head.')
                    k = k.replace('.attn.layernorm_qkv.layer_norm_weight', '.attn.layernorm_qkv.0.weight')
                    k = k.replace('.attn.layernorm_qkv.layer_norm_bias', '.attn.layernorm_qkv.0.bias')
                    k = k.replace('.attn.layernorm_qkv.weight', '.attn.layernorm_qkv.1.weight')
                    k = k.replace('.ffn.layer_norm_weight', '.ffn.0.weight')
                    k = k.replace('.ffn.layer_norm_bias', '.ffn.0.bias')
                    k = k.replace('.ffn.fc1_weight', '.ffn.1.weight')
                    k = k.replace('.ffn.fc2_weight', '.ffn.3.weight')
                    return k
                _new = {}
                for _k, _v in _sd.items():
                    _nk = _rm6b(_k)
                    if _nk:
                        _new[_nk] = _v
                self.pretrained_model = ESMC(d_model=2560, n_heads=40, n_layers=80,
                                             tokenizer=_get_tok())
                self.pretrained_model.load_state_dict(_new, strict=True)
            else:
                self.pretrained_model = ESMC.from_pretrained(_bm)
            self.pretrained_model.eval()
            import fair_esm as _fe
            _alph = _fe.data.Alphabet.from_architecture('ESM-1b')
            _ev = self.pretrained_model.tokenizer.get_vocab()
            _spec = {'<cls>': '<cls>', '<eos>': '<eos>', '<pad>': '<pad>',
                     '<mask>': '<mask>', '<unk>': '<unk>'}
            _lut = torch.full((len(_alph.all_toks),), _ev['<unk>'], dtype=torch.long)
            for _i, _t in enumerate(_alph.all_toks):
                _k = _spec.get(_t, _t)
                if _k in _ev:
                    _lut[_i] = _ev[_k]
            self.register_buffer('esm2_to_esmc', _lut, persistent=False)
        else:
            # route through fair_esm; `base_model` matches a fn in fair_esm.pretrained
            self.pretrained_model, _ = getattr(_fair_esm_pretrained, base_model)()
            self.pretrained_model.eval()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        layer_mapping = {
            "esm1v_t33_650M_UR90S_1": 33,
            "esm2_t33_650M_UR50D": 33,
            "esm2_t36_3B_UR50D": 36,
            # DPLM keeps the depth of the ESM2 it was trained from, so dplm_3b is
            # read at the same layer as the esm2_t36_3B_UR50D baseline.
            "dplm_150m": 30,
            "dplm_650m": 33,
            "dplm_3b": 36,
        }
        self.dim_tokens = dim_tokens
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)
            

        self.layer = layer_mapping.get(self.base_model, 33) 

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        seq_token = d['tensor']
        if self.is_dplm:
            # transformers needs the padding told to it explicitly; fair_esm reads
            # it off padding_idx internally. 1 is <pad> in the ESM-1b alphabet the
            # corpus is tokenised with.
            attn = (seq_token != 1).long()
            with torch.no_grad():
                out = self.pretrained_model(input_ids=seq_token,
                                            attention_mask=attn,
                                            output_hidden_states=True)
            # hidden_states is layers+1 long because index 0 is the embedding
            # output, so index self.layer is the last block - the same tensor
            # fair_esm returns for repr_layers=[self.layer].
            _proj_dtype = next(self.emb_proj.parameters()).dtype
            seq_embeddding = out.hidden_states[self.layer][:, 1:-1, :].to(_proj_dtype)
        elif self.is_esmc:
            # remap ESM2 alphabet ids -> ESMC ids, then take final-layer per-residue
            esmc_token = self.esm2_to_esmc.to(seq_token.device)[seq_token]
            with torch.no_grad():
                out = self.pretrained_model(esmc_token)
            # ESMC runs in bf16; cast per-residue emb to the projection dtype
            _proj_dtype = next(self.emb_proj.parameters()).dtype
            seq_embeddding = out.embeddings[:, 1:-1, :].to(_proj_dtype)  # strip BOS/EOS
        else:
            with torch.no_grad():
                results = self.pretrained_model(seq_token, repr_layers=[self.layer], return_contacts=False)
            seq_embeddding = results["representations"][self.layer][:,1:-1,:]  # Exclude the first and last tokens (BOS and EOS)
        
        B = seq_embeddding.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        # Map to embedding
        x = self.emb_proj(seq_embeddding)
        B, L, D = x.shape  
        mod_emb_expanded = self.mod_emb.expand(B, L, D)
        x_emb = mod_emb_expanded
        
        d['x'] = x
        d['emb'] = x_emb
        return d
    
class ProteinStructureEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 ):
        super().__init__()
        self.pretrained_model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open")
        self.pretrained_model.eval()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
      
        structure_token = d['tensor']  
        with torch.no_grad():
            structure_embeddding =self.pretrained_model.encoder.structure_tokens_embed(structure_token)

        B = structure_embeddding.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        # Map to embedding
        x = self.emb_proj(structure_embeddding)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d['input_mask']
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
