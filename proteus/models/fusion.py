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

"""
Multi-modal Fusion Modules for Ablation Study

Implements different fusion strategies for combining encoder tokens from multiple modalities:
- add: Simple element-wise addition (default, backward compatible)
- weighted_sum: Learnable scalar weights with softmax normalization
- gated: Position-sensitive gating via MLP
- concat_project: Concatenation followed by linear projection
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSumFusion(nn.Module):
    """
    Learnable weighted sum fusion.

    Each modality gets a learnable scalar weight, normalized via softmax.
    Initial weights are set to zeros so that softmax gives uniform weights at init.

    Args:
        num_modalities: Number of modalities to fuse (default: 3)

    Extra parameters: num_modalities (e.g., 3 parameters)
    """
    def __init__(self, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        # Initialize to zeros so softmax gives uniform weights
        self.weights = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: List of tensors, each of shape (B, L, D)

        Returns:
            Fused tensor of shape (B, L, D)
        """
        assert len(token_list) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(token_list)}"

        # Softmax to ensure weights sum to 1
        w = F.softmax(self.weights, dim=0)

        # Weighted sum
        result = w[0] * token_list[0]
        for i in range(1, len(token_list)):
            result = result + w[i] * token_list[i]

        return result


class GatedFusion(nn.Module):
    """
    Position-sensitive gated fusion via MLP.

    Concatenates all modality tokens and passes through an MLP to generate
    per-position gates (softmax normalized across modalities).

    Args:
        dim: Hidden dimension (D)
        num_modalities: Number of modalities to fuse (default: 3)

    Extra parameters: ~886K for dim=768, num_modalities=3
        - Linear(dim*3, dim//2): 768*3 * 384 = 884,736
        - Linear(dim//2, 3): 384 * 3 = 1,152
        Total: ~886K
    """
    def __init__(self, dim: int, num_modalities: int = 3):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        hidden_dim = dim // 2

        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities)
        )

        # Initialize the last layer to zeros for uniform initial gates
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: List of tensors, each of shape (B, L, D)

        Returns:
            Fused tensor of shape (B, L, D)
        """
        assert len(token_list) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(token_list)}"

        # Concatenate along feature dimension: (B, L, D*num_modalities)
        concat = torch.cat(token_list, dim=-1)

        # Generate gates: (B, L, num_modalities)
        gates = F.softmax(self.gate_mlp(concat), dim=-1)

        # Stack modalities: (B, L, num_modalities, D)
        stacked = torch.stack(token_list, dim=2)

        # Apply gates and sum: (B, L, D)
        # gates: (B, L, num_modalities) -> (B, L, num_modalities, 1)
        result = (stacked * gates.unsqueeze(-1)).sum(dim=2)

        return result


class ConcatProjectFusion(nn.Module):
    """
    Concatenation followed by linear projection.

    Simply concatenates all modality tokens and projects back to original dimension.

    Args:
        dim: Hidden dimension (D)
        num_modalities: Number of modalities to fuse (default: 3)

    Extra parameters: ~1.77M for dim=768, num_modalities=3
        - Linear(dim*3, dim): 768*3 * 768 = 1,769,472
    """
    def __init__(self, dim: int, num_modalities: int = 3):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities

        self.projection = nn.Linear(dim * num_modalities, dim)

    def forward(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            token_list: List of tensors, each of shape (B, L, D)

        Returns:
            Fused tensor of shape (B, L, D)
        """
        assert len(token_list) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(token_list)}"

        # Concatenate along feature dimension: (B, L, D*num_modalities)
        concat = torch.cat(token_list, dim=-1)

        # Project back to original dimension: (B, L, D)
        return self.projection(concat)


def create_fusion_module(
    fusion_method: str,
    dim: int,
    num_modalities: int
) -> Optional[nn.Module]:
    """
    Factory function to create fusion modules.

    Args:
        fusion_method: One of ['add', 'weighted_sum', 'gated', 'concat_project']
        dim: Hidden dimension (D)
        num_modalities: Number of modalities to fuse

    Returns:
        Fusion module, or None if fusion_method is 'add' (for backward compatibility)
    """
    if fusion_method == 'add':
        # Return None for 'add' to maintain backward compatibility
        # The model will use the original element-wise addition
        return None
    elif fusion_method == 'weighted_sum':
        return WeightedSumFusion(num_modalities=num_modalities)
    elif fusion_method == 'gated':
        return GatedFusion(dim=dim, num_modalities=num_modalities)
    elif fusion_method == 'concat_project':
        return ConcatProjectFusion(dim=dim, num_modalities=num_modalities)
    else:
        raise ValueError(
            f"Unknown fusion method: {fusion_method}. "
            f"Choose from ['add', 'weighted_sum', 'gated', 'concat_project']"
        )
