# Copyright 2024 EPFL and Apple Inc.
# Copyright 2026 The Proteus Authors
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
#
# Derived from ml-4m's fourm/data/modality_info.py. Upstream declared 38 vision
# modalities plus four MODALITY_TRANSFORMS tables, which made cv2, albumentations,
# PIL and torchvision hard dependencies. Proteus keeps only the five modalities the
# released checkpoint was trained on, verbatim, and drops the transform tables --
# protein inputs arrive pre-tokenized as .npy, so they never had a transform.
#
# Each 'id' is a hash of a fixed string, so removing entries cannot renumber the
# ones that remain. Do not change the hash seed strings.
from functools import partial

from proteus.models.encoder_embeddings import (MSAEmbEncoderEmbedding,
                                               ProteinSequenceEncoderEmbedding,
                                               ProteinStructureEncoderEmbedding)
from proteus.utils import generate_uint15_hash

MODALITY_INFO = {
    'sequence': {
        'vocab_size': 33,
        'encoder_embedding': partial(ProteinSequenceEncoderEmbedding, max_length=512, padding_idx=1, orig_emb_dim=1280, use_bottleneck=True),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 512,
        'type': 'protein_sequence',
        'id': generate_uint15_hash('protein_sequence'),
        'pretokenized': True,
    },
    'structure': {
        'vocab_size': 4096,
        'encoder_embedding': partial(ProteinStructureEncoderEmbedding, max_length=512, padding_idx=4099, orig_emb_dim=1536, use_bottleneck=True),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 512,
        'type': 'protein_structure',
        'id': generate_uint15_hash('protein_structure'),
        'pretokenized': True,
    },
    'msa_emb_128': {
        'encoder_embedding': partial(MSAEmbEncoderEmbedding, max_length=512, padding_idx=1, orig_emb_dim=768, use_bottleneck=True),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': 512,
        'type': 'protein_msa_emb_128',
        'id': generate_uint15_hash('protein_msa_emb_128'),
    },
    'scores_normalized': {
        'min_tokens': 1,
        'max_tokens': 1,
        'type': 'scores_normalized',
        'id': generate_uint15_hash('protein_scores_normalized'),
    },
    'mutation_mask': {
        'min_tokens': 0,
        'max_tokens': 512,
        'type': 'protein_mutation_mask',
        'id': generate_uint15_hash('protein_mutation_mask'),
    },
}
