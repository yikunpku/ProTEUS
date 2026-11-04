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
# Upstream (ml-4m) re-exported 22 modules here, which pulled torchvision, wandb,
# boto3, webdataset and the image-augmentation stack into every import. Inference
# needs three names, so that is all this exports.
from .misc import generate_uint15_hash, to_1tuple, to_2tuple, to_3tuple, to_4tuple
from .timm.registry import model_entrypoint, register_model
from .timm.model_builder import create_model

__all__ = [
    "generate_uint15_hash",
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "model_entrypoint",
    "register_model",
    "create_model",
]
