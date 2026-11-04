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
# Deliberately empty. Upstream (ml-4m) star-imported the image augmenter, the
# modality transforms and the pretraining dataset builders here, which made
# cv2, albumentations and webdataset hard dependencies of every import.
