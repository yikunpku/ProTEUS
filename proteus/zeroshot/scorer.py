"""Zero-shot fitness scoring.

The class body is this project's earlier scoring implementation, carried over
unchanged so that published numbers stay reproducible. Two things were edited
and nothing else: the class was renamed to match this package, and the helper
modules it reaches for now live under `proteus.zeroshot` and
`proteus.third_party`. No threshold, weight, sampling rule or default was
touched -- including a few that a fresh implementation would write
differently, such as the global `random.seed` and the silent fallbacks.

Note for readers: this scorer is an ensemble of three frozen third-party
models -- ESM2 masked marginals, ESM-MSA-1b masked marginals and ESM3
structure-conditioned logits. It does not run the Proteus trunk; the
checkpoint supplies one scalar per modality, used as the ensemble weights
(see `get_modality_weights`).
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

class ProteusZeroShot:
    """Proteus Zero-Shot Fitness Predictor with multi-modal protein representation."""

    def __init__(self,
                 checkpoint: Optional[str] = None,
                 use_sequence: bool = True,
                 use_msa: bool = True,
                 use_structure: bool = True,
                 structure_dir: Optional[str] = None,
                 device: str = 'cuda',
                 temperature: float = 1.0,
                 use_conservation: bool = True,
                 seq_basemodel: str = 'esm2_t36_3B_UR50D',
                 structure_score_dir: Optional[str] = None):
        """Initialize Proteus zero-shot predictor.

        structure_score_dir (opt-in, default None): directory of precomputed
        per-variant structure scores, one CSV per dataset with columns
        (mutant, <score>). When given, that column replaces the ESM3-token
        structure modality — used to score models whose structure modality is
        ESM-IF1's sequence-conditioned log-likelihood instead of ESM3 VQ tokens.
        Leaving it None keeps the original behaviour byte-for-byte.
        """
        import os
        import sys
        from pathlib import Path

        # Add parent directory to path for imports
        parent_dir = Path(__file__).resolve().parent.parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        import fair_esm
        from proteus.third_party.esm_fork.pretrained import load_model_and_alphabet
        from proteus.zeroshot.windows import get_optimal_window
        import numpy as np

        self.device = device
        self.use_sequence = use_sequence
        self.use_msa = use_msa
        self.use_structure = use_structure
        self.structure_dir = structure_dir
        self.structure_score_dir = structure_score_dir
        self.temperature = temperature
        self.use_conservation = use_conservation
        self.conservation_weights = None

        # Store utility functions
        self._get_optimal_window = get_optimal_window

        # Initialize models
        self.models = {}
        self.tokenizers = {}
        self.modality_embeddings = {}

        self.is_esmc = str(seq_basemodel).lower().startswith('esmc')
        if self.use_sequence:
            print(f"Loading sequence module ({seq_basemodel})...")
            if self.is_esmc:
                # ESMC (EvolutionaryScale) as the frozen sequence encoder. ESMC shares
                # ESM2's vocab ordering (cls0/pad1/eos2/unk3/mask32, AAs same idx), so we
                # reuse the ESM2 alphabet for tokenization + mask_idx and score with
                # ESMC's sequence_logits (validated: 47/49 argmax==WT, vocab-aligned).
                from esm.models.esmc import ESMC
                _bm = str(seq_basemodel).lower()
                if _bm in ('esmc_6b', 'esmc-6b', 'esmc6b'):
                    # ESMC-6B has no local entrypoint (esm routes 6B via forge API);
                    # load biohub's HF weights into esm's native ESMC(2560d/40h/80L)
                    # via a state_dict remap (same as ml-4m's encoder branch).
                    import glob as _glob
                    from safetensors.torch import load_file as _lf
                    from esm.tokenization import get_esmc_model_tokenizers as _get_tok
                    # Same cache location as before, resolved through the HF
                    # environment rather than a literal home directory.
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
                    _m = ESMC(d_model=2560, n_heads=40, n_layers=80, tokenizer=_get_tok())
                    _m.load_state_dict(_new, strict=True)
                    self.models['sequence'] = _m.to(device).eval()
                else:
                    self.models['sequence'] = ESMC.from_pretrained(_bm).to(device).eval()
                self.tokenizers['sequence'] = fair_esm.data.Alphabet.from_architecture('ESM-1b')
            else:
                self.models['sequence'], self.tokenizers['sequence'] = \
                    fair_esm.pretrained.esm2_t36_3B_UR50D()
                self.models['sequence'] = self.models['sequence'].to(device).eval()

        if self.use_msa:
            print("Loading MSA module...")
            self.models['msa'], self.tokenizers['msa'] = \
                load_model_and_alphabet("esm_msa1b_t12_100M_UR50S")
            self.msa_max_length = None
            self.models['msa'] = self.models['msa'].to(device).eval()

        if self.use_structure:
            print("Loading structure module...")
            try:
                # Initialize structure components
                try:
                    from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
                    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
                    from esm.utils.constants import esm3 as C
                except ImportError:
                    import esm
                    esm.login()
                    from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
                    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
                    from esm.utils.constants import esm3 as C

                self.models['structure'] = ESM3_sm_open_v0(device)
                self.models['structure'].eval()
                self.structure_encoder = ESM3_structure_encoder_v0(device)
                self.sequence_tokenizer = EsmSequenceTokenizer()

                # Store structure constants
                self.STRUCTURE_BOS_TOKEN = C.STRUCTURE_BOS_TOKEN
                self.STRUCTURE_EOS_TOKEN = C.STRUCTURE_EOS_TOKEN
                self.STRUCTURE_PAD_TOKEN = C.STRUCTURE_PAD_TOKEN
                self.STRUCTURE_MASK_TOKEN = C.STRUCTURE_MASK_TOKEN

            except Exception as e:
                print(f"Could not initialize structure module: {e}")
                self.use_structure = False

        # Load Proteus checkpoint
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path: str):
        """Load Proteus checkpoint with modality embeddings."""
        import os

        if not os.path.exists(checkpoint_path):
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)

            # Modality mapping
            modality_map = {
                'sequence': 'sequence',
                'msa': 'msa_emb_128',
                'structure': 'structure'
            }

            for mod_name, checkpoint_name in modality_map.items():
                if (mod_name == 'sequence' and not self.use_sequence) or \
                   (mod_name == 'msa' and not self.use_msa) or \
                   (mod_name == 'structure' and not self.use_structure):
                    continue

                # Extract modality embeddings
                for key in state_dict.keys():
                    if 'encoder_embeddings' in key and checkpoint_name in key and 'mod_emb' in key:
                        mod_emb = state_dict[key].detach().clone()
                        self.modality_embeddings[checkpoint_name] = mod_emb.to(self.device)
                        break

        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def compute_conservation_weights(self, msa):
        """Compute position-specific conservation weights from MSA."""
        import numpy as np

        if not msa or len(msa) < 2:
            return None

        # Convert to numpy array
        seqs = np.array([list(seq) for _, seq in msa])
        n_seqs, seq_len = seqs.shape

        # Calculate conservation (Shannon entropy)
        conservation = np.zeros(seq_len)
        for pos in range(seq_len):
            col = seqs[:, pos]
            # Count amino acid frequencies
            aa_counts = {}
            for aa in col:
                if aa != '-' and aa != 'X':
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1

            # Calculate entropy
            total = sum(aa_counts.values())
            if total > 0:
                entropy = 0
                for count in aa_counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * np.log(p)
                # Normalize entropy to [0, 1] and invert to get conservation
                max_entropy = np.log(20)  # 20 amino acids
                conservation[pos] = 1.0 - (entropy / max_entropy)
            else:
                conservation[pos] = 0.5  # Default for gap columns

        return conservation

    def compute_sequence_logits(self, sequence: str) -> torch.Tensor:
        """Compute masked-marginals for sequence modality."""
        from tqdm import tqdm

        batch_converter = self.tokenizers['sequence'].get_batch_converter()
        data = [("protein", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        tokenizer = self.tokenizers['sequence']
        model = self.models['sequence']

        all_log_probs = []

        for i in tqdm(range(tokens.size(1)), desc="Computing sequence predictions"):
            tokens_masked = tokens.clone()
            tokens_masked[0, i] = tokenizer.mask_idx

            with torch.no_grad():
                if getattr(self, 'is_esmc', False):
                    output = model(tokens_masked).sequence_logits  # ESMC: [B,L,vocab]
                else:
                    output = model(tokens_masked, repr_layers=[], return_contacts=False)["logits"]
                log_probs = F.log_softmax(output, dim=-1)

            all_log_probs.append(log_probs[:, i, :].cpu())

        return torch.cat(all_log_probs, dim=0).unsqueeze(0)

    def compute_msa_logits(self, msa, wt_sequence: str) -> torch.Tensor:
        """Compute masked-marginals for MSA modality."""
        from tqdm import tqdm

        # Truncate if needed
        if self.msa_max_length and len(msa[0][1]) > self.msa_max_length:
            msa = [(desc, seq[:self.msa_max_length]) for desc, seq in msa]

        batch_converter = self.tokenizers['msa'].get_batch_converter()
        data = [msa]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self.device)

        tokenizer = self.tokenizers['msa']
        model = self.models['msa']

        all_log_probs = []

        for i in tqdm(range(tokens.size(2)), desc="Computing MSA predictions"):
            tokens_masked = tokens.clone()
            tokens_masked[0, 0, i] = tokenizer.mask_idx

            # Handle long sequences with windowing
            if tokens.size(-1) > 1024:
                large_tokens_masked = tokens_masked.clone()
                start, end = self._get_optimal_window(i, len(wt_sequence) + 2, 1024)
                tokens_masked = large_tokens_masked[:, :, start:end]
            else:
                start = 0

            with torch.no_grad():
                output = model(tokens_masked, repr_layers=[], return_contacts=False)["logits"]
                log_probs = F.log_softmax(output, dim=-1)

            all_log_probs.append(log_probs[:, 0, i - start, :].detach().cpu())

        return torch.cat(all_log_probs, dim=0).unsqueeze(0)

    def compute_structure_logits(self, pdb_path: str, sequence: str):
        """
        Compute masked-marginals for structure from PDB file.

        Returns:
            Tuple of (structure_log_probs, sequence_log_probs, structure_tokens)
        """
        from biotite.structure.io.pdb import PDBFile
        from esm.utils.structure.protein_chain import ProteinChain
        from tqdm import tqdm
        import numpy as np

        # Load structure
        structure = PDBFile.read(pdb_path)
        chain_ids = np.unique(structure.get_structure().chain_id)
        if len(chain_ids) == 0:
            raise ValueError("No chains found in PDB")

        # Use first chain
        chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_ids[0])
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.to(self.device)
        residue_index = residue_index.to(self.device)

        # Get the actual sequence from the structure
        structure_sequence = chain.sequence  # Sequence with resolved coordinates

        # Encode structure
        with torch.no_grad():
            torch.cuda.empty_cache()
            _, structure_tokens = self.structure_encoder.encode(coords, residue_index=residue_index)

        structure_tokens = structure_tokens.squeeze(0)

        # Flatten residue_index if it's 2D
        if residue_index.dim() == 2:
            residue_index = residue_index.squeeze(0)

        # Tokenize sequence
        sequence_tokens = self.sequence_tokenizer.encode(sequence)
        sequence_tokens_tensor = torch.tensor(sequence_tokens).to(self.device)
        seq_length_no_special = len(sequence_tokens) - 2  # Remove BOS/EOS

        # Create full structure tokens array with padding for missing residues
        full_structure_tokens = torch.full(
            (seq_length_no_special,),
            self.STRUCTURE_PAD_TOKEN,
            dtype=torch.long,
            device=self.device
        )

        # Map structure tokens to their correct positions based on residue indices
        if len(residue_index) == len(structure_tokens):
            for i, res_idx in enumerate(residue_index):
                # residue_index is 1-indexed, convert to 0-indexed
                seq_pos = res_idx.item() - 1
                if 0 <= seq_pos < seq_length_no_special:
                    full_structure_tokens[seq_pos] = structure_tokens[i]
        else:
            # Fallback: if we can't map by residue index, try sequence alignment
            try:
                from Bio import pairwise2
                # Perform global alignment
                alignments = pairwise2.align.globalxx(sequence, structure_sequence)
                if alignments:
                    alignment = alignments[0]
                    seq_aligned, struct_aligned = alignment[0], alignment[1]

                    seq_idx = 0
                    struct_idx = 0
                    for s_char, struct_char in zip(seq_aligned, struct_aligned):
                        if s_char != '-':
                            if struct_char != '-' and struct_idx < len(structure_tokens):
                                full_structure_tokens[seq_idx] = structure_tokens[struct_idx]
                            seq_idx += 1
                        if struct_char != '-':
                            struct_idx += 1
            except:
                # If alignment fails, just pad at the end
                full_structure_tokens[:len(structure_tokens)] = structure_tokens

        # Add special tokens
        structure_tokens_full = torch.cat([
            torch.tensor([self.STRUCTURE_BOS_TOKEN]).to(self.device),
            full_structure_tokens,
            torch.tensor([self.STRUCTURE_EOS_TOKEN]).to(self.device)
        ])

        # Compute masked marginals
        all_structure_log_probs = []
        all_sequence_log_probs = []

        for i in tqdm(range(len(structure_tokens_full)), desc="Computing structure predictions"):
            masked_structure = structure_tokens_full.clone()
            masked_structure[i] = self.STRUCTURE_MASK_TOKEN

            with torch.no_grad():
                output = self.models['structure'](
                    sequence_tokens=sequence_tokens_tensor.unsqueeze(0),
                    structure_tokens=masked_structure.unsqueeze(0)
                )

                structure_logits = output.structure_logits
                structure_log_probs = F.log_softmax(structure_logits, dim=-1)

                sequence_logits = output.sequence_logits
                sequence_log_probs = F.log_softmax(sequence_logits, dim=-1)

            all_structure_log_probs.append(structure_log_probs[:, i, :].cpu())
            all_sequence_log_probs.append(sequence_log_probs[:, i, :].cpu())

        structure_log_probs_all = torch.cat(all_structure_log_probs, dim=0).unsqueeze(0)
        sequence_log_probs_all = torch.cat(all_sequence_log_probs, dim=0).unsqueeze(0)

        return structure_log_probs_all, sequence_log_probs_all, structure_tokens_full.cpu()

    def get_modality_weights(self, active_modalities):
        """Get modality contribution weights from checkpoint."""
        if not self.modality_embeddings:
            return {mod: 1.0 / len(active_modalities) for mod in active_modalities}

        modality_map = {
            'sequence': 'sequence',
            'msa': 'msa_emb_128',
            'structure': 'structure'
        }

        weights = {}
        for mod in active_modalities:
            checkpoint_name = modality_map.get(mod, mod)
            if checkpoint_name in self.modality_embeddings:
                mod_emb = self.modality_embeddings[checkpoint_name]
                weight = torch.abs(mod_emb).mean().item()
                weights[mod] = weight
            else:
                weights[mod] = 1.0

        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights

    def score_mutations(self, mutation_str: str, sequence: str,
                       token_probs: torch.Tensor, tokenizer,
                       offset_idx: int = 1, conservation_weights=None,
                       use_epistasis: bool = True) -> float:
        """Score mutations using log probability differences with conservation weighting and epistatic effects."""
        import numpy as np

        # Apply temperature scaling
        if self.temperature != 1.0:
            token_probs = token_probs / self.temperature

        mutations = mutation_str.split(":")
        n_mutations = len(mutations)

        # Parse all mutations first
        mut_info = []
        for mutation in mutations:
            wt = mutation[0]
            idx = int(mutation[1:-1]) - offset_idx
            mt = mutation[-1]

            if idx < 0 or idx >= len(sequence):
                continue
            if sequence[idx] != wt:
                continue

            wt_encoded = tokenizer.get_idx(wt)
            mt_encoded = tokenizer.get_idx(mt)
            mut_info.append((idx, wt_encoded, mt_encoded))

        # Calculate individual mutation scores
        individual_scores = []
        for idx, wt_encoded, mt_encoded in mut_info:
            mutation_score = (token_probs[0, 1 + idx, mt_encoded] -
                            token_probs[0, 1 + idx, wt_encoded]).item()

            # Apply conservation weighting if available
            if conservation_weights is not None and self.use_conservation:
                conservation_weight = conservation_weights[idx] if idx < len(conservation_weights) else 1.0
                # Higher conservation = larger penalty for mutations
                mutation_score *= (1.0 + conservation_weight)

            individual_scores.append(mutation_score)

        # Sum individual effects
        score = sum(individual_scores)

        # Add epistatic correction for multiple mutations
        if use_epistasis and n_mutations > 1:
            # Calculate pairwise distance-based epistasis
            epistasis_penalty = 0
            for i in range(len(mut_info)):
                for j in range(i + 1, len(mut_info)):
                    idx1, _, _ = mut_info[i]
                    idx2, _, _ = mut_info[j]
                    distance = abs(idx1 - idx2)

                    # Closer mutations have stronger epistatic effects
                    if distance < 10:  # Within 10 residues
                        # Penalty increases as mutations get closer
                        proximity_factor = 1.0 - (distance / 10.0)
                        epistasis_penalty += proximity_factor * 0.1 * abs(individual_scores[i] * individual_scores[j]) ** 0.5

            # Apply epistatic penalty (negative epistasis assumption)
            score -= epistasis_penalty

        return score

    def score_structure_mutations(self, mutation_str: str, sequence: str,
                                  structure_log_probs: torch.Tensor,
                                  wt_structure_tokens: torch.Tensor,
                                  offset_idx: int = 1) -> float:
        """Score mutations for structure modality."""
        score = 0.0
        vocab_size = structure_log_probs.shape[-1]

        for mutation in mutation_str.split(":"):
            wt = mutation[0]
            idx = int(mutation[1:-1]) - offset_idx
            mt = mutation[-1]

            # Validate
            if idx < 0 or idx >= len(sequence):
                continue
            if sequence[idx] != wt:
                continue

            # Get WT structure token
            wt_struct_token = wt_structure_tokens[1 + idx].item()

            # Check validity
            if wt_struct_token >= vocab_size:
                continue

            # Structure score
            log_prob = structure_log_probs[0, 1 + idx, wt_struct_token].item()
            score += log_prob

        return score

    def score_sequence_from_structure(self, mutation_str: str, sequence: str,
                                      sequence_log_probs: torch.Tensor,
                                      offset_idx: int = 1) -> float:
        """Score mutations using structure-conditioned sequence predictions."""
        score = 0.0

        for mutation in mutation_str.split(":"):
            wt = mutation[0]
            idx = int(mutation[1:-1]) - offset_idx
            mt = mutation[-1]

            # Validate
            if idx < 0 or idx >= len(sequence):
                continue
            if sequence[idx] != wt:
                continue

            # Get token indices
            wt_encoded = self.sequence_tokenizer.encode(wt)[1]  # Skip BOS
            mt_encoded = self.sequence_tokenizer.encode(mt)[1]  # Skip BOS

            # Compute score
            score += (sequence_log_probs[0, 1 + idx, mt_encoded] -
                     sequence_log_probs[0, 1 + idx, wt_encoded]).item()

        return score

    @staticmethod
    def get_mt_tokens_path(mutant: str, dataset: str, token_dir: str) -> str:
        """
        Generate path to pre-computed MT structure tokens file.

        Uses the same naming convention as get_structure_tokens_from_boltz.py:
        - WT: {dataset}_WT.npy
        - Mutants: {dataset}_{num_muts}mut_{md5_hash[:16]}.npy

        Args:
            mutant: Mutation string (e.g., "A123V" or "A123V:G456D")
            dataset: Dataset name (e.g., "aav", "gb1")
            token_dir: Directory containing pre-computed tokens

        Returns:
            Path to the token file
        """
        import hashlib
        import os

        if mutant == "WT" or mutant == "wild_type":
            return os.path.join(token_dir, f"{dataset}_WT.npy")

        num_muts = len(mutant.split(':'))
        mut_hash = hashlib.md5(mutant.encode()).hexdigest()[:16]
        return os.path.join(token_dir, f"{dataset}_{num_muts}mut_{mut_hash}.npy")

    def load_mt_structure_tokens(self, mutant: str, dataset: str, token_dir: str):
        """
        Load pre-computed MT structure tokens from file.

        Args:
            mutant: Mutation string
            dataset: Dataset name
            token_dir: Directory containing pre-computed tokens

        Returns:
            numpy array of structure tokens, or None if not found
        """
        import numpy as np
        import os

        token_path = self.get_mt_tokens_path(mutant, dataset, token_dir)

        if os.path.exists(token_path):
            try:
                return np.load(token_path)
            except Exception as e:
                print(f"Error loading MT tokens from {token_path}: {e}")
                return None
        return None

    def compute_mt_structure_score(self,
                                    mt_structure_tokens,
                                    mt_sequence: str,
                                    wt_sequence: str,
                                    mutation_str: str,
                                    offset_idx: int = 1) -> float:
        """
        Compute fitness score using MT structure to condition sequence prediction.

        This method uses the predicted mutant structure to evaluate how well
        the mutant sequence fits with its predicted structure.

        Scientific rationale:
        1. A good mutation should have a structure that "accepts" the new sequence
        2. If MT structure predicts high probability for MT amino acids, the mutation is favorable
        3. This captures structure-sequence consistency after mutation

        Args:
            mt_structure_tokens: Pre-computed structure tokens for the mutant (numpy array)
            mt_sequence: The mutant sequence
            wt_sequence: The wild-type sequence
            mutation_str: Mutation string (e.g., "A123V:G456D")
            offset_idx: Position offset for mutation indexing

        Returns:
            Structure-conditioned sequence score (higher = better fit)
        """
        import numpy as np

        # Convert to tensor and add BOS/EOS
        if hasattr(mt_structure_tokens, 'numpy'):
            mt_structure_tokens = mt_structure_tokens
        else:
            mt_structure_tokens = np.array(mt_structure_tokens)

        # Build full structure tokens with BOS/EOS
        structure_tokens_full = np.concatenate([
            [self.STRUCTURE_BOS_TOKEN],
            mt_structure_tokens,
            [self.STRUCTURE_EOS_TOKEN]
        ])
        structure_tokens_tensor = torch.tensor(structure_tokens_full, dtype=torch.long).to(self.device)

        # Tokenize MT sequence
        mt_sequence_tokens = self.sequence_tokenizer.encode(mt_sequence)
        mt_sequence_tensor = torch.tensor(mt_sequence_tokens).to(self.device)

        # Also tokenize WT sequence for comparison
        wt_sequence_tokens = self.sequence_tokenizer.encode(wt_sequence)
        wt_sequence_tensor = torch.tensor(wt_sequence_tokens).to(self.device)

        # Parse mutations to know which positions to score
        mutations = []
        for mut in mutation_str.split(':'):
            if len(mut) < 3:
                continue
            try:
                wt_aa = mut[0]
                pos = int(mut[1:-1]) - offset_idx  # 0-indexed
                mt_aa = mut[-1]
                mutations.append((pos, wt_aa, mt_aa))
            except ValueError:
                continue

        if not mutations:
            return 0.0

        score = 0.0

        # For each mutation position, compute the log probability difference
        # P(mt_aa | mt_structure) vs P(wt_aa | mt_structure)
        for pos, wt_aa, mt_aa in mutations:
            # Validate position
            if pos < 0 or pos >= len(mt_sequence):
                continue

            # Position in tokenized sequence (accounting for BOS token)
            token_pos = pos + 1

            # Get token indices
            wt_token = self.sequence_tokenizer.encode(wt_aa)[1]  # Skip BOS
            mt_token = self.sequence_tokenizer.encode(mt_aa)[1]  # Skip BOS

            # Mask the position and get prediction
            masked_seq = mt_sequence_tensor.clone()
            masked_seq[token_pos] = self.sequence_tokenizer.mask_token_id

            with torch.no_grad():
                output = self.models['structure'](
                    sequence_tokens=masked_seq.unsqueeze(0),
                    structure_tokens=structure_tokens_tensor.unsqueeze(0)
                )
                seq_logits = output.sequence_logits
                log_probs = F.log_softmax(seq_logits, dim=-1)

            # Score: P(mt_aa | mt_structure) - P(wt_aa | mt_structure)
            # If mt_structure "prefers" mt_aa, this will be positive
            mt_log_prob = log_probs[0, token_pos, mt_token].item()
            wt_log_prob = log_probs[0, token_pos, wt_token].item()

            score += (mt_log_prob - wt_log_prob)

        return score

    def compute_structure_consistency_score(self,
                                            wt_structure_tokens,
                                            mt_structure_tokens,
                                            wt_sequence: str,
                                            mt_sequence: str,
                                            mutation_str: str,
                                            offset_idx: int = 1) -> float:
        """
        Compute structure consistency score by comparing WT and MT structure-sequence compatibility.

        This method measures:
        1. How well MT sequence fits with MT structure (should be high for good mutations)
        2. How well WT sequence fits with WT structure (baseline)

        A positive score indicates the mutation maintains or improves structure-sequence compatibility.

        Args:
            wt_structure_tokens: WT structure tokens
            mt_structure_tokens: MT structure tokens
            wt_sequence: Wild-type sequence
            mt_sequence: Mutant sequence
            mutation_str: Mutation string
            offset_idx: Position offset

        Returns:
            Consistency score (higher = better structural adaptation)
        """
        import numpy as np

        # Parse mutations
        mutations = []
        for mut in mutation_str.split(':'):
            if len(mut) < 3:
                continue
            try:
                pos = int(mut[1:-1]) - offset_idx
                mt_aa = mut[-1]
                mutations.append((pos, mt_aa))
            except ValueError:
                continue

        if not mutations:
            return 0.0

        # Build structure tokens with BOS/EOS for both WT and MT
        def prepare_tokens(tokens):
            if hasattr(tokens, 'numpy'):
                tokens = tokens.numpy()
            return np.concatenate([[self.STRUCTURE_BOS_TOKEN], tokens, [self.STRUCTURE_EOS_TOKEN]])

        wt_struct_full = torch.tensor(prepare_tokens(wt_structure_tokens), dtype=torch.long).to(self.device)
        mt_struct_full = torch.tensor(prepare_tokens(mt_structure_tokens), dtype=torch.long).to(self.device)

        # Tokenize sequences
        wt_seq_tokens = torch.tensor(self.sequence_tokenizer.encode(wt_sequence)).to(self.device)
        mt_seq_tokens = torch.tensor(self.sequence_tokenizer.encode(mt_sequence)).to(self.device)

        score = 0.0

        for pos, mt_aa in mutations:
            if pos < 0 or pos >= len(mt_sequence):
                continue

            token_pos = pos + 1  # Account for BOS
            mt_token = self.sequence_tokenizer.encode(mt_aa)[1]

            # Score MT sequence at mutation position given MT structure
            masked_mt_seq = mt_seq_tokens.clone()
            masked_mt_seq[token_pos] = self.sequence_tokenizer.mask_token_id

            with torch.no_grad():
                # MT structure + MT sequence
                output_mt = self.models['structure'](
                    sequence_tokens=masked_mt_seq.unsqueeze(0),
                    structure_tokens=mt_struct_full.unsqueeze(0)
                )
                mt_log_probs = F.log_softmax(output_mt.sequence_logits, dim=-1)
                mt_score = mt_log_probs[0, token_pos, mt_token].item()

                # WT structure + MT sequence (for comparison)
                output_wt = self.models['structure'](
                    sequence_tokens=masked_mt_seq.unsqueeze(0),
                    structure_tokens=wt_struct_full.unsqueeze(0)
                )
                wt_log_probs = F.log_softmax(output_wt.sequence_logits, dim=-1)
                wt_score = wt_log_probs[0, token_pos, mt_token].item()

            # If MT structure predicts MT amino acid better than WT structure does,
            # this indicates good structural adaptation
            score += (mt_score - wt_score)

        return score

    def compute_structure_perturbation(self,
                                        wt_tokens,
                                        mt_tokens,
                                        mutation_str: str,
                                        offset_idx: int = 1,
                                        window: int = 3,
                                        sigma: float = 1.5) -> float:
        """
        Compute local structure perturbation around mutation sites.

        This method calculates how much the structure changes locally around
        each mutation position, using a Gaussian-weighted window.

        Scientific rationale:
        1. Mutations affect not just the mutated residue, but neighboring residues too
        2. The effect decays with distance from the mutation site
        3. Gaussian weighting provides smooth decay

        Args:
            wt_tokens: Wild-type structure tokens (numpy array or tensor)
            mt_tokens: Mutant structure tokens (numpy array or tensor)
            mutation_str: Mutation string (e.g., "A123V:G456D")
            offset_idx: Position offset for mutation indexing
            window: Window size around mutation position
            sigma: Gaussian sigma for distance weighting

        Returns:
            Structure perturbation score (higher = more perturbation)
        """
        import numpy as np

        # Convert to numpy if needed
        if hasattr(wt_tokens, 'numpy'):
            wt_tokens = wt_tokens.numpy()
        if hasattr(mt_tokens, 'numpy'):
            mt_tokens = mt_tokens.numpy()

        # Parse mutation positions
        positions = []
        for mut in mutation_str.split(':'):
            if len(mut) < 3:
                continue
            try:
                pos = int(mut[1:-1]) - offset_idx
                positions.append(pos)
            except ValueError:
                continue

        if not positions:
            return 0.0

        # Calculate perturbation with Gaussian weighting
        total_perturbation = 0.0

        for mut_pos in positions:
            # Define window around mutation
            start = max(0, mut_pos - window)
            end = min(len(wt_tokens), mut_pos + window + 1)

            for i in range(start, end):
                # Check if tokens are different
                if i < len(wt_tokens) and i < len(mt_tokens):
                    if wt_tokens[i] != mt_tokens[i]:
                        # Gaussian weight based on distance from mutation
                        dist = abs(i - mut_pos)
                        weight = np.exp(-dist**2 / (2 * sigma**2))
                        total_perturbation += weight

        return total_perturbation


    def _load_external_structure_scores(self, dataset_name, df):
        """Per-variant structure scores from `structure_score_dir` aligned to df.

        Opt-in helper (only called when structure_score_dir is set). Accepts any
        CSV with a `mutant` column plus one score column; the score column is
        auto-detected (first column whose name ends with `_score` and is not a
        DMS label). wild_type is pinned to 0.0 like the other modalities, and
        variants with no entry fall back to 0.0 (counted and reported).
        """
        import os
        import pandas as pd
        for cand in (f'{dataset_name}.csv', f'{dataset_name}_zeroshot.csv',
                     f'{dataset_name.upper()}.csv'):
            path = os.path.join(self.structure_score_dir, cand)
            if os.path.exists(path):
                break
        else:
            print(f"  [structure_score_dir] no CSV for {dataset_name}, "
                  f"keeping the default structure modality")
            return None
        ext = pd.read_csv(path, low_memory=False)
        if 'mutant' not in ext.columns:
            print(f"  [structure_score_dir] {path} has no 'mutant' column")
            return None
        score_cols = [c for c in ext.columns
                      if c.endswith('_score') and not c.startswith('DMS')]
        if not score_cols:
            print(f"  [structure_score_dir] {path} has no *_score column")
            return None
        col = score_cols[0]
        lut = dict(zip(ext['mutant'].astype(str), ext[col].astype(float)))
        out, missing = [], 0
        for _, row in df.iterrows():
            m = str(row['mutant'])
            if m in ('wild_type', 'WT'):
                out.append(0.0)
                continue
            if m in lut:
                out.append(float(lut[m]))
            else:
                out.append(0.0)
                missing += 1
        print(f"  [structure_score_dir] {os.path.basename(path)} column '{col}': "
              f"{len(df) - missing}/{len(df)} variants matched, {missing} missing->0.0")
        return out

    def score(self,
              dms_file: str,
              msa_dir: Optional[str] = None,
              msa_weights_dir: Optional[str] = None,
              structure_dir: Optional[str] = None,
              mt_tokens_dir: Optional[str] = None,
              msa_samples: int = 400,
              offset_idx: int = 1,
              num_seeds: int = 5,
              use_epistasis: bool = True,
              filter_msa: bool = False,
              path_to_hhfilter: Optional[str] = None,
              use_structure_perturbation: bool = True,
              lambda_perturbation: float = 0.1,
              perturbation_window: int = 3,
              perturbation_sigma: float = 1.5,
              use_mt_structure_score: bool = True,
              lambda_mt_structure: float = 0.3):
        """
        Execute zero-shot fitness prediction on a DMS dataset.

        Args:
            dms_file: Path to DMS CSV file with mutation data
            msa_dir: Directory containing MSA files
            msa_weights_dir: Directory containing MSA weight files
            structure_dir: Directory with PDB structure files
            mt_tokens_dir: Directory with pre-computed MT structure tokens (from Boltz)
            msa_samples: Number of MSA sequences to sample
            offset_idx: Offset for mutation position indexing
            num_seeds: Number of random seeds for MSA sampling
            use_epistasis: Model epistatic effects for multiple mutations
            filter_msa: Filter MSA with HHfilter
            path_to_hhfilter: Path to hhfilter executable
            use_structure_perturbation: Whether to use structure perturbation signal (deprecated, use use_mt_structure_score)
            lambda_perturbation: Weight for structure perturbation penalty (deprecated)
            perturbation_window: Window size around mutation for perturbation calculation (deprecated)
            perturbation_sigma: Gaussian sigma for distance weighting (deprecated)
            use_mt_structure_score: Whether to use MT structure-conditioned scoring (NEW)
            lambda_mt_structure: Weight for MT structure score contribution (NEW)

        Returns:
            List of fitness scores for each variant
        """
        import pandas as pd
        from pathlib import Path
        from tqdm import tqdm
        import sys

        # Import MSA utilities
        parent_dir = Path(__file__).resolve().parent.parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from proteus.zeroshot.msa_sampling import process_msa, sample_msa

        print("\n=== Proteus Zero-Shot Fitness Prediction ===")
        print(f"Active modalities: Sequence={self.use_sequence}, MSA={self.use_msa}, Structure={self.use_structure}")

        # Load DMS data
        df = pd.read_csv(dms_file)
        wt_row = df[df['mutant'] == 'wild_type']
        wt_sequence = wt_row['mutated_sequence'].values[0].upper()
        print(f"Dataset: {Path(dms_file).stem}, Variants: {len(df)}, WT length: {len(wt_sequence)}")

        modality_logits = {}
        wt_structure_tokens = None
        structure_seq_logits = None

        # Compute sequence logits
        if self.use_sequence:
            print("\nProcessing sequence modality...")
            modality_logits['sequence'] = self.compute_sequence_logits(wt_sequence)

        # Compute MSA logits
        if self.use_msa and msa_dir:
            print("\nProcessing MSA modality...")
            dataset_name = Path(dms_file).stem
            msa_file = None
            for ext in ['.a3m', '.a2m', '.fasta', '.afa']:
                candidate = Path(msa_dir) / f"{dataset_name}{ext}"
                if candidate.exists():
                    msa_file = str(candidate)
                    break

            if msa_file:
                weight_file = None
                if msa_weights_dir:
                    weight_candidate = Path(msa_weights_dir) / f"{dataset_name}.npy"
                    weight_file = str(weight_candidate)

                # Process MSA with optional filtering
                processed_msa = process_msa(msa_file, weight_file, filter_msa, path_to_hhfilter)

                # Multi-seed inference
                all_msa_logits = []
                all_msa_sequences = []  # Store for conservation computation
                for seed in range(num_seeds):
                    msa_sequences = sample_msa(
                        msa_file, msa_samples, seed+1, weight_file, processed_msa
                    )

                    if msa_sequences and len(msa_sequences) >= 2:
                        try:
                            seed_logits = self.compute_msa_logits(msa_sequences, wt_sequence)
                            all_msa_logits.append(seed_logits)
                            all_msa_sequences.extend(msa_sequences)  # Collect for conservation
                        except Exception as e:
                            print(f"Error processing MSA: {e}")

                if all_msa_logits:
                    modality_logits['msa'] = torch.stack(all_msa_logits).mean(dim=0)

                    # Compute conservation weights from all MSA sequences
                    if self.use_conservation and all_msa_sequences:
                        # Remove duplicates while preserving order
                        unique_msa = []
                        seen = set()
                        for desc, seq in all_msa_sequences:
                            if seq not in seen:
                                unique_msa.append((desc, seq))
                                seen.add(seq)

                        self.conservation_weights = self.compute_conservation_weights(unique_msa)
                        if self.conservation_weights is not None:
                            print(f"Computed conservation weights from {len(unique_msa)} unique sequences")

        # Compute structure logits
        if self.use_structure and structure_dir:
            print("\nProcessing structure modality...")
            dataset_name = Path(dms_file).stem
            pdb_path = Path(structure_dir) / f"{dataset_name}.pdb"

            if pdb_path.exists():
                try:
                    structure_logits, seq_logits, wt_structure_tokens = \
                        self.compute_structure_logits(str(pdb_path), wt_sequence)

                    modality_logits['structure'] = structure_logits
                    structure_seq_logits = seq_logits

                except Exception as e:
                    print(f"Error processing structure: {e}")
                    self.use_structure = False

        # Score variants
        print("\nScoring variants...")
        modality_scores = {}
        active_modalities = []

        # Score each modality
        for mod_name, logits in modality_logits.items():
            scores = []

            if mod_name == 'structure':
                if wt_structure_tokens is None or structure_seq_logits is None:
                    continue

                structure_scores = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Structure scoring"):
                    if row['mutant'] == 'wild_type':
                        structure_scores.append(0.0)
                    else:
                        score = self.score_structure_mutations(
                            row['mutant'], wt_sequence, logits,
                            wt_structure_tokens, offset_idx
                        )
                        structure_scores.append(score)

                sequence_scores = []
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Sequence-structure scoring"):
                    if row['mutant'] == 'wild_type':
                        sequence_scores.append(0.0)
                    else:
                        score = self.score_sequence_from_structure(
                            row['mutant'], wt_sequence, structure_seq_logits, offset_idx
                        )
                        sequence_scores.append(score)

                combined_scores = [(s1 + s2) / 2.0 for s1, s2 in zip(structure_scores, sequence_scores)]
                modality_scores[mod_name] = combined_scores
                active_modalities.append(mod_name)

            else:
                tokenizer = self.tokenizers[mod_name]

                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{mod_name.capitalize()} scoring"):
                    if row['mutant'] == 'wild_type':
                        scores.append(0.0)
                    else:
                        score = self.score_mutations(
                            row['mutant'], wt_sequence, logits, tokenizer, offset_idx,
                            conservation_weights=self.conservation_weights,
                            use_epistasis=use_epistasis
                        )
                        scores.append(score)

                modality_scores[mod_name] = scores
                active_modalities.append(mod_name)

        # ── opt-in: external per-variant structure scores ───────────────────
        # Only runs when `structure_score_dir` was passed (default None), so the
        # ESM3-token path above is unchanged for every existing caller.
        if getattr(self, 'structure_score_dir', None):
            ext = self._load_external_structure_scores(Path(dms_file).stem, df)
            if ext is not None:
                modality_scores['structure'] = ext
                if 'structure' not in active_modalities:
                    active_modalities.append('structure')
                print(f"Structure modality taken from external scores "
                      f"({self.structure_score_dir})")

        modality_weights = self.get_modality_weights(active_modalities)

        # Load WT structure tokens for MT structure scoring
        wt_structure_tokens_for_mt = None
        if (use_mt_structure_score or use_structure_perturbation) and mt_tokens_dir:
            import numpy as np
            import os

            dataset_name = Path(dms_file).stem

            # Try to load WT tokens from pre-computed directory
            wt_token_path = os.path.join(mt_tokens_dir, f"{dataset_name}_WT.npy")
            if os.path.exists(wt_token_path):
                try:
                    wt_structure_tokens_for_mt = np.load(wt_token_path)
                    print(f"Loaded WT structure tokens from {wt_token_path}")
                except Exception as e:
                    print(f"Error loading WT tokens: {e}")

            # Fallback: use tokens from structure computation (if available)
            if wt_structure_tokens_for_mt is None and wt_structure_tokens is not None:
                # wt_structure_tokens includes BOS/EOS, remove them
                wt_structure_tokens_for_mt = wt_structure_tokens[1:-1].numpy()
                print("Using WT structure tokens from ESM3 encoding")

        # Compute MT structure-conditioned scores (NEW: truly utilizes MT structure)
        mt_structure_scores = {}
        if use_mt_structure_score and mt_tokens_dir and self.use_structure and wt_structure_tokens_for_mt is not None:
            print(f"\nComputing MT structure-conditioned scores (λ={lambda_mt_structure})...")
            print("This uses Boltz-predicted MT structures to evaluate sequence-structure consistency")
            dataset_name = Path(dms_file).stem

            mt_score_list = []
            mt_found_count = 0
            mt_missing_count = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc="MT structure scoring"):
                if row['mutant'] == 'wild_type':
                    mt_score_list.append(0.0)
                else:
                    # Load MT tokens
                    mt_tokens = self.load_mt_structure_tokens(
                        row['mutant'], dataset_name, mt_tokens_dir
                    )

                    if mt_tokens is not None:
                        mt_found_count += 1
                        try:
                            # Get mutant sequence
                            mt_sequence = row.get('mutated_sequence', None)
                            if mt_sequence is None:
                                # Reconstruct from WT + mutations
                                mt_sequence = list(wt_sequence)
                                for mut in row['mutant'].split(':'):
                                    if len(mut) >= 3:
                                        pos = int(mut[1:-1]) - offset_idx
                                        mt_aa = mut[-1]
                                        if 0 <= pos < len(mt_sequence):
                                            mt_sequence[pos] = mt_aa
                                mt_sequence = ''.join(mt_sequence)

                            # Compute structure consistency score
                            # This measures: P(mt_aa | mt_structure) - P(mt_aa | wt_structure)
                            mt_score = self.compute_structure_consistency_score(
                                wt_structure_tokens_for_mt,
                                mt_tokens,
                                wt_sequence,
                                mt_sequence,
                                row['mutant'],
                                offset_idx=offset_idx
                            )
                            mt_score_list.append(mt_score)
                        except Exception as e:
                            print(f"Error computing MT score for {row['mutant']}: {e}")
                            mt_score_list.append(0.0)
                    else:
                        mt_missing_count += 1
                        mt_score_list.append(0.0)

            mt_structure_scores['mt_structure'] = mt_score_list
            print(f"MT tokens found: {mt_found_count}, missing: {mt_missing_count}")

        # Legacy: Compute structure perturbation scores if enabled (for backward compatibility)
        perturbation_scores = {}
        if use_structure_perturbation and not use_mt_structure_score and mt_tokens_dir and wt_structure_tokens_for_mt is not None:
            print(f"\nComputing structure perturbation scores (legacy mode)...")
            dataset_name = Path(dms_file).stem

            perturbation_list = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Structure perturbation"):
                if row['mutant'] == 'wild_type':
                    perturbation_list.append(0.0)
                else:
                    mt_tokens = self.load_mt_structure_tokens(
                        row['mutant'], dataset_name, mt_tokens_dir
                    )
                    if mt_tokens is not None:
                        perturbation = self.compute_structure_perturbation(
                            wt_structure_tokens_for_mt,
                            mt_tokens,
                            row['mutant'],
                            offset_idx=offset_idx,
                            window=perturbation_window,
                            sigma=perturbation_sigma
                        )
                        perturbation_list.append(perturbation)
                    else:
                        perturbation_list.append(0.0)

            perturbation_scores['structure_perturbation'] = perturbation_list

        # Compute final scores
        final_scores = []
        for i in range(len(df)):
            if df.iloc[i]['mutant'] == 'wild_type':
                final_scores.append(0.0)
            else:
                # Base score from modalities (sequence, MSA, WT structure)
                score = sum(modality_weights[mod] * modality_scores[mod][i]
                           for mod in active_modalities)

                # ADD MT structure-conditioned score (NEW - truly uses MT structure)
                if 'mt_structure' in mt_structure_scores:
                    mt_score = mt_structure_scores['mt_structure'][i]
                    score += lambda_mt_structure * mt_score

                # Legacy: Subtract structure perturbation penalty
                if 'structure_perturbation' in perturbation_scores:
                    perturbation = perturbation_scores['structure_perturbation'][i]
                    score -= lambda_perturbation * perturbation

                final_scores.append(score)

        return final_scores
