#!/usr/bin/env python3
"""
Sequence windowing utilities for handling long protein sequences.
"""

from typing import List, Tuple


def get_optimal_windows(mutation_positions: List[int], seq_len_wo_special: int,
                        model_window: int = 1024, overlap: float = 0.2) -> List[Tuple[int, int]]:
    """
    Select optimal sequence windows for long sequences with overlapping regions.

    Args:
        mutation_positions: List of mutation positions (0-indexed)
        seq_len_wo_special: Length of sequence without special tokens
        model_window: Maximum window size
        overlap: Fraction of window to overlap (0.2 = 20% overlap)

    Returns:
        List of (start, end) window positions
    """
    if seq_len_wo_special <= model_window:
        return [(0, seq_len_wo_special)]

    windows = []
    overlap_size = int(model_window * overlap)
    stride = model_window - overlap_size

    # For each mutation, ensure it's covered by at least one window
    for mut_pos in mutation_positions:
        if mut_pos < 0 or mut_pos >= seq_len_wo_special:
            continue

        # Find windows that cover this mutation
        covered = False
        for start, end in windows:
            if start <= mut_pos < end:
                covered = True
                break

        if not covered:
            # Create a window centered on the mutation
            half_window = model_window // 2
            start = max(0, mut_pos - half_window)
            end = min(seq_len_wo_special, start + model_window)

            # Adjust if we're at the sequence end
            if end == seq_len_wo_special:
                start = max(0, end - model_window)

            windows.append((start, end))

    # Sort and merge overlapping windows if needed
    windows = sorted(set(windows))

    # If no mutations or fallback, use sliding windows
    if not windows:
        for start in range(0, seq_len_wo_special, stride):
            end = min(start + model_window, seq_len_wo_special)
            windows.append((start, end))
            if end >= seq_len_wo_special:
                break

    return windows


def get_optimal_window(mutation_position_relative: int, seq_len_wo_special: int, model_window: int = 1024) -> Tuple[int, int]:
    """Select optimal sequence window for long sequences (backward compatibility)."""
    windows = get_optimal_windows([mutation_position_relative], seq_len_wo_special, model_window)
    return windows[0] if windows else (0, min(seq_len_wo_special, model_window))
