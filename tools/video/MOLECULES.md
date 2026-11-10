# Molecular animation sources

`render_molecules.py` renders each frame in ChimeraX 1.12 from the paper's actual
native molecular scenes. It does not read the old website movies or paper-panel
images. Input files in Dropbox are read-only.

```sh
/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX --notools --nocolor --exit --script 'tools/video/render_molecules.py --asset sacb'
```

Repeat with `--asset aldh`, `--asset protein`, and `--asset attention`. Defaults
are 8 seconds, 24 fps, 1280 × 720, 2× supersampling, white background. Use
`--preview` for frames at 0, 2 and 4 seconds. `--source-root` changes the location
of the paper source folder, and `--frames-dir` changes the scratch frame folder.
On macOS ChimeraX needs its graphics session; the packaged application does not
include the OSMesa renderer required for `--offscreen`.

| Output in `docs/assets/video/` | Molecular input |
| --- | --- |
| `sacb-structure.mp4` | `Main Figure/proteus_figures/figures/RAW/cell_v9/protein_structures/lsc_overview.cxs` |
| `aldh-structure.mp4` | `Main Figure/proteus_figures/figures/RAW/cell_v9/protein_structures/fig3f_aldh.cxs` |
| `protein-state.mp4` | `Main Figure/fig1b_1PGA.pdb` |
| `attention-structure.mp4` | `Main Figure/proteus_figures/figures/RAW/cell_v9/protein_structures/figure1b_structure_1MBA.cxs` and `figures/RAW/cell_v9/attention/attention_30_residues.json` |

Each output has a `-poster.webp` and a `.json` record of source hashes, model
coordinate and secondary-structure hashes, and rendering parameters. MP4s are
H.264 with fast-start. The frames remain in `/private/tmp/proteus-paper-molecules/`.

The enzyme style follows the paper's `scripts/cell_v11/render3d/f45/render_lsc.py`
and `scripts/cell_v11/renders/f45/render_aldh.py`: pale green helices (`#7BD0AC`),
blue strands (`#6B9BD1`), beige coils (`#E6DED3`), 73% transparent gray solvent
excluded surface (`#BFC7D1`), rose site markers (`#C85F6B`), and wine catalytic
residues (`#6B2233`). Native gentle lighting and thin opaque contours are kept;
the viewport, antialiasing and camera animation are configured for video.
Attention uses the same ribbon, gentle lighting and light solvent-excluded
surface. Smaller red markers at the 30 sampled Cα positions encode the
recorded attention scores; both models use the same color and radius mapping.

Only the camera turns, through a closed ±18° arc. Atom coordinates, model
transforms and secondary structure are checked unchanged after rendering.
These clips show reference structures, not molecular dynamics, a reaction
simulation, or measured mutant conformations.

SacB uses WT 1OYG with sucrose aligned from 1PT2. Rose markers are the paper's
resolved evolved positions 34, 240, 328 and 359; G23 is in the signal peptide
and is unresolved in 1OYG, so no spatial marker is invented. ALDH uses the WT
AlphaFold reference AF-P46368-F1, with positions 103, 116, 332 and 358 marked.
The 1PGA clip is an illustrative protein state. The 1MBA clip displays fixed
recorded attention weights while changing the viewing angle.

## Attention comparison

Render `--asset attention-esm2` with the same molecular script, then run
`python tools/video/render_attention_story.py`. The 14-second
`attention-comparison.mp4` begins with ProTEUS and smoothly makes room for the
ESM-2 comparison. Both use the same native reference, camera frames, 30 residue
indices, protein color range 0–0.22 and symmetric matrix range 0–0.4. The actual
30 × 30 matrices and their mean residue scores are retained in
`tools/video/attention-data.json`; an assertion checks that the displayed
scores equal the matrix column means. Blue triangular axis ticks locate G25A, P58A, F91A
and A124G. No values are changed while the layout animates.

## Gauss–Seidel trajectory

`render_projection_molecules.py` separately re-renders the original 7Y9A
trajectory used in the user's slides, from
`/Users/mhg/Projects/protein-rendering/gs-rendering/tmp/cartesian/trajectory.dcd`.
Unlike the reference-structure clips above, these atomic coordinates genuinely
change through recorded sequential projection updates. The original topology,
atom order, frame plan and residual trace are verified before rendering.
This geometry-restoration example starts from Cartesian perturbations with
σ = 0.3 Å and enforces 1–2, 1–3 and 1–4 distance constraints. Its measured
distance-constraint RMS decreases from about 0.430 Å to 0.002594 Å; it is not
a molecular-dynamics simulation or a demonstration of all seven families at once.
The other seven-family motifs summarize the method's broader constraints.

## WT and final variant sequences

Render `--asset sacb-final` to show only the final SacB variant's resolved
positions Q34/G240 on the WT reference, then run
`python tools/video/render_enzyme_sequences.py`. The resulting 8-second
`sacb-sequence.mp4` and `aldh-sequence.mp4` pair the moving WT reference with
aligned WT/final sequence windows and an overview spanning every residue.

The complete strings and original FASTA hashes are saved in
`tools/video/enzyme-sequences.json`. The source filenames are
`Main Figure/fig3_4_protein_structure/Levensrucase_sequence.fasta` (the source's
spelling) and `ALDH_sequence.fasta`. Mutations are validated against the
original residues before the final sequence is constructed. Exactly three
SacB or four ALDH positions differ; all others are unchanged. Window reveals
always show the correct complete sequences, never invented intermediate
sequences. SacB keeps full precursor numbering, including G23. Species names
are omitted because the ALDH FASTA header and manuscript use different species
names; both identify the P46368 reference.
