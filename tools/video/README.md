# ProTEUS project film

The 140-second film is built from the manuscript's native molecular scenes,
landscape mesh, measured data, and newly animated explanatory diagrams.
The original Dropbox sources are read-only. The output contains no cropped PDF
panels or footage from the older dark-background website movies.

The website uses these same clips in its model and experiment tabs. Its
benchmark charts remain interactive Canvas drawings.

## Outputs

- `docs/assets/proteus-overview.mp4`: 140-second, 1920×1080, 24 fps film.
- `docs/assets/proteus-overview-poster.webp`: film cover.
- `docs/assets/proteus-overview.vtt`: English captions.
- `docs/assets/video/`: individual scenes, posters and provenance records.

## Rebuild

Requirements: ChimeraX 1.12, Blender 5.1, Python with Pillow, NumPy and aggdraw, Node with
`@napi-rs/canvas`, FFmpeg with libx264, `cwebp`, and locally installed Georgia fonts.
Fonts and the large native molecular/Blender sources are not redistributed.
The default source root is `/Users/mhg/Dropbox/PROTEUS`.

From the repository root:

```sh
# Repeat for sacb, aldh, protein, attention, attention-esm2 and sacb-final.
/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX --notools --nocolor --exit --script 'tools/video/render_molecules.py --asset sacb'
/Applications/Blender.app/Contents/MacOS/Blender -b --python tools/video/render_landscape.py
node tools/video/render_charts.cjs
python tools/video/render_model_story.py
/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX --notools --nocolor --exit --script 'tools/video/render_projection_molecules.py'
python tools/video/render_structure_story.py
python tools/video/render_attention_story.py
python tools/video/render_guided_exploration.py
python tools/video/render_enzyme_sequences.py
python tools/video/render_wetlab_story.py
python tools/render_project_film.py
```

Use a Python environment containing Pillow. Each renderer supports a preview
mode; see its script header. The film compositor's `SCENES` list controls
durations, source intervals, titles and captions, and generates MP4 chapters
and WebVTT timing from the same edit list. `--preview-only --frames-dir PATH`
exports the composed scenes for visual review without encoding the film.

## Component colors and playback

Component identity persists from the overview into its detail view:
blue `#5290D3` for sequence, MutationFormer and prediction; green `#51AA8E`
for MSA and wet-lab measurement; coral `#DA6457` for structure prediction,
Gauss–Seidel projection and target-specific refitting. These are separately
named roles even where they share a color. Mutation annotations use `#E74C3C`.
AF3 uses slate `#778A9B` throughout the structure comparison. Paper-derived
round palettes, attention scales, molecular element colors and the Figure 2F
comparison palette remain scientific encodings rather than component accents.

The four model clips advance on actual video completion and wrap to the first
clip. Clicking a tab starts that clip and continues the sequence. Manual pause,
viewport visibility and initial reduced-motion preferences are respected.

## Scientific sources

- **Molecules:** [MOLECULES.md](MOLECULES.md) maps each native structure and its
  rendering recipe. The camera moves; atom coordinates and secondary structure
  remain unchanged. Enzyme clips show WT references with selected positions
  marked. G23 is unresolved in SacB's reference and has no invented marker.
- **Attention:** the recorded ProTEUS and ESM-2 weights for the same 30 sampled
  residues in 1MBA determine the marker colors/sizes and symmetric matrices.
  Matched views use fixed shared scales (0–0.22 for mean residue attention;
  0–0.4 for matrix entries). ProTEUS moves left before ESM-2 is revealed.
- **Landscape:** the original organic surface mesh, with three consecutive
  single-mutation decisions and multiple dashed candidate paths per step. The
  surface is conceptual, not a measured fitness or energy surface. The actual
  SacB lineage is WT → Q34N → Q34N:G240W → G23A:Q34N:G240W. Native sequence
  windows show one new change at each step. A fixed grid shows the 30 actual
  nominees, and A/B/C identify three same-parent examples, not the top three
  predictions. Measurements reveal the retained variant. The final batch
  tests 30 candidates across two parents; 18 extend the displayed parent.
  The CSV's later-round `final_score` is the fixed-weight composite used in
  the paper's score–activity analysis. Sorting the assayed batch by that
  column does not recover the actual multi-shortlist nomination order; no
  such rank is displayed. A/B/C identifiers link the surface paths, batch
  grid and single-mutation sequence window. Surface labels use the native
  Blender camera projection.
- **Benchmarks:** `docs/chart-engine.js` supplies the unchanged numerical data
  also used by the live website.
- **Enzyme activity:** `chart-data.json` stores the manuscript CSV values,
  cumulative top-ten selections, and source hashes. Run the chart renderer
  with `--refresh-data --source-root PATH_TO_SUBMISSION_SCRIPTS` to reimport.
- **Enzyme sequences:** `enzyme-sequences.json` contains full WT/final strings
  validated against the supplied FASTAs. Sequence clips show aligned windows
  beside WT structural geometry, not predicted mutant structures.
- **Model diagrams:** `render_model_story.py` unfolds the whole original
  protein ribbon, including its left helix, into a fine blue–green–coral rounded hexagonal circuit
  with directional arcs, preserving the sharp original symbol. Boxed nodes show mutation-effect
  prediction, wet-lab measurement and target-specific refitting. It animates clustered
  sequence/MSA/structure inputs, mutations, MutationFormer and the joint
  embedding space. The space is a learned representation across protein
  backgrounds; aligned residue features themselves move into the 2D plane
  using one perspective transform. The coordinates are illustrative. The main
  structure box follows Figure 1E: “Fast 3D prediction / Physics-constrained”.
  Methods additionally specifies ESM3 for structure feature encoding; the GS
  predictor and that encoder are distinct stages. Pretraining uses
  a corpus of 1,322,111 variant measurements spanning 161 protein backgrounds.
  This is the sum of the 182 assay counts in the native Figure 2A CSV
  (`RAW/cell_v9/benchmark_data/fig1/fig1c_scatter.csv`), not the number of
  gradient-training examples: the paper partitions the corpus 3:1:1 and
  exact post-cleaning split counts are not supplied.
- **Structure projection:** `render_projection_molecules.py` renders the
  original 355-frame 7Y9A GS trajectory underlying the user's slide movies,
  using the same molecular palette as the other clips. Global and local views
  use the same fixed camera orientation, with an exact region box and connectors.
  Active pairs show sequential corrections; the distance-constraint RMS is
  recomputed from the trajectory, decreasing from 0.430 Å to 0.002594 Å.
  The example restores 1–2/1–3/1–4 distances after a σ = 0.3 Å Cartesian
  perturbation; it does not claim all seven constraint families were exercised.
  The final coordinates remain on screen for 2.2 seconds.
  `render_structure_story.py` shows AF3's 200 denoising steps and ProTEUS's
  two steps finishing in the same animation interval. A separate differentiable
  Gauss–Seidel module connects upward to the post-trained model. Our predictor
  is initialized from Boltz-1, with denoising loss on projected coordinates.
  Seven newly drawn ball-and-stick motifs show all specified constraint
  families; validity is limited to these checks. No slide screenshots are used.
- **Wet-lab loop:** `render_wetlab_story.py` links nomination, build/measure,
  and refitting the target predictor. ProTEUS representations stay fixed.
  Both recursive-improvement clips reproduce the paper's **cumulative top ten**
  activity distributions (SacB Figure 4C, ALDH Figure 5B): at round r, select
  the ten highest measured activities from all variants assayed through r.
  Each dot is one variant mean; an earlier variant may appear in several
  snapshots. The 90/150 totals are distinct variants tested, not plotted dots.
  Violins are Gaussian KDEs with bandwidth factor 0.35, restricted to the
  observed range, using the paper's horizontal point packing and colors.
  No per-batch maximum curve, cumulative-best line, median, or boxplot is added.
  ALDH R3 retains the earlier 6.08× lead in its cumulative top ten. This replaces
  the previous per-batch distribution view, whose statistic differed from the
  manuscript figure.

## Story and comparisons

The edit is 140 seconds: world model → joint embedding → matched attention
comparison → single-mutation exploration → constrained structure prediction
and speed → scaling law → zero-shot prediction → budget-matched simulated
evolution → SacB → ALDH → closing loop. The opening world-model loop
establishes wet-lab feedback before both enzyme examples. Web chapter buttons match the compositor's start times.

The zero-shot panel combines Figures 2E and 2F: all 15 model means/SEMs over
19 assays excluded from pretraining, plus first-round selection results over
13 held-out single-mutant assays, without target fitness labels. The right
panel shows Spearman, AUROC and AUPR with mean ± SEM, and a top-K selection
curve. Bar axes follow Figure 2F exactly: Spearman .05–.35, AUROC .52–.72,
and AUPR .50–.76, with explicit broken baselines. Significance stars are
omitted as requested; the original tests remain in `selection-metrics.json`
alongside source rows and hashes. The webpage exposes these results in
separate tabs as well. `zero-shot-data.json` records all
285 cells, including four missing/undefined correlations; the original
published averaging convention fills these with zero. Zero-shot scoring uses
sequence/MSA likelihoods with learned modality weights, not the full neural
fitness head. The selection comparison uses the paper’s MULTI-evolve rank
fusion adaptation, with hits defined as at-or-above the assay mean. At most
one substitution is selected per residue position.
The evolution comparison is Figure 3G: 17 multi-mutant DMS datasets, 30 initial
single-mutant observations and five guided batches of 30, with the same
candidate library and nomination budget. This is an **in silico** comparison,
separate from ProTEUS's wet-lab experiments. The cited baselines are
[EVOLVEpro, Science 2025](https://doi.org/10.1126/science.adr6006) and
[MULTI-evolve, Science 2026](https://doi.org/10.1126/science.aea1820).

The film has no narration or music. Short transitions and brief holds preserve
time to read each result. Original website assets remain available in
`docs/assets/` for comparison.

## Narrative and verification

The two user-authored positioning notes in `notes/` guide the scientific scope:
mutation-sensitive prediction and accumulated wet-lab feedback. The user's
subsequent direction replaces question hooks with declarative scientific
titles. The opening and closing ribbon-loop animation carries this theme.
Each self-titled scene fills the video frame, without repeated caption prose.
Website chapter offsets are 0, 37, 52, 76, 103 and 114 seconds.

The structure scene receives 21 seconds, and the three-step search receives
15 seconds. Architecture, pretraining and structure scenes hold the complete
pipeline for about three seconds before the local explanation. Each local
component has a reading hold; pretraining occupies 9 seconds and the closing
loop occupies 13 seconds, returning to the same three-component opening view.
The complete film is 140 seconds (3360 frames at 24 fps). Browser
verification uses an isolated headless Chrome session, with user approval.

The webpage uses the palette from minghaoguo.com: dark gray text, blue links,
coral emphasis, and blue/green/coral modality accents. DOM text and interactive
Canvas labels use only 18, 24 and 40 CSS pixels, including on narrow screens.
Video annotations retain their native image-space typography.
