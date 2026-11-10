"""Render true 3D molecular loops from the paper's native ChimeraX scenes.

Run with ChimeraX 1.12 (macOS requires its graphics session):
  ChimeraX --notools --nocolor --exit --script 'tools/video/render_molecules.py --asset sacb'

Inputs are read-only. Coordinates, model transforms and secondary structure are
never edited. Only the orthographic camera moves. Rose spheres mark selected
positions on the WT reference, not experimentally measured mutant structures.
The style is adapted from the paper's cell_v11 render_lsc.py / render_aldh.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
from chimerax.atomic.molsurf import MolecularSurface
from chimerax.geometry import Place
from chimerax.graphics.opengl import Silhouette

ROOT = Path(__file__).resolve().parents[2]
P = dict(helix='#7BD0AC', strand='#6B9BD1', coil='#E6DED3', shell='#BFC7D1',
         rose='#C85F6B', wine='#6B2233', ligand='#F2B632', nad='#D9962F')
NAMES = {'sacb': 'sacb-structure', 'sacb-final': 'sacb-final-reference', 'aldh': 'aldh-structure',
         'protein': 'protein-state', 'attention': 'attention-structure',
         'attention-esm2': 'attention-esm2-structure'}


def command(text):
    return run(session, text, log=False)


def digest_models():
    return [{'name': m.name, 'coordinates_sha256': hashlib.sha256(m.atoms.coords.tobytes()).hexdigest(),
             'secondary_structure_sha256': hashlib.sha256(m.residues.ss_types.tobytes()).hexdigest(),
             'model_matrix': m.position.matrix.tolist(), 'residues': m.num_residues}
            for m in session.models.list(type=AtomicStructure)]


def opaque_outline(self, render):
    # Same rule as the paper: contour opaque cartoon/atoms, never the SES shell.
    fb = render.pop_framebuffer()
    render.current_framebuffer().copy_from_framebuffer(fb, depth=False)


def cartoon(spec='protein', teal=False):
    command(f'show {spec} cartoons; transparency {spec} 0 target c')
    command(f'color {spec} {P["helix"] if teal else P["coil"]} target c')
    command(f'color {spec} & helix {P["helix"]} target c')
    command(f'color {spec} & strand {P["helix"] if teal else P["strand"]} target c')
    command(f'cartoon style {spec} modeHelix default xsection oval width 1.5 thickness 0.26')
    command(f'cartoon style {spec} & coil thickness 0.26')
    command(f'cartoon style {spec} & strand arrows true width 1.7 thickness 0.24')


def lighting():
    for text in ['set bgColor white', 'lighting gentle',
                 'lighting multiShadow 32 msMapSize 128 msDepthBias 0.05',
                 'lighting ambientIntensity 0.80 intensity 0.75 fillIntensity 0.20',
                 'lighting direction 1,-1,-1 fillDirection -0.4,-0.2,-1',
                 'lighting shadows false depthCue false', 'material dull',
                 'material reflectivity 0.80 ambientReflectivity 0.80 specularReflectivity 0.06 exponent 20 transparentCastShadows false',
                 'graphics silhouettes true width 0.70 color #3A4A55 depthJump 0.01']:
        command(text)


def residue_atoms(model, numbers, ca=False):
    return [a for r in model.residues if r.number in numbers and r.polymer_type == r.PT_AMINO
            for a in r.atoms if (a.name == 'CA' if ca else a.element.name != 'H')]


def setup_surface(model, evolved):
    spec = '#' + model.id_string + '&protein'
    command(f'surface {spec} enclose {spec} probeRadius 1.4 gridSpacing 0.4')
    command(f'color {spec} {P["shell"]} target s; transparency {spec} 73 target s')
    surfaces = session.models.list(type=MolecularSurface)
    evo_points = np.array([a.scene_coord for a in residue_atoms(model, evolved)])
    cached = []
    for surface in surfaces:
        vertices = surface.scene_position * surface.vertices
        colors = surface.vertex_colors
        if colors is None:
            colors = np.tile(np.asarray(surface.color, np.uint8), (len(vertices), 1))
        colors = colors.astype(float)
        counts = np.array(cKDTree(surface.atoms.coords).query_ball_point(surface.vertices, 8.0, return_length=True), float)
        low, high = np.percentile(counts, [5, 95])
        burial = np.clip((counts - low) / max(high - low, 1e-6), 0, 1)
        colors[:, :3] *= (1 - .40 * burial)[:, None]
        if len(evo_points):
            distance, _ = cKDTree(evo_points).query(vertices)
            weight = np.clip((3.0 - distance) / 1.8, 0, 1)
            weight = weight * weight * (3 - 2 * weight)
            rose = np.array([200, 95, 107], float)
            colors[:, :3] = colors[:, :3] * (1-weight[:, None]) + rose * weight[:, None]
            colors[:, 3] = np.maximum(colors[:, 3], 255*.55*weight)
        cached.append((surface, vertices, colors))
    return cached


def clear_site_windows(cached, sites):
    camera = session.main_view.camera
    matrix = camera.position.matrix
    right, up, direction, position = matrix[:, 0], matrix[:, 1], -matrix[:, 2], matrix[:, 3]
    for surface, vertices, base_colors in cached:
        colors = base_colors.copy()
        relative = vertices - position
        vx, vy, vz = relative @ right, relative @ up, relative @ direction
        for point, radius, transparency in sites:
            p = point - position
            px, py, pz = p @ right, p @ up, p @ direction
            distance = np.hypot(vx-px, vy-py)
            weight = np.clip((radius-distance)/(.4*radius), 0, 1) * (vz < pz-.3)
            target = np.minimum(colors[:, 3], 255*(1-transparency/100))
            colors[:, 3] = colors[:, 3]*(1-weight) + target*weight
        surface.vertex_colors = np.round(colors).astype(np.uint8)


def configure(args):
    paper = args.source_root / 'Main Figure/proteus_figures'
    native = paper / 'figures/RAW/cell_v9/protein_structures'
    command('close all')
    sources = []
    enzyme = args.asset in ('sacb', 'sacb-final', 'aldh')
    is_sacb = args.asset.startswith('sacb')
    is_attention = args.asset.startswith('attention')
    if enzyme or is_attention:
        filename = 'lsc_overview.cxs' if is_sacb else (
            'figure1b_structure_1MBA.cxs' if is_attention else 'fig3f_aldh.cxs')
        source = native / filename
    else:
        source = args.source_root / 'Main Figure/fig1b_1PGA.pdb'
    sources.append(source)
    command(f'open "{source}"')
    models = session.models.list(type=AtomicStructure)
    model = models[0]
    original = digest_models()
    command('surface close; clip off; hide atoms; hide pbonds; hide cartoons')
    for structure in models:
        for group in structure.pbg_map.values():
            if group.name == 'chain trace':
                group.display = False
    command('camera ortho')
    cartoon()
    sites, cached, annotations = [], [], {}
    if enzyme:
        evolved = ((34, 240) if args.asset == 'sacb-final' else (34, 240, 328, 359)) if is_sacb else (103, 116, 332, 358)
        catalytic = (86, 247, 342) if is_sacb else (262, 301)
        ev, cat = ','.join(map(str, evolved)), ','.join(map(str, catalytic))
        command(f'color #1:{ev} {P["rose"]} target c; color #1:{cat} {P["wine"]} target c')
        command(f'cartoon #1:{ev},{cat} suppressBackboneDisplay false')
        command(f'show #1:{ev}@CA atoms; style #1:{ev}@CA sphere; size #1:{ev}@CA atomRadius 1.7; color #1:{ev}@CA {P["rose"]} target a')
        command(f'show #1:{cat} & sidechain atoms; style #1:{cat} stick; size #1:{cat} stickRadius 0.30; color #1:{cat} {P["wine"]} target a')
        for atom in residue_atoms(model, evolved, ca=True):
            sites.append((np.array(atom.scene_coord), 3.8, 93))
        for atom in residue_atoms(model, catalytic):
            if atom.name not in ('N', 'CA', 'C', 'O'):
                sites.append((np.array(atom.scene_coord), 3.5, 92))
        if is_sacb:
            command('show #1 & ions atoms; style #1 & ions sphere; size #1 & ions atomRadius 1.1; color #1 & ions #A0A8AD')
            command(f'show #2 & ~H atoms; style #2 stick; size #2 stickRadius 0.42; color #2 & C {P["ligand"]} target ab; color #2 & O #E0201B target ab')
            for atom in models[1].atoms:
                if atom.element.name != 'H':
                    sites.append((np.array(atom.scene_coord), 5.0, 93))
            annotations['unresolved_site'] = 'G23 lies in the signal peptide and has no coordinates in 1OYG.'
            annotations['structure'] = 'WT reference 1OYG; sucrose transferred by alignment from 1PT2.'
        else:
            command(f'color #1:240-245 {P["nad"]} target c')
            annotations['structure'] = 'WT AlphaFold reference AF-P46368-F1, 506 residues.'
        cached = setup_surface(model, evolved)
        annotations.update(evolved_positions=evolved, catalytic_positions=catalytic)
    elif is_attention:
        command('cartoon protein suppressBackboneDisplay false')
        attention_source = paper / 'figures/RAW/cell_v9/attention/attention_30_residues.json'
        sources.append(attention_source)
        attention = json.loads(attention_source.read_text())
        attention_model = 'esm2' if args.asset == 'attention-esm2' else 'proteus'
        scores = dict(zip(attention['positions'], attention['models'][attention_model]['scores']))
        palette = np.array([[253,241,240], [236,140,147], [199,52,70], [142,23,48]], float)
        for position, score in scores.items():
            intensity = float(np.clip(score/.22, 0, 1))
            rgb = [round(np.interp(intensity, [0,.33,.66,1], palette[:, c])) for c in range(3)]
            color = '#%02X%02X%02X' % tuple(rgb)
            # The shared ribbon/shell recipe carries the molecular structure;
            # localized markers carry the exact recorded mean attention. Keep
            # them smaller than the old spheres so helices remain legible.
            radius = .45 + 1.25*math.sqrt(intensity)
            command(f'show #1:{position}@CA atoms; style #1:{position}@CA sphere; size #1:{position}@CA atomRadius {radius}; color #1:{position}@CA {color} target a')
            for atom in residue_atoms(model, (position,), ca=True):
                sites.append((np.array(atom.scene_coord), radius+1.3, 93))
        cached = setup_surface(model, ())
        annotations.update(structure='Myoglobin reference 1MBA.', attention_model=attention_model,
                           attention_source=str(attention_source), attention_scores=scores,
                           mutation_positions=[25, 58, 91, 124],
                           attention_visualization=dict(
                               scale=[0, .22], palette=palette.astype(int).tolist(),
                               encoding='CA marker color and radius; ribbon and shell use the shared molecular palette.',
                               marker_radius_angstrom='0.45 + 1.25 * sqrt(clip(score / 0.22, 0, 1))',
                               shell_transparency_percent=73,
                               local_marker_window_transparency_percent=93))
    else:
        annotations['structure'] = 'Protein G B1-domain WT reference 1PGA; illustrative model state.'
    lighting()
    Silhouette.finish_silhouette_drawing = opaque_outline
    assert original == digest_models(), 'Model coordinates or secondary structure changed'
    # Reframe the existing paper view for a wide movie viewport, with room for rotation.
    camera = session.main_view.camera
    coords = np.array([a.scene_coord for a in model.atoms if a.residue.polymer_type == a.residue.PT_AMINO])
    matrix = np.array(camera.position.matrix, float)
    projected = coords @ matrix[:, :3]
    low, high = projected.min(0), projected.max(0)
    center = matrix[:, :3] @ ((low + high)/2)
    distance = max(np.linalg.norm(matrix[:, 3]-center), 150)
    matrix[:, 3] = center + matrix[:, 2]*distance
    camera.position = Place(matrix=matrix)
    camera.field_width = max((high[0]-low[0])*1.32, (high[1]-low[1])*args.width/args.height*1.24)
    return original, sources, annotations, cached, sites, center, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', required=True, choices=NAMES)
    parser.add_argument('--source-root', type=Path, default=Path('/Users/mhg/Dropbox/PROTEUS'))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'docs/assets/video')
    parser.add_argument('--frames-dir', type=Path)
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--supersample', type=int, default=2)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.frames_dir or Path('/private/tmp/proteus-paper-molecules')/args.asset
    frame_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    original, sources, annotations, cached, sites, center, matrix = configure(args)
    count = round(args.seconds*args.fps)
    indices = [0, count//4, count//2] if args.preview else range(count)
    camera = session.main_view.camera
    axis = matrix[:, 1]/np.linalg.norm(matrix[:, 1])
    cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    for index in indices:
        # A closed, smooth 36-degree viewing arc; no atom motion or synthetic trajectory.
        angle = math.radians(18*math.sin(2*math.pi*index/count))
        rotate = np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)
        current = matrix.copy()
        current[:, :3] = rotate@matrix[:, :3]
        current[:, 3] = center + rotate@(matrix[:, 3]-center)
        camera.position = Place(matrix=current)
        clear_site_windows(cached, sites)
        command(f'save "{frame_dir / f"{index:04d}.png"}" width {args.width} height {args.height} supersample {args.supersample}')
        if index % 24 == 0:
            print(f'FRAME {args.asset} {index}/{count} elapsed={time.monotonic()-started:.1f}s', flush=True)
    assert original == digest_models(), 'Molecular structure changed during camera animation'
    metadata = dict(asset=args.asset, name=NAMES[args.asset], frames=count, fps=args.fps,
                    seconds=args.seconds, dimensions=[args.width,args.height], palette=P,
                    preview=args.preview, rendered_frames=len(indices), supersample=args.supersample,
                    coordinate_integrity_verified=True,
                    motion='Orthographic camera yaw, ±18 degrees; atom coordinates unchanged.',
                    annotations=annotations, models=original,
                    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
                    elapsed_seconds=time.monotonic()-started)
    (args.output_dir/f'{NAMES[args.asset]}.json').write_text(json.dumps(metadata, indent=2))
    if not args.preview:
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-v','error','-y','-framerate',str(args.fps),
                        '-i',str(frame_dir/'%04d.png'),'-frames:v',str(count),'-an','-c:v','libx264',
                        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
                        str(args.output_dir/f'{NAMES[args.asset]}.mp4')], check=True)
        Image.open(frame_dir/'0000.png').convert('RGB').save(
            args.output_dir/f'{NAMES[args.asset]}-poster.webp', quality=92, method=6)
    print('MOLECULE_DONE', args.asset, time.monotonic()-started, flush=True)
    command('exit')


main()
