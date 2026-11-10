"""Animate the original paper landscape mesh, without changing Dropbox sources.

Run with Blender 5.1 in background mode:
  Blender -b --python tools/video/render_landscape.py -- --preview
  Blender -b --python tools/video/render_landscape.py

This is illustrative state-space exploration, not fitted experimental data.
The mesh, height coloring, lighting, and camera derive from the paper sources;
Four state markers and three selected single-mutation steps simplify Figure 1B.
Candidate routes are schematic alternatives, not predicted numeric fitness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]
RAW = Path('/Users/mhg/Dropbox/PROTEUS/Main Figure/proteus_figures')
SOURCE = RAW / 'energy_landscape/revision5_brighter_blue/landscape_blue_gold_brighter.blend'
OUTPUT = ROOT / 'docs/assets/video/landscape.mp4'
FRAMES = Path('/private/tmp/proteus-landscape-frames')
FPS, COUNT = 24, 288


def linear_hex(value):
    rgb = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb) + (1,)


def material(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    e = nodes.new('ShaderNodeEmission')
    e.inputs['Color'].default_value = linear_hex(color)
    output = nodes.new('ShaderNodeOutputMaterial')
    m.node_tree.links.new(e.outputs[0], output.inputs['Surface'])
    return m


def curve(name, points, mat, radius=.027):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 1
    data.bevel_depth = radius
    data.bevel_resolution = 3
    data.use_fill_caps = True
    spline = data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    data.materials.append(mat)
    return obj


def smooth(t):
    t = max(0, min(1, t))
    return t * t * (3 - 2 * t)


def main():
    global SOURCE, FRAMES, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--engine', choices=['EEVEE', 'CYCLES'], default='EEVEE')
    parser.add_argument('--source', type=Path, default=SOURCE, help='Original paper landscape .blend')
    parser.add_argument('--frames-dir', type=Path, default=FRAMES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    SOURCE, FRAMES, OUTPUT = args.source.resolve(), args.frames_dir.resolve(), args.output.resolve()
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
    print('LANDSCAPE_SETUP loaded', flush=True)
    scene = bpy.context.scene
    surf = bpy.data.objects['Energy landscape — organic light blue surface']
    mesh_digest = hashlib.sha256(b''.join(struct.pack('3f', *v.co) for v in surf.data.vertices)).hexdigest()
    for obj in list(scene.objects):
        if obj.name.startswith(('Evolution path ', 'Node ', 'Decision ')):
            bpy.data.objects.remove(obj, do_unlink=True)

    # These are the exact pale palette stops in cell_v11/restyle.py:228.
    palette = [(0, '8CB2E3'), (.35, 'AFD1EC'), (.6, 'C9E4DA'), (.8, 'F0EAC4'), (1, 'F8DCA2')]
    ramp = surf.data.materials[0].node_tree.nodes['HEIGHT PALETTE — low to high'].color_ramp
    for stop, (position, color) in zip(ramp.elements, palette):
        stop.position = position
        stop.color = linear_hex(color)
    grid = bpy.data.objects['Black Cartesian grid — on surface only']
    border = bpy.data.objects['Organic silhouette — fine black edge']
    ink = material('Paper surface grid | subdued blue grey', '93AFB9')
    grid.data.materials.clear()
    grid.data.materials.append(ink)
    grid.data.bevel_depth = .007
    border.data.materials.clear()
    border.data.materials.append(ink)
    border.data.bevel_depth = .009
    # Subtle shadow as in the paper, composed directly over pure white.
    group = scene.compositing_node_group
    group.nodes['Shadow strength'].inputs[1].default_value = .12
    output = next(n for n in group.nodes if n.type == 'GROUP_OUTPUT')
    prior = output.inputs['Image'].links[0].from_socket
    white = group.nodes.new('CompositorNodeAlphaOver')
    white.inputs['Factor'].default_value = 1
    white.inputs['Background'].default_value = (1, 1, 1, 1)
    group.links.new(prior, white.inputs['Foreground'])
    group.links.new(white.outputs[0], output.inputs['Image'])
    if args.engine == 'EEVEE':
        scene.render.engine = 'BLENDER_EEVEE'
        # Cycles' shadow catcher pass is not available in Eevee. Keep the same
        # geometry/material/light rig and render directly over a white world.
        for obj in scene.objects:
            if obj.name.startswith('SHADOW CATCHER'):
                obj.hide_render = True
        render_layer = next(n for n in group.nodes if n.type == 'R_LAYERS')
        group.links.new(render_layer.outputs['Image'], white.inputs['Foreground'])
        print('LANDSCAPE_SETUP Eevee', flush=True)

    world_vertices = [surf.matrix_world @ v.co for v in surf.data.vertices]
    bvh = BVHTree.FromPolygons(world_vertices, [list(p.vertices) for p in surf.data.polygons])
    print('LANDSCAPE_SETUP surface acceleration ready', flush=True)

    def on_surface(x, y, lift=.065):
        point, normal, index, distance = bvh.ray_cast(Vector((x, y, 20)), Vector((0, 0, -1)))
        if point is None:
            raise ValueError(f'Path is outside the original surface: {x}, {y}')
        return point + Vector((0, 0, lift))

    # Schematic state positions lie on the original mesh; they carry no numeric fitness.
    coordinates = [(-2.70, -3.85), (-.80, -2.40), (.60, -.80), (1.77, 1.47)]
    alternatives = [[(-2.3, -1.7), (-1.8, -3.1)],
                    [(-.90, -.70), (1.30, -2.00)],
                    [(.40, 1.50), (2.40, .35)]]
    states = [on_surface(*xy, .16) for xy in coordinates]
    charcoal = material('Selected transition | paper charcoal', '344D58')
    blue = material('Predicted transition | paper blue', '4B97BF')
    pale = material('State | pale blue', 'A9D8EB')
    white_mat = material('Untaken state | white', 'FFFFFF')

    def path(a, b, bend=.0):
        result = []
        for i in range(121):
            t = i / 120
            x = (1 - t) * a[0] + t * b[0] + bend * math.sin(math.pi * t)
            y = (1 - t) * a[1] + t * b[1]
            result.append(on_surface(x, y))
        return result

    paths = [path(coordinates[i], coordinates[i+1], [.10, -.15, -.20][i]) for i in range(3)]
    measured = [curve(f'Selected step {i + 1}', pts, charcoal) for i, pts in enumerate(paths)]
    predicted = []
    for j in range(3):
        branches = []
        for branch, pts in enumerate([paths[j]] + [path(coordinates[j], xy, -.12) for xy in alternatives[j]]):
            parts = []
            for k in range(0, 116, 11):
                parts.append(curve(f'Candidate step {j + 1} branch {branch} dash {k}', pts[k:k + 7], blue, .021))
            branches.append(parts)
        predicted.append(branches)

    def disc(name, position, mat, radius):
        # A world-positioned camera-facing circle, as in the original figure.
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=radius, fill_type='NGON', location=position)
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(mat)
        return obj

    markers = []
    for i, position in enumerate(states + [on_surface(*xy, .16) for pair in alternatives for xy in pair]):
        outside = disc(f'State {i} outline', position, blue, .125)
        inside = disc(f'State {i} fill', position, pale if i < 4 else white_mat, .093)
        markers.append((outside, inside))
    print('LANDSCAPE_SETUP annotations ready', flush=True)

    camera = scene.camera
    camera_start = camera.location.copy()
    target = Vector((0, 0, 1.35))
    camera.data.ortho_scale = 18.2
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_animated_seed = False
    scene.cycles.seed = 7
    scene.render.use_persistent_data = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        if any(d.type == 'METAL' for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == 'METAL'
            scene.cycles.device = 'GPU'
    except (RuntimeError, TypeError):
        pass
    print('LANDSCAPE_SETUP render ready', scene.render.engine, flush=True)

    def frame(f):
        t = (f - 1) / FPS
        angle = math.radians(-3.5 + 7 * smooth(t / 12))
        camera.location = target + Matrix.Rotation(angle, 3, 'Z') @ (camera_start - target)
        camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        toward = camera.rotation_euler.to_matrix() @ Vector((0, 0, 1))
        for i, (outside, inside) in enumerate(markers):
            appear = [0, 3.35, 7.35, 11.35][i] if i < 4 else 4*((i-4)//2)+1.1
            scale = .001 + .999 * smooth((t - appear) / .4)
            for obj in (outside, inside):
                obj.rotation_euler = camera.rotation_euler
                obj.scale = (scale, scale, scale)
                obj.hide_render = i >= 4 and t >= 4*((i-4)//2)+3.65
            inside.location = outside.location + toward * .007
        for j, obj in enumerate(measured):
            obj.data.bevel_factor_end = max(.00001, smooth((t - (4*j+2.65)) / .95))
            obj.hide_render = t < 4*j+2.65
        for j, branches in enumerate(predicted):
            for branch, parts in enumerate(branches):
                progress = smooth((t - (4*j+.35+branch*.12)) / 1.05)
                for k, obj in enumerate(parts):
                    obj.hide_render = k / len(parts) > progress or progress == 0 or t >= 4*j+3.65
        scene.frame_set(f)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    selected = [36, 85, 132, 181, 228, 277] if args.preview else range(args.start, COUNT + 1)
    for f in selected:
        frame(f)
        scene.render.filepath = str(FRAMES / f'landscape_{f:04d}.png')
        print(f'LANDSCAPE_RENDER {f}', flush=True)
        bpy.ops.render.render(write_still=True)
        print(f'LANDSCAPE_FRAME {f}/{COUNT}', flush=True)
    assert mesh_digest == hashlib.sha256(b''.join(struct.pack('3f', *v.co) for v in surf.data.vertices)).hexdigest()
    manifest = {
        'source': str(SOURCE), 'surface_vertices_sha256': mesh_digest,
        'geometry_preserved': True, 'palette_source': 'scripts/cell_v11/restyle.py:228',
        'concept': 'Illustrative state-space exploration, not measured fitness data.',
        'original_scene_changes': ['Hide historical 23-node tree', 'Pale paper palette', 'Three selected steps; three dashed candidates from every current state', '7-degree camera arc'],
        'selected_lineage': ['WT', 'Q34N', 'Q34N:G240W', 'G23A:Q34N:G240W'],
        'coordinates_are_conceptual': True,
        'resolution': [1280, 720], 'fps': FPS, 'frames': COUNT, 'engine': scene.render.engine,
    }
    (OUTPUT.parent / 'landscape.source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not args.preview:
        subprocess.run(['ffmpeg', '-y', '-framerate', str(FPS), '-i', str(FRAMES / 'landscape_%04d.png'),
                        '-frames:v', str(COUNT), '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUTPUT)], check=True)
        poster = OUTPUT.with_name(OUTPUT.stem + '-poster.webp')
        subprocess.run(['cwebp', '-quiet', '-q', '90', str(FRAMES / f'landscape_{COUNT:04d}.png'),
                        '-o', str(poster)], check=True)
        print(f'LANDSCAPE_COMPLETE {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
