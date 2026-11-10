#!/usr/bin/env python3
"""Animate a matched ProTEUS / ESM-2 comparison from the paper's exact data.

First run render_molecules.py for attention and attention-esm2. This script
composes their real 3D renders with the stored 30×30 symmetric matrices.
The model, residue order, frame times and numerical color scales stay fixed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS, SECONDS = 1920, 1080, 24, 14
INK, BLUE, MUTED, LINE, ROSE = '#2c3e50', '#3c78cd', '#77838d', '#dce3e9', '#a52c49'
MODEL_BLUE = '#5290d3'
FONT = Path('/System/Library/Fonts/Supplemental/Georgia.ttf')
POSITIONS = (25, 58, 91, 124)
PALETTE = np.array([[253,241,240],[236,140,147],[199,52,70],[142,23,48]], float)


def ease(v):
    v=max(0,min(1,v)); return v*v*(3-2*v)


def font(size, bold=False):
    return ImageFont.truetype(str(FONT.with_name('Georgia Bold.ttf') if bold else FONT), size)


def text(draw, xy, value, size=26, fill=INK, anchor='mm', bold=False):
    draw.text(xy, value, font=font(size,bold), fill=fill, anchor=anchor)


def color(values, maximum):
    values=np.clip(np.asarray(values)/maximum,0,1)
    return np.stack([np.interp(values,[0,.33,.66,1],PALETTE[:,i]) for i in range(3)],axis=-1).round().astype(np.uint8)


def decode_frames(path):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height','-of','json',str(path)]))['streams'][0]
    width,height=probe['width'],probe['height']
    proc=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    result=[]; size=width*height*3
    while True:
        raw=proc.stdout.read(size)
        if not raw: break
        if len(raw)!=size: raise RuntimeError('Incomplete decoded frame')
        # Common crop; all selected native views fit inside this fixed region.
        result.append(Image.frombytes('RGB',(width,height),raw).crop((220,25,1060,690)))
    if proc.wait(): raise RuntimeError('Could not decode molecular clip')
    return result


def heatmap(matrix, positions):
    canvas=Image.new('RGB',(440,465),'white'); d=ImageDraw.Draw(canvas)
    x,y,side=62,47,360
    image=Image.fromarray(color(matrix,.4)).resize((side,side),Image.Resampling.NEAREST)
    canvas.paste(image,(x,y)); d.rectangle((x,y,x+side,y+side),outline='#bac8d1',width=1)
    # Fixed blue ticks identify mutated positions independently of red weights.
    ticks=[1,25,58,91,124,146]
    for pos in ticks:
        index=positions.index(pos); center=(index+.5)*side/len(positions)
        fill=BLUE if pos in POSITIONS else MUTED
        text(d,(x+center,y+side+18),str(pos),17,fill)
        text(d,(x-23,y+center),str(pos),17,fill,anchor='rm')
    for pos in POSITIONS:
        center=(positions.index(pos)+.5)*side/len(positions)
        d.polygon([(x+center,y-4),(x+center-5,y-14),(x+center+5,y-14)],fill=BLUE)
        d.polygon([(x-4,y+center),(x-14,y+center-5),(x-14,y+center+5)],fill=BLUE)
    text(d,(x+side/2,453),'Residue position',20,MUTED)
    text(d,(x+side/2,12),'Symmetric attention',23,INK)
    return canvas


def panel(label, explanation, protein, matrix):
    canvas=Image.new('RGB',(860,590),'white'); d=ImageDraw.Draw(canvas)
    text(d,(430,28),label,37,MODEL_BLUE,bold=True)
    text(d,(430,72),explanation,23,MUTED)
    fitted=protein.resize((430,340),Image.Resampling.LANCZOS)
    canvas.paste(fitted,(0,147))
    canvas.paste(matrix,(420,105))
    text(d,(210,530),'Mean residue attention',22,MUTED)
    return canvas


def legend(draw, image, x, y, maximum, label):
    gradient=np.tile(color(np.linspace(0,maximum,260),maximum)[None,:,:],(17,1,1))
    image.paste(Image.fromarray(gradient),(x,y))
    draw.rectangle((x,y,x+260,y+17),outline=LINE,width=1)
    text(draw,(x,y+37),'0',20,MUTED,anchor='lm')
    text(draw,(x+260,y+37),str(maximum),20,MUTED,anchor='rm')
    text(draw,(x+130,y-25),label,22,INK)


def frame(t, clips, maps):
    canvas=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(canvas)
    text(d,(960,49),'Mutation-resolved attention',39,INK,bold=True)
    text(d,(960,100),'1MBA variant: G25A · P58A · F91A · A124G',27,MUTED)
    shift=ease((t-4.0)/2.2)
    scale=1.25-.25*shift
    left=422+(60-422)*shift
    top=132+(205-132)*shift
    index=int(t*FPS)%len(clips['proteus'])
    first=panel('ProTEUS','With mutation bias',clips['proteus'][index],maps['proteus'])
    first=first.resize((round(860*scale),round(590*scale)),Image.Resampling.LANCZOS)
    canvas.paste(first,(round(left),round(top)))
    reveal=ease((t-6.0)/1.2)
    if reveal>0:
        second=panel('ESM-2','Without explicit mutation bias',clips['esm2'][index%len(clips['esm2'])],maps['esm2'])
        second=Image.blend(Image.new('RGB',second.size,'white'),second,reveal)
        canvas.paste(second,(1000,round(205+25*(1-reveal))))
        d.line((960,213,960,776),fill=LINE,width=1)
    # Identical scales apply before, during and after the layout change.
    legend(d,canvas,405,927,.22,'Attention marker color and size')
    legend(d,canvas,1255,927,.4,'Attention matrix color')
    return canvas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=Path('/Users/mhg/Dropbox/PROTEUS/Main Figure/proteus_figures/figures/RAW/cell_v9/attention/attention_30_residues.json'))
    ap.add_argument('--assets',type=Path,default=ROOT/'docs/assets/video')
    ap.add_argument('--preview-only',action='store_true')
    args=ap.parse_args(); data=json.loads(args.source.read_text())
    positions=data['positions']; assert len(positions)==30
    for name in ('proteus','esm2'):
        matrix=np.array(data['models'][name]['matrix'])
        assert matrix.shape==(30,30) and np.allclose(matrix,matrix.T)
        assert np.allclose(matrix.mean(axis=0),data['models'][name]['scores'])
    clips={name:decode_frames(args.assets/filename) for name,filename in
           [('proteus','attention-structure.mp4'),('esm2','attention-esm2-structure.mp4')]}
    maps={name:heatmap(np.array(data['models'][name]['matrix']),positions) for name in clips}
    args.assets.mkdir(parents=True,exist_ok=True)
    snapshot=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  positions=positions,mutation_positions=POSITIONS,matrix_scale=[0,.4],protein_scale=[0,.22],
                  symmetrization=data['symmetrization'],projection=data['protein_projection'],models=data['models'])
    (ROOT/'tools/video/attention-data.json').write_text(json.dumps(snapshot,indent=2))
    preview=Path('/private/tmp/proteus-attention-story');preview.mkdir(exist_ok=True)
    for t in (1,5,8,13): frame(t,clips,maps).save(preview/f'{t:02}.png')
    frame(10,clips,maps).save(args.assets/'attention-comparison-poster.webp',quality=93,method=6)
    if not args.preview_only:
        process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
            '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(args.assets/'attention-comparison.mp4')],stdin=subprocess.PIPE)
        for i in range(SECONDS*FPS): process.stdin.write(frame(i/FPS,clips,maps).tobytes())
        process.stdin.close()
        if process.wait(): raise RuntimeError('Encoding failed')
    (args.assets/'attention-comparison.source.json').write_text(json.dumps(dict(
        source=str(args.source),source_sha256=snapshot['source_sha256'],data_snapshot='tools/video/attention-data.json',
        duration=SECONDS,fps=FPS,size=[W,H],stages={'ProTEUS': [0,4], 'shift': [4,6.2], 'comparison_reveal':[6,7.2], 'comparison_hold':[7.2,14]},
        molecular_coordinates_unchanged=True,matrix_values_unchanged=True,
        molecular_style='Shared thin ribbon, gray 73% transparent SES shell, gentle lighting; true mean attention shown only by local marker color and radius.',
        molecular_assets=[dict(path=filename,
                              sha256=hashlib.sha256((args.assets/filename).read_bytes()).hexdigest())
                          for filename in ('attention-structure.mp4','attention-esm2-structure.mp4')],
        shared_scales=dict(marker=[0,.22],matrix=[0,.4]),
        model_heading_color=MODEL_BLUE,mutation_tick_color=BLUE,
        mutation_annotation='Fixed blue axis ticks at positions 25, 58, 91 and 124; separate from the red weight scale.'),indent=2))
    print('attention-comparison ready',flush=True)


if __name__=='__main__': main()
