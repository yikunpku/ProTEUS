#!/usr/bin/env python3
"""Animate the paper's multimodal representation and experimental-feedback logic.

These are conceptual diagrams, not quantitative embeddings or trajectories.
Palette and semantics follow Figure 1 in the supplied manuscript. Requires
Pillow and FFmpeg. Run with a Python environment containing Pillow.
"""
from pathlib import Path
import argparse
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS = 1280, 720, 24
C = dict(ink='#2F3133', muted='#77838D', blue='#4383AB', sequence='#6D9EC6',
         msa='#529C96', structure='#C09A65', mutation='#CC6681', line='#CDD8DF',
         pale='#EAF2F7', green='#E8F4EF', gold='#F7EED8', pink='#FBEFF2')
FONT = Path('/System/Library/Fonts/Supplemental/Georgia.ttf')
BOLD = FONT.with_name('Georgia Bold.ttf')


def ease(value):
    p = max(0, min(1, value))
    return p*p*(3-2*p)


def text(d, xy, value, size=27, fill=None, bold=False, anchor='mm'):
    d.text(xy, value, font=ImageFont.truetype(str(BOLD if bold else FONT), size),
           fill=fill or C['ink'], anchor=anchor)


def arrow(d, points, progress=1, color=None, width=3):
    color = color or C['blue']
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    remain = sum(lengths)*max(0, min(progress, 1))
    end, prev = points[0], points[0]
    for a, b, length in zip(points, points[1:], lengths):
        if length == 0:
            continue
        if remain <= 0:
            break
        portion = min(1, remain/length)
        end = (a[0]+(b[0]-a[0])*portion, a[1]+(b[1]-a[1])*portion)
        d.line([a, end], fill=color, width=width)
        prev = a
        remain -= length
    if progress > .02:
        angle = math.atan2(end[1]-prev[1], end[0]-prev[0])
        d.polygon([end,
            (end[0]-13*math.cos(angle-.4), end[1]-13*math.sin(angle-.4)),
            (end[0]-13*math.cos(angle+.4), end[1]-13*math.sin(angle+.4))], fill=color)


def feature(d, x, y, cols, rows, colors, step=20, active=-1):
    for r in range(rows):
        for c in range(cols):
            col = colors[(r*7+c*5+r*c) % len(colors)]
            d.rectangle((x+c*step, y+r*step, x+c*step+step-4, y+r*step+step-4), fill=col)
    if active >= 0:
        d.rectangle((x+active*step-3, y-4, x+(active+1)*step-1, y+rows*step), outline=C['mutation'], width=2)


def architecture(t):
    image = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(image)
    centers = (220, 640, 1060)
    for x, name, color in zip(centers, ('Sequence', 'Evolutionary context', 'Structure'),
                               (C['sequence'], C['msa'], C['structure'])):
        text(d, (x, 68), name, 28, color)
        d.line((x-145, 98, x+145, 98), fill=C['line'], width=1)
    letters = 'M K L V A G F D S T V T'.split()
    changed = t > 1.6
    if changed: letters[3] = 'W'
    for j, letter in enumerate(letters):
        x = 65+j*26
        d.rectangle((x, 146, x+23, 185), fill=C['pink'] if j == 3 else C['pale'])
        text(d, (x+11, 165), letter, 21, C['mutation'] if j == 3 else C['ink'])
    text(d, (180, 218), 'V', 23, C['mutation'])
    arrow(d, [(200,218),(238,218)], color=C['mutation'], width=2)
    text(d, (260,218), 'W',23,C['mutation'])
    feature(d, 501, 131, 12, 4, [C['msa'], '#B6DAD3', '#E8F4EF'], step=24, active=3)
    if changed: d.rectangle((573,131,593,151),fill=C['mutation'])
    # Structure features are conceptual input tokens, not an invented molecular fold.
    feature(d, 921, 131, 12, 4, ['#F7D5AB','#F4C286','#EFAA56'], step=24)
    text(d, (1060, 248), 'Residue features', 21, C['muted'])
    for x, color in zip(centers, (C['sequence'], C['msa'], C['structure'])):
        arrow(d, [(x, 280), (x, 326), (640, 326), (640, 359)], ease((t-1.8)/1.7), color, 2)
    # Position-wise fused tokens, with all three modalities in each microgrid.
    for j in range(12):
        y = 376 + (1-ease((t-2.8-j*.045)/1.3))*24
        x = 360+j*47
        feature(d, x, y, 3, 2, [C['sequence'], C['msa'], '#F4C286'], step=12)
        if j == 3: d.rectangle((x-4, y-4, x+35, y+25), outline=C['mutation'], width=2)
    text(d, (640, 431), 'Joint residue embeddings', 24, C['muted'])
    arrow(d, [(640, 456), (640, 485)], ease((t-4)/.8), C['blue'])
    d.rectangle((455, 490, 825, 560), fill=C['pale'], outline='#91B6D5', width=2)
    text(d, (640, 521), 'MutationFormer', 30, C['blue'], True)
    text(d, (860, 527), '× 12', 24, C['muted'], anchor='lm')
    arrow(d, [(640, 565), (640, 614)], ease((t-5.4)/.9))
    text(d, (640, 648), 'Fitness prediction', 30, C['blue'])
    return image


def feedback(t):
    image = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(image)
    nodes = [(215, 315), (640, 315), (1065, 315)]
    progress = ease(t/4.2)
    for index, (x, y) in enumerate(nodes):
        text(d, (x, 155), ('Nominate', 'Measure', 'Update')[index], 34, C['blue'], True)
    # Ranked candidates: illustrative sequence tokens, as in Figure 1C.
    for j, label in enumerate(('V4W', 'L40F', 'I9D · S33A')):
        y = 240+j*56
        text(d, (87, y+18), str(j+1), 22, C['muted'])
        d.rectangle((113, y, 307, y+37), fill=C['pink'])
        text(d, (210, y+18), label, 23, C['mutation'])
    text(d, (215, 455), '30 variants', 26)
    # Assay plate with a quiet scan, not fabricated measured data.
    d.rounded_rectangle((532, 218, 748, 389), radius=10, outline=C['line'], width=3)
    for r in range(4):
        for c in range(6):
            x,y = 554+c*34,242+r*40
            bright = (r*6+c) < int(max(0, min(1, (t-1.4)/2))*24)
            d.ellipse((x-9,y-9,x+9,y+9), fill='#A7CFC8' if bright else C['green'], outline='#8BAFA7', width=1)
    text(d, (640, 455), 'Experimental activity', 25)
    # Same candidate, prediction moves toward a fixed observed reference.
    d.line((945, 312, 1188, 312), fill=C['line'], width=3)
    for j in range(6): d.line((948+j*47,304,948+j*47,320), fill=C['muted'], width=1)
    px = 985+ease((t-2.2)/1.7)*122
    d.ellipse((px-10,302,px+10,322), fill=C['sequence'], outline=C['blue'], width=2)
    d.polygon([(1145,300),(1157,312),(1145,324),(1133,312)], fill=C['ink'])
    text(d, (985,267), 'Predicted', 20, C['blue'])
    text(d, (1145,267), 'Measured', 20)
    text(d, (1065, 455), 'Target-specific predictor', 24)
    arrow(d, [(340, 312),(491,312)], ease((t-.3)/1.2), width=3)
    arrow(d, [(781, 312),(915,312)], ease((t-1.5)/1.2), width=3)
    arrow(d, [(1065,497),(1065,568),(215,568),(215,495)], ease((t-3)/1.5), width=3)
    text(d, (640, 620), 'Select the next batch', 29, C['blue'])
    return image


def encode(name, seconds, fn, output, preview):
    output.mkdir(parents=True, exist_ok=True)
    fn(seconds*.65).save(output/f'{name}-poster.webp', quality=92)
    if preview: return
    cmd = ['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
           '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
           '-pix_fmt','yuv420p','-movflags','+faststart',str(output/f'{name}.mp4')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(seconds*FPS)): process.stdin.write(fn(i/FPS).tobytes())
    process.stdin.close()
    if process.wait(): raise RuntimeError(f'Encoding failed: {name}')
    print(f'{name}: {seconds}s, {FPS}fps', flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'docs/assets/video')
    ap.add_argument('--preview-only',action='store_true'); args=ap.parse_args()
    encode('architecture',10,architecture,args.output,args.preview_only)
    encode('feedback',6,feedback,args.output,args.preview_only)
