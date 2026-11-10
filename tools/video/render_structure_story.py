#!/usr/bin/env python3
"""Native 21-second structure story: architecture, diffusion, physical geometry.

First render the matched native 7Y9A GS trajectory viewports with
render_projection_molecules.py in ChimeraX. Then use this script with --preview
before encoding. The seven newly drawn vector/ball-stick panels are explanatory
constraint diagrams, not seven measured trajectories. No slide movie or crop is
used. The source structures remain read-only.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from render_model_story import Draw, P, W, H, FPS, AA, overview, ease, flow, bezier, moving_bead, fade, rgb, prepare_inputs, aggdraw

ROOT=Path(__file__).resolve().parents[2]
TEMP=Path('/private/tmp/proteus-structure-story')
MOLECULES=Path('/private/tmp/proteus-projection-molecules')
DURATION=21
STRUCTURE_COLOR=P['coral']
STRUCTURE_FILL=P['cream']
COMPARATOR_COLOR='#778A9B'
COMPARATOR_FILL='#F1F4F6'
CONSTRAINTS=[
    ('Nonbonded inter-chain steric separation',('Nonbonded inter-chain','steric separation')),
    ('Tetrahedral chirality',('Tetrahedral','chirality')),
    ('Double-bond E/Z stereochemistry',('Double-bond E/Z','stereochemistry')),
    ('Carbon double-bond planarity',('Carbon double-bond','planarity')),
    ('Ligand internal-distance geometry',('Ligand internal-','distance geometry')),
    ('Symmetric-chain separation',('Symmetric-chain','separation')),
    ('Inter-chain covalent bonds',('Inter-chain','covalent bonds')),
]


def arrow(d,points,color=P['blue'],width=2.2):
    flow(d,points,1,color,width)


def dashed(d,a,b,color=P['muted'],width=1.2,on=5,off=4):
    length=math.dist(a,b)
    if not length:return
    u=((b[0]-a[0])/length,(b[1]-a[1])/length)
    for q in np.arange(0,length,on+off):
        end=min(q+on,length)
        d.line([(a[0]+u[0]*q,a[1]+u[1]*q),(a[0]+u[0]*end,a[1]+u[1]*end)],color,width)


def ring(d,cx,cy,r,color,width=1.4):
    points=[(cx+r*math.cos(a),cy+r*math.sin(a)) for a in np.linspace(0,2*math.pi,100)]
    d.line(points,color,width)


def ball(d,x,y,r,color):
    # Editable analytic spheres, with the same quiet highlights as the native
    # molecule render. These drawings represent geometry, not measured atoms.
    base=np.asarray(rgb(color),float)
    d.dot(x,y,r,tuple(np.clip(base*.76,0,255).astype(int)))
    for k in range(13,0,-1):
        p=k/14
        light=.22*(1-p)
        shade=base*(.86+.14*(1-p))+(255-base)*light
        d.dot(x-r*.16*(1-p),y-r*.21*(1-p),r*p,tuple(np.clip(shade,0,255).astype(int)))
    d.dot(x-r*.29,y-r*.32,r*.16,fade(color,.36))


def bond(d,a,b,color=P['muted'],width=3,double=False):
    if double:
        dx,dy=b[0]-a[0],b[1]-a[1];length=max(1,math.hypot(dx,dy));nx,ny=-dy/length*2.2,dx/length*2.2
        for sign in (-1,1):d.line([(a[0]+sign*nx,a[1]+sign*ny),(b[0]+sign*nx,b[1]+sign*ny)],color,width*.62)
    else:d.line([a,b],color,width)


def constraint_icon(d,index,cx,cy,scale=1):
    """New canonical geometry diagrams; none is claimed to be a solver frame."""
    def p(x,y):return(cx+x*scale,cy+y*scale)
    def b(x,y,r=8,c=P['blue']):ball(d,*p(x,y),r*scale,c)
    def line(a,z,c=P['muted'],w=3,double=False):bond(d,p(*a),p(*z),c,w*scale,double)
    if index==0:
        for x,col in [(-30,P['blue']),(30,P['teal'])]:
            ring(d,*p(x,0),24*scale,fade(col,.40),1.5*scale);b(x,0,10,col)
        dashed(d,p(-19,0),p(19,0),P['gold'],1.5*scale,4*scale,3*scale)
        line((-41,-14),(-55,-24),P['blue'],2);b(-55,-24,5,P['blue'])
        line((41,14),(55,24),P['teal'],2);b(55,24,5,P['teal'])
    elif index==1:
        line((0,0),(-37,-24));line((0,0),(35,-20))
        d.polygon([p(-3,3),p(3,3),p(15,35),p(-1,39)],P['gold'])
        for q in np.linspace(.15,.92,6):
            x,y=-30*q,30*q;w=7*q
            d.line([p(x-w,y-w*.2),p(x+w,y+w*.2)],P['muted'],1.6*scale)
        b(0,0,10,P['gold']);b(-37,-24,8,P['blue']);b(35,-20,8,P['rose']);b(7,37,7,P['teal']);b(-30,30,6,P['gray'])
    elif index==2:
        line((-15,0),(15,0),double=True)
        for a,z in [((-15,0),(-45,-26)),((-15,0),(-45,26)),((15,0),(45,-26)),((15,0),(45,26))]:line(a,z)
        b(-15,0,9,P['gold']);b(15,0,9,P['gold']);b(-45,-26,8,P['blue']);b(45,26,8,P['blue'])
        b(-45,26,5,P['gray']);b(45,-26,5,P['gray'])
    elif index==3:
        d.polygon([p(-60,22),p(15,-36),p(61,-20),p(-13,38)],P['pale'],fade(P['blue'],.42),1)
        line((0,0),(37,-24),double=True);line((0,0),(-40,-8));line((0,0),(10,31))
        b(0,0,10,P['gold']);b(37,-24,9,P['rose']);b(-40,-8,8,P['blue']);b(10,31,8,P['teal'])
    elif index==4:
        pts=[(-39,12),(-24,-25),(15,-30),(41,0),(19,31)]
        for a,z in zip(pts,pts[1:]+pts[:1]):line(a,z)
        for i,(x,y) in enumerate(pts):b(x,y,8,P['rose'] if i==2 else P['gold'])
        dashed(d,p(*pts[0]),p(*pts[2]),P['blue'],1.5*scale,4*scale,3*scale)
        dashed(d,p(*pts[1]),p(*pts[3]),P['teal'],1.5*scale,4*scale,3*scale)
    elif index==5:
        for shift,col in [(-36,P['blue']),(36,P['teal'])]:
            pts=[p(shift+x,y) for x,y in [(-13,-28),(11,-15),(-10,2),(13,18),(-8,30)]]
            d.line(pts,col,5*scale)
            for x,y in pts:ball(d,x,y,5*scale,col)
            ring(d,*p(shift,0),4*scale,P['gold'],2*scale)
        dashed(d,p(-29,0),p(29,0),P['gold'],1.7*scale,4*scale,3*scale)
    else:
        pts=[(-57,-23),(-32,-8),(-13,7),(13,-7),(36,7),(56,27)]
        for i,(a,z) in enumerate(zip(pts,pts[1:])):line(a,z,P['gold'] if i==2 else P['muted'],4 if i==2 else 3)
        for i,(x,y) in enumerate(pts):b(x,y,8,P['blue'] if i<3 else P['teal'])


def loop_path(cx,cy):
    # The recurrent route returns the denoised coordinates to the next update.
    p=[(cx+125,cy),(cx+163,cy)]
    p+=bezier((cx+163,cy),(cx+194,cy),(cx+194,cy-128),(cx+160,cy-128),26)[1:]
    p.extend([(cx-160,cy-128)])
    p+=bezier((cx-160,cy-128),(cx-194,cy-128),(cx-194,cy),(cx-163,cy),26)[1:]
    p.append((cx-125,cy))
    # The last leg goes through the model, completing one denoising update.
    p.append((cx+125,cy))
    return p


def diffusion_panel(d,t,opacity=1):
    old=d.alpha;d.alpha*=opacity
    d.text(640,67,'Fewer denoising steps',34,STRUCTURE_COLOR,True)
    progress=ease((t-.2)/4.0)
    for cx,name,n,color,fill in [(330,'AlphaFold 3',200,COMPARATOR_COLOR,COMPARATOR_FILL),
                                (950,'ProTEUS',2,STRUCTURE_COLOR,STRUCTURE_FILL)]:
        cy=342
        d.text(cx,149,name,34,color,True)
        path=loop_path(cx,cy)
        d.line(path[:-1],fade(color,.42),2.3)
        # Every tick is one update. This prevents the 200-step loop's true
        # speed from appearing slow or stationary through 24-fps aliasing.
        completed=min(n,int(math.floor(progress*n+1e-8)))
        if n==200:
            # Batch independent native vector ticks into two paths. This is
            # identical geometry without 200 full-canvas flushes per frame.
            for lo,hi,stroke in [(0,completed,color),(completed,n,fade(color,.13))]:
                if hi<=lo:continue
                path_ticks=aggdraw.Path()
                for k in range(lo,hi):
                    x=cx-160+320*(k+.5)/n
                    path_ticks.moveto(*d.point(x,cy-145));path_ticks.lineto(*d.point(x,cy-133))
                ctx=aggdraw.Draw(d.image)
                ctx.path(path_ticks,None,aggdraw.Pen(fade(stroke,d.alpha),.88*d.scale*AA));ctx.flush()
        else:
            for k in range(n):
                x=cx-81+k*162
                d.dot(x,cy-139,7,color if k<completed else fade(color,.18),'white',1)
        if 0<progress<1:
            phase=(progress*n)%1
            # A time trail is only a visualization of the loop's recurrence.
            for tail in range(6):
                moving_bead(d,path,(phase-tail*.035)%1,fade(color,1-tail*.14),4.6-tail*.35)
        arrow(d,[(cx-163,cy),(cx-126,cy)],color,2.3)
        d.rect((cx-125,cy-39,250,78),fill,color,1.8,8)
        d.text(cx,cy,'Diffusion model',27,color,True)
        d.text(cx,435,f'{completed} / {n}',42,color,True)
        d.text(cx,478,'denoising steps',23,P['muted'])
    d.rect((748,565,402,83),STRUCTURE_FILL,STRUCTURE_COLOR,1.7,8)
    d.text(949,592,'Differentiable Gauss–Seidel',24,STRUCTURE_COLOR,True)
    d.text(949,622,'projection',24,STRUCTURE_COLOR,True)
    arrow(d,[(1166,603),(1199,603),(1199,414),(1037,414),(1037,381)],STRUCTURE_COLOR,2.2)
    d.text(1124,524,'Post-trained',21,STRUCTURE_COLOR,True)
    d.alpha=old


@functools.lru_cache(maxsize=1)
def molecular_metadata():
    return json.loads((MOLECULES/'source.json').read_text())


@functools.lru_cache(maxsize=24)
def molecule_image(kind,index):
    return Image.open(MOLECULES/kind/f'{index:04d}.png').convert('RGB')


def molecular_index(t):
    meta=molecular_metadata();index=min(meta['frames']-1,max(0,round(t*meta['fps'])))
    available=[int(k) for k in meta['frames_metadata']]
    return min(available,key=lambda k:abs(k-index))


def physical_page(d,t):
    d.text(640,61,'Physical validity with Gauss–Seidel projection',31,STRUCTURE_COLOR,True)
    index=molecular_index(t)
    meta=molecular_metadata();frame=meta['frames_metadata'][str(index)]
    global_box=(40,132,315,357)
    local_box=(477,148,690,352)
    d.text(197,112,'7Y9A',19.5,P['muted'])
    d.image_at(molecule_image('global',index),*global_box)
    def placed(q,box):return(box[0]+q[0]*box[2],box[1]+q[1]*box[3])
    corners=[placed(q,global_box) for q in frame['region_corners']]
    reveal=ease(t/.4)
    d.alpha=reveal
    d.line(corners+[corners[0]],STRUCTURE_COLOR,1.9)
    # Same orientation, exact local field rectangle; no hand-positioned crop.
    dashed(d,corners[1],(local_box[0],local_box[1]),fade(STRUCTURE_COLOR,.65),1.3,5,5)
    dashed(d,corners[2],(local_box[0],local_box[1]+local_box[3]),fade(STRUCTURE_COLOR,.65),1.3,5,5)
    d.rect(local_box,fill='white',stroke=STRUCTURE_COLOR,width=1.1)
    # Re-paste inside the fine border; the local render itself is unchanged.
    d.image_at(molecule_image('local',index),local_box[0]+1,local_box[1]+1,local_box[2]-2,local_box[3]-2)
    # These markers follow the atom pair actually visited in the original GS
    # log. No target-to-output morph or invented correction is displayed.
    inset=(local_box[0]+1,local_box[1]+1,local_box[2]-2,local_box[3]-2)
    pair=frame['local']['active_pair']
    if len(pair)==2 and all(.02<q[0]<.98 and .02<q[1]<.98 for q in pair):
        a,b=[placed(q,inset) for q in pair]
        dashed(d,a,b,STRUCTURE_COLOR,1.5,3,3)
        for point in (a,b):ring(d,*point,7.5,STRUCTURE_COLOR,1.8)
    d.alpha=1
    phase='Perturbed coordinates' if t<1.2 else ('Projected coordinates' if t>=8.6 else 'Gauss–Seidel projection')
    d.text(823,126,phase,24,STRUCTURE_COLOR,True)
    d.text(823,522,f'Distance constraint RMS  {frame["rms_residual_A"]:.3f} Å',21,P['muted'])
    d.text(640,552,'Method: seven constraint families',17,P['muted'])
    for k,(_,lines) in enumerate(CONSTRAINTS):
        cx=108+k*177
        # Method overview, not a claim that this distance-only demonstration
        # applies or passes every family shown below.
        constraint_icon(d,k,cx,602,.80)
        for j,line in enumerate(lines):d.text(cx,670+j*23,line,16.6,P['ink'])


def frame(t):
    d=Draw()
    if t<3:
        overview(d,4.8)
    elif t<5.4:
        p=ease((t-3)/.8)
        # Focus Fast 3D prediction and its protein, then hold the complete
        # local explanation for 1.3 seconds before changing diagrams.
        z=Draw(d.image,camera=(1+.75*p,-1197.5*p,-15*p),opacity=1-ease((t-5.1)/.15))
        overview(z,4.8,2)
        diffusion_panel(d,0,ease((t-5.25)/.15))
    elif t<10.2:
        diffusion_panel(d,t-5.4)
    else:
        physical_page(d,t-10.2)
    return d.image


def write_sources(output):
    molecules=molecular_metadata()
    source=Path('/Users/mhg/Downloads/ProTEUS_arXiv_updated_authors.pdf')
    source_paths=[source,ROOT/'docs/assets/video/protein-state.mp4',
                  Path('/Users/mhg/Dropbox/PROTEUS/Main Figure/fig1b_1PGA.pdb'),
                  Path(__file__),Path(__file__).with_name('render_projection_molecules.py')]
    metadata=dict(renderer='tools/video/render_structure_story.py',seconds=DURATION,fps=FPS,
        frames=DURATION*FPS,dimensions=[W,H],
        ui_palette=dict(structure=STRUCTURE_COLOR,structure_fill=STRUCTURE_FILL,
                        af3_comparator=COMPARATOR_COLOR,af3_comparator_fill=COMPARATOR_FILL,
                        scope='UI identity accents only; native molecular and constraint-illustration colors are preserved.'),
        layout_seconds=dict(full_pipeline=[0,3],structure_zoom=[3,3.8],focused_structure_hold=[3.8,5.1],
                            transition=[5.1,5.4],diffusion=[5.4,10.2],physical_geometry=[10.2,21],projected_result_hold=[18.8,21]),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths if p.exists()],
        molecules=molecules,constraints=[x[0] for x in CONSTRAINTS],
        scientific_limits=[
            '200-step AlphaFold 3 is a benchmark comparison. The ProTEUS predictor is initialized from Boltz-1, not AF3 weights.',
            'The synchronized counters illustrate denoising update budgets in a common animation window, not measured wall-clock runtime.',
            'Two denoising updates precede the physical projection. Post-training evaluates denoising loss on projected coordinates and differentiates through the projection.',
            'The projection module and upward Post-trained arrow indicate training supervision, not an additional diffusion iteration.',
            'The molecular animation newly renders the original 7Y9A GS trajectory used in ScientificWorldmodel.pptx media37/media38. Every position is a real recorded solver snapshot; coordinates are never interpolated.',
            'The demonstration starts from the source geometry with independent Gaussian Cartesian perturbations of standard deviation 0.3 A, seed 101. It restores 6418 target distances (1–2 bonds, 1–3 angles, 1–4 torsions).',
            'This source restoration demo is distinct from the paper seven-family benchmark. Its final RMS residual is 0.00259441699 A and maximum residual 0.0221581198 A; it is not labelled as passing all seven checks.',
            'The overview rectangle is the orthographic projection of the exact local field of view; the same model and orientation are used in both views.',
            'The seven new vector/ball-stick illustrations explain constraint families. They are not additional molecular measurements or trajectories.',
            'Reported validity is limited to the seven specified geometric and stereochemical checks, satisfied to numerical tolerance in the reported benchmarks.',
            'Neither the old PPTX projection movies nor image197 constraint crops are used.'],
        paper_references=dict(denoising_budgets='PDF p26 Figure 3; p27 caption',
                              post_training='PDF p7; p36–38 Methods',constraints='PDF p37; p52–53 Table S3'))
    (output/'structure-projection.source.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    global MOLECULES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--molecules-dir',type=Path,default=MOLECULES)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/video')
    parser.add_argument('--preview-dir',type=Path,default=TEMP/'previews')
    args=parser.parse_args();MOLECULES=args.molecules_dir
    args.output.mkdir(parents=True,exist_ok=True);args.preview_dir.mkdir(parents=True,exist_ok=True)
    prepare_inputs()
    for t in (1.5,3.6,4.4,6.5,9.8,10.4,11.1,14.5,20.5):
        frame(t).resize((W,H),Image.Resampling.LANCZOS).save(args.preview_dir/f'structure-projection-{t:g}.png')
    # A standalone native constraint sheet supports reuse and scientific review.
    icons=Draw()
    for k,(name,_) in enumerate(CONSTRAINTS):
        x=185+(k%4)*300;y=170+(k//4)*330
        constraint_icon(icons,k,x,y,1.55)
        for j,line in enumerate(CONSTRAINTS[k][1]):icons.text(x,y+100+j*28,line,21,P['ink'])
    icons.image.resize((W,H),Image.Resampling.LANCZOS).save(args.preview_dir/'physical-constraints-native.png')
    write_sources(args.output)
    if args.preview:return
    if molecular_metadata().get('preview'):raise RuntimeError('Render all native molecular frames before encoding.')
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
             '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'structure-projection.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    for i in range(DURATION*FPS):
        proc.stdin.write(frame(i/FPS).resize((W,H),Image.Resampling.LANCZOS).tobytes())
        if i%96==0:print(f'STRUCTURE_STORY {i}/{DURATION*FPS}',flush=True)
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('FFmpeg encoding failed')
    frame(14.5).resize((W,H),Image.Resampling.LANCZOS).save(args.output/'structure-projection-poster.webp',quality=93)
    icons.image.resize((W,H),Image.Resampling.LANCZOS).save(args.output/'physical-constraints-native.webp',quality=95)
    print('STRUCTURE_STORY_DONE',flush=True)


if __name__=='__main__':main()
