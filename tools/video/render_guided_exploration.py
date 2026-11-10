#!/usr/bin/env python3
"""Animate actual SacB nominated batches and three real single-mutation steps.

The fixed 30-point grid is ordered by mutation position, not prediction or
activity. A/B are position-based examples from the same parent; C is the real
retained assay winner. No nomination rank is reconstructed from final_score:
in later rounds it is a fixed-weight score used for the paper's correlation
analysis, not the multi-configuration shortlist procedure that assembled these
batches. The initial round uses zero-shot scores.
The landscape is conceptual; native geometry and camera are unchanged.
"""
from pathlib import Path
import argparse
import csv
import functools
import hashlib
import json
import math
import subprocess
import sys

# Run this same recipe under Blender once to derive exact on-screen candidate
# node locations from the unchanged native mesh and the landscape camera arc.
# Blender -b --python tools/video/render_guided_exploration.py -- --project-nodes
def project_landscape_nodes():
    import bpy
    from mathutils import Matrix, Vector
    from mathutils.bvhtree import BVHTree
    from bpy_extras.object_utils import world_to_camera_view
    root=Path(__file__).resolve().parents[2]
    source=root/'docs/assets/video/landscape.source.json'
    manifest=json.loads(source.read_text())
    bpy.ops.wm.open_mainfile(filepath=manifest['source'])
    scene=bpy.context.scene
    surface=bpy.data.objects['Energy landscape — organic light blue surface']
    vertices=[surface.matrix_world@v.co for v in surface.data.vertices]
    bvh=BVHTree.FromPolygons(vertices,[list(p.vertices) for p in surface.data.polygons])
    # These are the same node coordinates used by render_landscape.py.
    states=[(-2.70,-3.85),(-.80,-2.40),(.60,-.80),(1.77,1.47)]
    alternatives=[[(-2.3,-1.7),(-1.8,-3.1)],[(-.90,-.70),(1.30,-2.00)],[(.40,1.50),(2.40,.35)]]
    world=[]
    for step in range(3):
        points=[]
        # A/B are the two illustrative alternative endpoints. C is the
        # actual assay winner and the next retained state in the native path.
        for x,y in alternatives[step]+[states[step+1]]:
            point,_,_,_=bvh.ray_cast(Vector((x,y,20)),Vector((0,0,-1)))
            assert point is not None
            points.append(point+Vector((0,0,.16)))
        world.append(points)
    camera=scene.camera;start=camera.location.copy();target=Vector((0,0,1.35))
    camera.data.ortho_scale=18.2
    scene.render.resolution_x,scene.render.resolution_y=1280,720
    scene.render.resolution_percentage=100
    projected=[]
    for i in range(288):
        progress=(i/24)/12;progress=max(0,min(1,progress));progress=progress*progress*(3-2*progress)
        angle=math.radians(-3.5+7*progress)
        camera.location=target+Matrix.Rotation(angle,3,'Z')@(start-target)
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        bpy.context.view_layer.update()
        frame=[]
        for points in world:
            screen=[]
            for p in points:
                co=world_to_camera_view(scene,camera,p)
                screen.append([co.x*1280,(1-co.y)*720])
            frame.append(screen)
        projected.append(frame)
    result=dict(source=manifest['source'],landscape_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                landscape_recipe_sha256=hashlib.sha256((root/'tools/video/render_landscape.py').read_bytes()).hexdigest(),
                coordinates_are_conceptual=True,resolution=[1280,720],frames=projected,
                identity_mapping={'A':'first alternative','B':'second alternative','C':'next retained state'})
    (root/'docs/assets/video/guided-landscape-projections.json').write_text(json.dumps(result))
    print('Native candidate projections ready',flush=True)

if '--project-nodes' in sys.argv:
    project_landscape_nodes()
    raise SystemExit(0)

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/assets/video'
RAW=Path('/Users/mhg/Dropbox/PROTEUS')
SCORES=RAW/'Submission_V2/main_figure/scripts/Figure4/data/tables/fig3a_final_score_vs_activity_LSC_source.csv'
FASTA=RAW/'Main Figure/fig3_4_protein_structure/Levensrucase_sequence.fasta'
FRAMES=Path('/private/tmp/proteus-landscape-frames')
W,H,FPS,SECONDS=1920,1080,24,15
INK,BLUE,TEAL,ROSE,MUTED,LINE='#2c3e50','#5290D3','#51AA8E','#E74C3C','#77838d','#dce3e9'
ACTIONS=['Q34N','G240W','G23A']
LEADS=['Q34N','Q34N:G240W','G23A:Q34N:G240W']
COLORS=[BLUE,TEAL,ROSE]

@functools.lru_cache(maxsize=64)
def font(size,bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia'+(' Bold' if bold else '')+'.ttf',size)

def text(d,xy,value,size=32,color=INK,bold=False,anchor='lm'):
    d.text(xy,value,font=font(size,bold),fill=color,anchor=anchor)

def ease(t):
    t=max(0,min(1,t));return t*t*(3-2*t)

def mix(color,a):
    return tuple(round(255+(int(color[i:i+2],16)-255)*a) for i in (1,3,5))

def mutate(sequence,actions):
    result=list(sequence)
    for action in actions:
        index=int(action[1:-1])-1
        assert result[index]==action[0]
        result[index]=action[-1]
    return ''.join(result)

def load_data(scores,fasta):
    sequence=''.join(s.strip() for s in fasta.read_text().splitlines() if not s.startswith('>'))
    assert len(sequence)==473
    raw=list(csv.DictReader(scores.open()))
    assert len(raw)==90
    steps=[]
    for r,lead in enumerate(LEADS):
        batch=[dict(x,activity=float(x['activity'])) for x in raw if int(float(x['round']))==r]
        parents=[set()] if r==0 else [{'Q34N'}] if r==1 else [{'Q34N','G240W'},{'Q34N','S359F'}]
        for row in batch:
            mutations=set(row['variant'].split(':'))
            parent=next(p for p in parents if p<mutations and len(mutations-p)==1)
            row['action']=next(iter(mutations-parent))
        # The grid has no predictive or measured ranking. Position order is
        # fixed before the assay stage and never rearranges after measurement.
        batch.sort(key=lambda row:(int(row['action'][1:-1]),row['action'],row['variant']))
        for index,row in enumerate(batch):row['grid_index']=index
        parent=set(ACTIONS[:r])
        compatible=[x for x in batch if parent<set(x['variant'].split(':')) and len(set(x['variant'].split(':'))-parent)==1]
        winner=next(x for x in batch if x['variant']==lead)
        assert winner['activity']==max(x['activity'] for x in batch)
        representatives=[x for x in compatible if x['variant']!=lead]
        # Span the mutation positions without consulting scores or activities.
        candidates=[representatives[round((len(representatives)-1)/3)],
                    representatives[round(2*(len(representatives)-1)/3)],winner]
        for i,row in enumerate(candidates):row['identity']='ABC'[i]
        before=mutate(sequence,ACTIONS[:r]);after=mutate(sequence,ACTIONS[:r+1])
        assert sum(a!=b for a,b in zip(before,after))==1
        for row in candidates:
            row['preview_sequence']=mutate(before,[row['action']])
            assert sum(a!=b for a,b in zip(before,row['preview_sequence']))==1
        assert 0<winner['grid_index']<29
        steps.append(dict(parent='WT' if r==0 else LEADS[r-1],action=ACTIONS[r],winner=winner,
                          candidates=candidates,batch=batch,before=before,after=after,compatible_nominees=len(compatible)))
    return sequence,steps,raw


def badge(d,center,identity,color,active=False,alpha=1,radius=20):
    x,y=center
    fill=mix(color,alpha) if active else 'white'
    outline=mix(color,alpha)
    if active:d.ellipse((x-radius-6,y-radius-6,x+radius+6,y+radius+6),outline=mix(color,alpha*.25),width=3)
    d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=fill,outline=outline,width=2)
    text(d,(x,y),identity,28,'white' if active else outline,True,'mm')


def frame(t,sequence,steps,logo,projections):
    step=min(2,int(t/5));local=min(5,t-step*5);data=steps[step]
    candidates=data['candidates']
    measured=local>=2.35;selected=local>=3.35
    active_index=min(2,int(local/.72)) if not measured else 2
    active_candidate=candidates[active_index]
    identity=active_candidate['identity'];color=ROSE if selected else BLUE
    image=Image.new('RGB',(W,H),'white')
    native_frame=min(288,int(min(11.99,t*.8)*24)+1)
    landscape=Image.open(FRAMES/f'landscape_{native_frame:04d}.png').convert('RGB')
    landscape=landscape.crop((175,0,1145,590)).resize((1130,687),Image.Resampling.LANCZOS)
    image.paste(landscape,(15,85))
    d=ImageDraw.Draw(image)
    text(d,(75,48),'Mutation-guided exploration',42,bold=True)
    text(d,(75,111),'SacB',32,BLUE,bold=True)
    # A/B/C are example identities, never ranks or a prediction color scale.
    for i,(px,py) in enumerate(projections['frames'][native_frame-1][step]):
        xy=(15+(px-175)*1130/970,85+py*687/590)
        col=ROSE if selected and i==2 else BLUE
        badge(d,xy,'ABC'[i],col,active=i==active_index,alpha=.28 if selected and i!=2 else 1)
    for i in range(7):d.line((83+i*13,750,90+i*13,750),fill='#4B97BF',width=4)
    text(d,(190,750),'Candidate mutations',30,MUTED)
    d.line((632,750,720,750),fill='#344D58',width=4)
    text(d,(740,750),'Retained after assay',30,MUTED)
    d.line((1165,90,1165,780),fill=LINE,width=2)
    image.paste(logo,(1210,45),logo)
    text(d,(1380,100),'ProTEUS',42,BLUE,True)
    text(d,(1845,100),f'Step {step+1} of 3',33,MUTED,anchor='rm')
    text(d,(1220,180),'From '+data['parent'].replace(':',' + '),32,INK)
    if measured:
        text(d,(1220,255),'Measure in wet lab',35,TEAL,True)
    else:
        flow_index=0 if local<.72 else 1 if local<1.44 else 2
        for i,(label,x) in enumerate([('Predict',1220),('Rank',1465),('Select',1695)]):
            text(d,(x,255),label,35,BLUE if i==flow_index else MUTED,i==flow_index)
            if i<2:
                start=x+145;end=start+68
                d.line((start,255,end,255),fill=LINE,width=3)
                d.polygon([(end,255),(end-11,249),(end-11,261)],fill=LINE)
    text(d,(1220,324),'Nominated batch · 30 variants',32,INK,True)
    gx,gy,dx,dy=1258,386,111,51
    examples={row['variant']:row for row in candidates}
    # Every point represents a real tested variant; all 30 retain their fixed
    # position throughout the scene, regardless of score or measured activity.
    for row in data['batch']:
        index=row['grid_index'];xy=(gx+dx*(index%6),gy+dy*(index//6))
        if row['variant'] in examples:
            example=examples[row['variant']];i='ABC'.index(example['identity'])
            col=ROSE if selected and i==2 else BLUE
            badge(d,xy,example['identity'],col,i==active_index,1,radius=18)
        else:
            x,y=xy;d.ellipse((x-8,y-8,x+8,y+8),fill='#c9d2d8')
    if not measured:
        text(d,(1220,718),f'Preview {identity} · '+active_candidate['action'],35,BLUE,True)
    else:
        text(d,(1220,635),'Measured activity (× WT)',30,MUTED)
        for i,row in enumerate(candidates):
            x=1241+i*214
            badge(d,(x,678),row['identity'],ROSE if selected and i==2 else BLUE,selected and i==2,radius=15)
            text(d,(x+30,678),f"{row['activity']:.2f}×",31,ROSE if selected and i==2 else INK)
        if selected:
            text(d,(1220,733),'Retain C · '+data['action']+f" added · {data['winner']['activity']:.2f}× WT",32,ROSE,True)
    label='18 of 30 share this parent' if step==2 else '3 examples from the nominated batch'
    text(d,(1220,779),label,29,MUTED)
    action=active_candidate['action'];pos=int(action[1:-1]);preview=active_candidate['preview_sequence']
    assert preview[pos-1]==action[-1] and data['before'][pos-1]==action[0]
    d.line((75,811,1845,811),fill=LINE,width=2)
    badge(d,(500,849),identity,color,True,radius=18)
    text(d,(535,849),f'{action} · position {pos}',32,color,True)
    text(d,(1845,849),'Retained mutation' if selected else 'Candidate preview',30,MUTED,anchor='rm')
    text(d,(75,904),'Parent protein',33,BLUE,True)
    text(d,(75,971),'Mutated protein' if selected else 'Candidate protein',33,BLUE,True)
    start=max(0,min(len(sequence)-13,pos-7));center=1040;spacing=53
    for row,seq in enumerate((data['before'],preview)):
        y=904+row*67
        for j,index in enumerate(range(start,start+13)):
            x=center+(j-6)*spacing;changed=seq[index]!=data['before'][index]
            if index==pos-1:
                d.rounded_rectangle((x-22,y-26,x+22,y+27),radius=3,
                                    fill=mix(color,.13 if changed else .055),outline=color if changed else None,width=2)
            text(d,(x,y),seq[index],35,color if changed else INK,changed,'mm')
    text(d,(960,1035),'Single mutation',31,MUTED,anchor='mm')
    return image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    parser.add_argument('--scores',type=Path,default=SCORES);parser.add_argument('--fasta',type=Path,default=FASTA)
    args=parser.parse_args();sequence,steps,raw=load_data(args.scores,args.fasta)
    if not (FRAMES/'landscape_0288.png').exists():
        raise FileNotFoundError('First run the three-step render_landscape.py recipe.')
    projection_path=OUT/'guided-landscape-projections.json'
    projections=json.loads(projection_path.read_text())
    assert projections['landscape_recipe_sha256']==hashlib.sha256((ROOT/'tools/video/render_landscape.py').read_bytes()).hexdigest()
    logo=Image.open(ROOT/'docs/assets/proteus-logo.png').convert('RGBA');logo=logo.crop(logo.getbbox())
    logo.thumbnail((145,125),Image.Resampling.LANCZOS)
    OUT.mkdir(parents=True,exist_ok=True)
    preview=Path('/private/tmp/proteus-guided-story');preview.mkdir(exist_ok=True)
    for t in (.4,1.1,1.8,2.7,3.6,4.6,6.1,9.6,11.1,14.6):frame(t,sequence,steps,logo,projections).save(preview/f'{t:g}.png')
    frame(14.6,sequence,steps,logo,projections).save(OUT/'guided-exploration-poster.webp',quality=94)
    if not args.preview:
        p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
            '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'guided-exploration.mp4')],stdin=subprocess.PIPE)
        for i in range(SECONDS*FPS):p.stdin.write(frame(i/FPS,sequence,steps,logo,projections).tobytes())
        p.stdin.close()
        if p.wait():raise RuntimeError('FFmpeg encoding failed')
    (OUT/'guided-exploration.source.json').write_text(json.dumps(dict(
        description=__doc__,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (args.scores,args.fasta)],
        wt_sequence=sequence,steps=steps,raw_nominees=raw,duration=SECONDS,fps=FPS,
        nomination_display='Predict / Rank / Select describes the workflow only; no actual nomination ordering is reconstructed or displayed.',
        score_exclusion='In later rounds, final_score is the fixed D_zs020_base020 score used for the paper score-activity analysis, not the multi-weight shortlist or consensus rank. Round 0 uses a zero-shot rank. Neither is used to order the grid or choose A/B.',
        grid_order='Ascending position of the single newly added mutation; ties by action string and full variant string. No movement or reordering after the assay.',
        example_rule='Within same-parent measured candidates excluding the retained lead, A/B are the one-third and two-thirds indices in mutation-position order. C is the real retained lead. These examples are retrospective illustrations, not model-evaluation samples or top-three nominees.',
        selection_definition='Measured batch winner is retained as the next parent; activity is shown after assay without a rank-flip animation.',
        candidate_linking='Stable A/B/C identities link each real batch point, exact single-mutant sequence preview and conceptual landscape endpoint. Before assay colors indicate identity/highlight only, not predicted strength.',
        projection_source=dict(path=str(projection_path),sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest()),
        landscape='Original paper mesh; conceptual coordinates, not fitted experimental fitness.',
        semantic_palette=dict(prediction=BLUE,candidate_identity=BLUE,measurement=TEAL,retained_mutation=ROSE)),indent=2)+'\n')
    print('guided-exploration ready',flush=True)

if __name__=='__main__':main()
