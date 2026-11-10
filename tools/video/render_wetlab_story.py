#!/usr/bin/env python3
"""Render the experimental loop and the paper's cumulative top-10 snapshots.

Quantitative positions come from chart-data.json.points, including the original
horizontal packing. Each snapshot contains ten variant means selected from all
variants assayed through that round; earlier variants may reappear. Violins use
the paper's Gaussian KDE factor .35 over the observed range, without boxes,
medians, whiskers, connecting lines, or additional observations. The distinct
90/150 tested-variant budgets are calculated separately from snapshot marks.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import math
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
W,H,FPS=1920,1080,24
INK,BLUE,MUTED,LINE='#2c3e50','#5290d3','#77838d','#dde5eb'
TEAL,ROSE,REFIT='#51aa8e','#e74c3c','#da6457'
STROKES=['#AC936B','#529C96','#4383AB','#85919B','#0A4363']
FILLS=['#F7EED8','#E8F4EF','#8BBBD5','#DADEE1','#6D9EC6']
FONT=Path('/System/Library/Fonts/Supplemental/Georgia.ttf')


def ease(v):
    v=max(0,min(1,v));return v*v*(3-2*v)


def font(size,bold=False):
    return ImageFont.truetype(str(FONT.with_name('Georgia Bold.ttf') if bold else FONT),size)


def text(d,xy,value,size=32,fill=INK,anchor='mm',bold=False):
    assert size>=30
    d.text(xy,value,font=font(size,bold),fill=fill,anchor=anchor)


def blend_color(color,alpha):
    rgb=tuple(int(color[i:i+2],16) for i in (1,3,5))
    return tuple(round(255+(v-255)*alpha) for v in rgb)


def arrow(d,points,progress=1,color=BLUE,width=4):
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    remaining=sum(lengths)*max(0,min(1,progress));end=points[0];previous=end
    for a,b,length in zip(points,points[1:],lengths):
        if remaining<=0:break
        part=min(1,remaining/length);end=(a[0]+(b[0]-a[0])*part,a[1]+(b[1]-a[1])*part)
        d.line([a,end],fill=color,width=width);previous=a;remaining-=length
    if progress>.02:
        angle=math.atan2(end[1]-previous[1],end[0]-previous[0])
        d.polygon([end,(end[0]-16*math.cos(angle-.4),end[1]-16*math.sin(angle-.4)),
                   (end[0]-16*math.cos(angle+.4),end[1]-16*math.sin(angle+.4))],fill=color)


def load_logo():
    logo=Image.open(ROOT/'docs/assets/proteus-logo.png').convert('RGBA')
    return logo.crop(logo.getbbox())


def put_logo(canvas,logo,center,width):
    image=logo.resize((width,round(logo.height*width/logo.width)),Image.Resampling.LANCZOS)
    canvas.paste(image,(round(center[0]-image.width/2),round(center[1]-image.height/2)),image)


def candidates(d,center,scale=1,progress=1,columns=5):
    # Exactly thirty symbolic sequence rows; no invented molecular measurements.
    cx,cy=center
    rows=30//columns
    for row in range(rows):
        for col in range(columns):
            index=row*columns+col;x=cx+(col-(columns-1)/2)*49*scale;y=cy+(row-(rows-1)/2)*27*scale
            alpha=.25+.75*ease(progress*1.35-index/40)
            for aa in range(5):
                color=ROSE if aa==(index%5) else BLUE
                d.rectangle((x+(aa-2)*7*scale,y-5*scale,x+(aa-2)*7*scale+5*scale,y+5*scale),
                            fill=blend_color(color,alpha))


def assay_plate(d,center,scale=1,progress=1):
    cx,cy=center;w,h=250*scale,174*scale
    d.rounded_rectangle((cx-w/2,cy-h/2,cx+w/2,cy+h/2),radius=12*scale,outline=LINE,width=3)
    for row in range(5):
        for col in range(6):
            index=row*6+col;x=cx+(col-2.5)*36*scale;y=cy+(row-2)*30*scale
            alpha=.18+.82*ease(progress*1.3-index/40)
            radius=8*scale
            d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=blend_color(TEAL,alpha),outline=TEAL,width=1)


def predictor(d,center,scale=1,progress=1):
    cx,cy=center
    # A fixed model icon receives assay rows; no fabricated prediction accuracy.
    for r in range(4):
        y=cy+(r-1.5)*29*scale
        d.line((cx-88*scale,y,cx+88*scale,y),fill=LINE,width=3)
        x=cx-88*scale+176*scale*ease(progress-r*.10)
        d.ellipse((x-7*scale,y-7*scale,x+7*scale,y+7*scale),fill=REFIT)


def wetlab(t,logo):
    canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
    centers=[245,720,1180,1630]
    put_logo(canvas,logo,(245,435),315)
    text(d,(245,249),'ProTEUS',45,BLUE,bold=True)
    text(d,(720,249),'Nominate',41,BLUE,bold=True)
    text(d,(1180,249),'Build and measure',41,TEAL,bold=True)
    text(d,(1630,249),'Refit',41,REFIT,bold=True)
    candidates(d,(720,435),1.35,ease((t-.8)/1.7))
    assay_plate(d,(1180,435),1.2,ease((t-2.4)/1.8))
    predictor(d,(1630,435),1.35,ease((t-4.3)/1.7))
    text(d,(720,650),'30 sequence candidates',32)
    text(d,(1180,650),'Assay activity',32)
    for points,start,color in [([(420,435),(540,435)],.35,BLUE), ([(880,435),(1010,435)],2.0,TEAL), ([(1350,435),(1460,435)],4.0,REFIT)]:
        arrow(d,points,ease((t-start)/1.0),color)
    # The return path goes to nomination, not into the fixed representation.
    arrow(d,[(1630,725),(1630,830),(720,830),(720,725)],ease((t-5.8)/1.5),REFIT,5)
    text(d,(1175,902),'Nominate the next batch',37,BLUE)
    return canvas


@functools.lru_cache(maxsize=16)
def density(values):
    """Equivalent to scipy gaussian_kde(values, bw_method=.35), 256 samples."""
    mean=sum(values)/len(values)
    variance=sum((value-mean)**2 for value in values)/(len(values)-1)
    bandwidth=.35*math.sqrt(variance)
    assert bandwidth>0
    low,high=min(values),max(values)
    grid=[]
    for i in range(256):
        y=low+(high-low)*i/255
        value=sum(math.exp(-.5*((y-v)/bandwidth)**2) for v in values)
        value/=len(values)*bandwidth*math.sqrt(2*math.pi)
        grid.append((y,value))
    peak=max(v for _,v in grid)
    return [(y,.34*v/peak) for y,v in grid]


def validate_snapshots(data):
    raw,points,rounds=data['raw'],data['points'],data['rounds']
    assert len(raw)==rounds*30==data['assayed']
    assert len({p['variant'] for p in raw})==len(raw)
    assert len(points)==rounds*10
    lookup={p['variant']:p for p in raw}
    for r in range(rounds):
        selected=[p for p in points if p['snapshot_round']==r]
        eligible=[p for p in raw if p['round']<=r]
        expected=sorted(eligible,key=lambda p:p['activity'],reverse=True)[:10]
        assert len(selected)==10 and len(eligible)==30*(r+1)
        assert {p['variant'] for p in selected}=={p['variant'] for p in expected}
        for point in selected:
            original=lookup[point['variant']]
            assert point['activity']==original['activity']
            assert point['source_round']==original['round']<=r
            assert math.isfinite(point['plot_x']) and abs(point['plot_x']-r)<=.35


def round_state(t,rounds):
    # Finish all batches by eight seconds, then retain the complete data for two.
    duration=7.4/rounds
    progress=(t-.6)/duration
    current=min(rounds-1,max(0,math.floor(progress)))
    phase=max(0,min(1,progress-current))
    if progress>=rounds:phase=1
    return current,phase,duration


def mini_loop(canvas,d,logo,current,phase,total,done):
    cx=1590
    put_logo(canvas,logo,(1390,112),160)
    text(d,(1670,92),'ProTEUS',39,BLUE,bold=True)
    text(d,(cx,230),f'Round {current}',36,INK,bold=True)
    stages=[('Nominate 30',345,BLUE),('Build and measure',565,TEAL),('Refit',785,REFIT)]
    active=0 if phase<.24 else (1 if phase<.65 else 2)
    for index,(title,y,color) in enumerate(stages):
        fill=blend_color(color,.07) if active==index and not done else 'white'
        outline=color if active==index and not done else LINE
        d.rounded_rectangle((1320,y-78,1860,y+78),radius=13,fill=fill,outline=outline,width=3)
        text(d,(cx,y-37),title,33,color,bold=active==index)
        if index==0:candidates(d,(cx,y+28),.70,ease(phase/.24),columns=10)
        elif index==1:assay_plate(d,(cx,y+25),.44,ease((phase-.24)/.40))
        else:predictor(d,(cx,y+24),.67,ease((phase-.65)/.28))
    arrow(d,[(cx,425),(cx,481)],ease(phase/.25),TEAL)
    arrow(d,[(cx,645),(cx,701)],ease((phase-.5)/.22),REFIT)
    arrow(d,[(1868,785),(1892,785),(1892,345),(1868,345)],ease((phase-.81)/.19),REFIT,4)
    text(d,(cx,947),f'{total} variants tested',34,INK)
    text(d,(cx,1004),'Feedback guides the next batch',30,REFIT)


def recursive(t,name,data,logo):
    canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
    rounds=data['rounds'];points=data['points'];max_y=16 if name=='sacb' else 8
    left,top,right,bottom=140,235,1200,850
    xmin,xmax=(-.5,2.5) if name=='sacb' else (-.6,4.6)
    xm=lambda x:left+(x-xmin)/(xmax-xmin)*(right-left)
    ym=lambda y:bottom-y/max_y*(bottom-top)
    current,phase,duration=round_state(t,rounds)
    text(d,(100,80),'SacB' if name=='sacb' else 'ALDH',48,INK,anchor='lm',bold=True)
    text(d,(1185,80),'Cumulative top 10',38,BLUE,anchor='rm',bold=True)
    text(d,(100,150),'Fold improvement (× WT)',35,INK,anchor='lm')
    text(d,(1185,150),'through each round',31,MUTED,anchor='rm')
    d.line((1250,45,1250,1033),fill=LINE,width=2)
    ticks=[0,5,10,15] if name=='sacb' else [0,2,4,6]
    for value in ticks:
        y=ym(value)
        d.line((left-10,y,left,y),fill='#85919B',width=3)
        text(d,(left-22,y),str(value),32,MUTED,anchor='rm')
    d.line((left,top,left,bottom,right,bottom),fill='#85919B',width=3)
    for x in range(left,right,20):d.line((x,ym(1),min(x+11,right),ym(1)),fill='#85919B',width=2)
    text(d,(right-12,ym(1)-22),'WT',30,'#85919B',anchor='rm')
    for r in range(rounds):
        x=xm(r)
        d.line((x,bottom,x,bottom+10),fill='#85919B',width=3)
        text(d,(x,bottom+42),str(r),34,INK)
        if r>current:continue
        reveal=1 if r<current else ease((phase-.24)/.40)
        if reveal<=0:continue
        selected=[p for p in points if p['snapshot_round']==r]
        shape=density(tuple(p['activity'] for p in selected))
        polygon=[(xm(r-half),ym(y)) for y,half in shape]
        polygon += [(xm(r+half),ym(y)) for y,half in reversed(shape)]
        d.polygon(polygon,fill=blend_color(FILLS[r],reveal*.8))
        d.line(polygon+[polygon[0]],fill=blend_color(STROKES[r],reveal*.8),width=3)
        # Preserve source plot_x exactly. No re-packing or vertical animation.
        for point in selected:
            px,py=xm(point['plot_x']),ym(point['activity'])
            d.ellipse((px-9.5,py-9.5,px+9.5,py+9.5),
                      fill=blend_color(STROKES[r],reveal),outline='white',width=2)
        if r==rounds-1:
            best=max(p['activity'] for p in selected)
            offset=.7 if name=='sacb' else .35
            text(d,(x,ym(best+offset)),f'{best:.2f}×',36,blend_color(STROKES[r],reveal),bold=True)
    complete=current+int(phase>=.64)
    mini_loop(canvas,d,logo,current,phase,30*complete,t>=8)
    text(d,(670,977),'Evolution round',33,INK)
    text(d,(670,1030),'10 variant means per snapshot; earlier variants can reappear',30,MUTED)
    return canvas


def encode(name,seconds,fn,assets,preview):
    folder=Path('/private/tmp/proteus-wetlab-story');folder.mkdir(exist_ok=True)
    for t in [1,seconds/2,seconds-.5]:fn(t).save(folder/f'{name}-{t:g}.png')
    fn(seconds-.5).save(assets/f'{name}-poster.webp',quality=94,method=6)
    if preview:return
    process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
        '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(assets/f'{name}.mp4')],stdin=subprocess.PIPE)
    for i in range(round(seconds*FPS)):process.stdin.write(fn(i/FPS).tobytes())
    process.stdin.close()
    if process.wait():raise RuntimeError(f'Encoding failed: {name}')
    print(name,'ready',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'tools/video/chart-data.json')
    ap.add_argument('--assets',type=Path,default=ROOT/'docs/assets/video')
    ap.add_argument('--only',choices=['all','recursive'],default='all')
    ap.add_argument('--preview-only',action='store_true');args=ap.parse_args()
    data=json.loads(args.data.read_text());args.assets.mkdir(parents=True,exist_ok=True);logo=load_logo()
    provenance=dict(recipe='tools/video/render_wetlab_story.py',size=[W,H],fps=FPS,
                    model_update='Refit target-specific predictor; fixed ProTEUS representations.',
                    diagram='Conceptual workflow; plate wells do not encode fabricated activities.',
                    semantic_palette=dict(prediction_nomination=BLUE,measurement=TEAL,refit=REFIT,mutation=ROSE))
    if args.only=='all':
        encode('wetlab-loop',8,lambda t:wetlab(t,logo),args.assets,args.preview_only)
        (args.assets/'wetlab-loop.source.json').write_text(json.dumps(dict(provenance,duration=8),indent=2))
    for name in ('sacb','aldh'):
        sample=data[name];validate_snapshots(sample)
        encode(f'{name}-recursive',10,lambda t,n=name,s=sample:recursive(t,n,s,logo),args.assets,args.preview_only)
        source=dict(provenance,duration=10,data_file=str(args.data),
                    data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),original_sources=data['sources'],
                    paper_panel='Figure 4C' if name=='sacb' else 'Figure 5B',
                    number_tested=sample['assayed'],rounds=sample['rounds'],
                    number_snapshot_marks=len(sample['points']),snapshot_points=sample['points'],
                    statistic='Ten highest-activity variant means among all non-WT variants assayed through each round.',
                    repeated_variants='A variant may recur across cumulative snapshots; the 30/50 displayed marks are not independent discoveries.',
                    density=dict(bandwidth_factor=.35,grid_points=256,support='observed minimum to maximum',
                                 maximum_halfwidth_categories=.34,fill_alpha=.8),
                    boxplot_elements='None: no boxes, medians, quartiles, whiskers or connecting summary lines.',
                    colors=dict(strokes=STROKES[:sample['rounds']],fills=FILLS[:sample['rounds']]),
                    y_range=[0,16 if name=='sacb' else 8],
                    x_range=[-.5,2.5] if name=='sacb' else [-.6,4.6],
                    vertical_coordinate='Exact reported variant-mean screening activity / WT; no vertical jitter.',
                    horizontal_coordinate='Exact original paper plot_x; no repacking.',
                    original_methods=data['methods'])
        (args.assets/f'{name}-recursive.source.json').write_text(json.dumps(source,indent=2))


if __name__=='__main__':main()
