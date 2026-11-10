#!/usr/bin/env python3
"""Animated model explanations, using native scientific motifs and editable paths.

Run with a Python containing Pillow and NumPy:
  python tools/video/render_model_story.py --preview
  python tools/video/render_model_story.py
  python tools/video/render_model_story.py --only architecture

The logo uses the project's original PNG and native editable ribbon paths.
Protein input motifs use the native 1PGA ribbon movie. The MSA, function motif
and projected feature tiles are explanatory illustrations, not observations.
Raw inputs are read-only. Requires Pillow, NumPy, aggdraw and FFmpeg.
Structure projection is maintained separately by render_structure_story.py.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

ROOT = Path(__file__).resolve().parents[2]
W, H, FPS, AA = 1920, 1080, 24, 2
LW, LH = 1280, 720
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
P = dict(ink='#2C3E50', muted='#7F8C8D', blue='#5290D3', teal='#51AA8E',
         gold='#DA6457', coral='#DA6457', rose='#E74C3C', link='#3C78CD',
         pale='#EDF4FC', green='#EDF7F3', cream='#FBEEEB', pink='#FBEFED',
         line='#DBE3E9', gray='#B3BFC8')
# Roles remain independent even where the chosen palette reuses a color.
ROLE_COLORS = dict(sequence=P['blue'], msa=P['teal'], structure=P['coral'],
                   model=P['blue'], prediction=P['blue'], measurement=P['teal'],
                   refit=P['coral'], mutation=P['rose'])
ROLE_FILLS = dict(sequence=P['pale'], msa=P['green'], structure=P['cream'],
                 model=P['pale'], prediction=P['pale'], measurement=P['green'],
                 refit=P['cream'], mutation=P['pink'])
MOD_ROLES = ('sequence','msa','structure')
CYCLE_ROLES = ('prediction','measurement','refit')
MOD_COLORS = [ROLE_COLORS[role] for role in MOD_ROLES]
MOD_FILLS = [ROLE_FILLS[role] for role in MOD_ROLES]
CYCLE_COLORS = [ROLE_COLORS[role] for role in CYCLE_ROLES]
ASSETS = ROOT/'docs/assets/video'
LOGO = ROOT/'docs/assets/proteus-logo.png'
SOURCE = Path('/Users/mhg/Dropbox/PROTEUS')
PDB = SOURCE/'Main Figure/fig1b_1PGA.pdb'
PRETRAINING_CORPUS = SOURCE/'Main Figure/proteus_figures/figures/RAW/cell_v9/benchmark_data/fig1/fig1c_scatter.csv'
PRETRAINING_MEASUREMENTS = 1322111
LOGO_GEOMETRY = SOURCE/'Submission_V2/main_figure/scripts/Figure1/current_revision/overview/assets/original_logo_geometry.json'
TEMP = Path('/private/tmp/proteus-model-story')
try:
    import aggdraw
except ImportError:
    sys.path.insert(0,str(TEMP/'deps'))
    import aggdraw
SEQUENCE = 'MTYKLILNGKTL'
MSA = ('MTYKLILNGKTL', 'MTYKLVLNGKTL', 'MTYKLILNGETL', 'MTYKLVLNGKTL')


def ease(v):
    p = max(0., min(1., v))
    return p*p*(3-2*p)


def mix(a, b, p):
    return a+(b-a)*p


def rgb(value):
    return ImageColor.getrgb(value) if isinstance(value, str) else tuple(value)


def fade(color, alpha):
    return tuple(round(255+(v-255)*max(0,min(1,alpha))) for v in rgb(color))


@functools.lru_cache(maxsize=256)
def font(size, bold=False, italic=False):
    suffix = ' Bold' if bold else (' Italic' if italic else '')
    return ImageFont.truetype(str(FONT_DIR/f'Georgia{suffix}.ttf'), max(1, round(size)))


class Draw:
    """A small retained-coordinate painter; camera transforms preserve geometry."""
    def __init__(self, image=None, camera=(1,0,0), opacity=1):
        self.image = image if image is not None else Image.new('RGB',(LW*AA,LH*AA),'white')
        self.d = ImageDraw.Draw(self.image)
        self.scale, self.ox, self.oy = camera
        self.alpha = opacity

    def point(self, x, y):
        return ((x*self.scale+self.ox)*AA,(y*self.scale+self.oy)*AA)

    def box(self, rect):
        x,y,w,h = rect
        return (*self.point(x,y), *self.point(x+w,y+h))

    def text(self, x,y,value,size=26,color=None,bold=False,anchor='mm',italic=False):
        if self.alpha <= 0: return
        self.d.text(self.point(x,y),str(value),font=font(size*self.scale*AA,bold,italic),
                    fill=fade(color or P['ink'],self.alpha),anchor=anchor)

    def line(self, points, color=None, width=2):
        if self.alpha <= 0 or len(points)<2:return
        path=aggdraw.Path();x,y=self.point(*points[0]);path.moveto(x,y)
        for point in points[1:]:path.lineto(*self.point(*point))
        ctx=aggdraw.Draw(self.image)
        ctx.path(path,None,aggdraw.Pen(fade(color or P['line'],self.alpha),width*self.scale*AA))
        ctx.flush()

    def rect(self, rect, fill='white', stroke=None, width=1.5, radius=0):
        if self.alpha <= 0: return
        box = self.box(rect)
        if box[2] <= box[0] or box[3] <= box[1]: return
        kwargs = dict(fill=fade(fill,self.alpha),outline=fade(stroke,self.alpha) if stroke else None,
                      width=max(1,round(width*self.scale*AA)))
        if radius: self.d.rounded_rectangle(box,radius=max(1,round(radius*self.scale*AA)),**kwargs)
        else: self.d.rectangle(box,**kwargs)

    def dot(self,x,y,r,color,stroke=None,width=1.5):
        if self.alpha <= 0: return
        self.d.ellipse(self.box((x-r,y-r,r*2,r*2)),fill=fade(color,self.alpha),
                       outline=fade(stroke,self.alpha) if stroke else None,
                       width=max(1,round(width*self.scale*AA)))

    def polygon(self, points, fill, stroke=None, width=1):
        if self.alpha <= 0 or len(points)<3:return
        path=aggdraw.Path();path.moveto(*self.point(*points[0]))
        for point in points[1:]:path.lineto(*self.point(*point))
        path.close();ctx=aggdraw.Draw(self.image)
        ctx.path(path,aggdraw.Brush(fade(fill,self.alpha)),aggdraw.Pen(fade(stroke,self.alpha),width*self.scale*AA) if stroke else None)
        ctx.flush()

    def image_at(self,image,x,y,w,h,alpha=1):
        if self.alpha*alpha <= 0: return
        xx,yy = self.point(x,y)
        ww,hh = round(w*self.scale*AA),round(h*self.scale*AA)
        if ww < 1 or hh < 1: return
        im = image.resize((ww,hh),Image.Resampling.LANCZOS).convert('RGBA')
        total_alpha = self.alpha*alpha
        if total_alpha < 1: im.putalpha(im.getchannel('A').point(lambda a:round(a*total_alpha)))
        self.image.paste(im,(round(xx),round(yy)),im)


def bezier(a,b,c,d,steps=80):
    return [((1-t)**3*a[0]+3*(1-t)**2*t*b[0]+3*(1-t)*t*t*c[0]+t**3*d[0],
             (1-t)**3*a[1]+3*(1-t)**2*t*b[1]+3*(1-t)*t*t*c[1]+t**3*d[1])
            for t in np.linspace(0,1,steps)]


def flow(d, points, progress, color=None, width=2.2, head=True):
    progress = max(0,min(1,progress))
    if progress <= 0: return
    distances = [math.dist(a,b) for a,b in zip(points,points[1:])]
    remain = sum(distances)*progress
    path = [points[0]]
    for a,b,dist in zip(points,points[1:],distances):
        if remain <= 0: break
        q = min(1,remain/max(dist,1e-6))
        path.append((mix(a[0],b[0],q),mix(a[1],b[1],q)))
        remain -= dist
    d.line(path,color or P['blue'],width*1.18 if head else width)
    if len(path)>1 and head:
        a,b=path[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0])
        head_size=max(15,width*5)
        d.polygon([b,(b[0]-head_size*math.cos(angle-.4),b[1]-head_size*math.sin(angle-.4)),
                     (b[0]-head_size*math.cos(angle+.4),b[1]-head_size*math.sin(angle+.4))],color or P['blue'])


def moving_bead(d,points,t,color,r=4):
    p=max(0,min(1,t)); index=min(len(points)-2,int(p*(len(points)-1)))
    q=p*(len(points)-1)-index; a,b=points[index:index+2]
    d.dot(mix(a[0],b[0],q),mix(a[1],b[1],q),r,color,'white',1)


@functools.lru_cache(maxsize=64)
def ribbon_frame(index):
    p=TEMP/'ribbon'/f'{index:03d}.png'
    im=Image.open(p).convert('RGB')
    # Crop only the white rendering viewport, retaining the entire native motif.
    return im.crop((390,55,900,670))


def ribbon(d,cx,cy,w,h,t,alpha=1):
    im=ribbon_frame(int(t*6)%48)
    scale=min(w/im.width,h/im.height)
    iw,ih=im.width*scale,im.height*scale
    d.image_at(im,cx-iw/2,cy-ih/2,iw,ih,alpha)


@functools.lru_cache(maxsize=1)
def logo_image():
    return Image.open(LOGO).convert('RGBA')


def logo(d,cx,cy,width,alpha=1):
    im=logo_image(); height=width*im.height/im.width
    d.image_at(im,cx-width/2,cy-height/2,width,height,alpha)


def feature_cell_color(index,row,col):
    return fade(MOD_COLORS[(index+row+col)%3],.55+.3*((index+row*3+col*7)%9)/8)


def token(d,x,y,size,index=0,mask=False,alpha=1):
    old=d.alpha;d.alpha*=alpha
    if mask:d.rect((x-3,y-3,size+6,size+6),'white',ROLE_COLORS['mutation'],2)
    for row in range(3):
        for col in range(3):
            d.rect((x+col*size/3,y+row*size/3,size/3-1.1,size/3-1.1),feature_cell_color(index,row,col))
    d.alpha=old


def sequence(d,cx,cy,cell=23,mutate=0,alpha=1):
    old=d.alpha; d.alpha*=alpha
    for j,ch in enumerate(SEQUENCE):
        x=cx+(j-5.5)*cell
        highlight=j==3
        d.rect((x-cell*.46,cy-15,cell*.92,30),ROLE_FILLS['mutation'] if highlight and mutate else ROLE_FILLS['sequence'],radius=2)
        d.text(x,cy,'W' if highlight and mutate>.5 else ch,18,ROLE_COLORS['mutation'] if highlight and mutate else ROLE_COLORS['sequence'],bold=highlight)
    d.alpha=old


def msa(d,cx,cy,cell=20):
    for row,seq in enumerate(MSA):
        for col,ch in enumerate(seq):
            x,y=cx+(col-5.5)*cell,cy+(row-1.5)*23
            fill=ROLE_FILLS['msa'] if ch==SEQUENCE[col] else fade(ROLE_COLORS['msa'],.4)
            d.rect((x-cell*.45,y-10,cell*.9,20),fill,radius=1)
            d.text(x,y,ch,14,ROLE_COLORS['msa'])


def plate(d,x,y,w=150,h=80,progress=1):
    d.rect((x,y,w,h),'white',ROLE_COLORS['measurement'],1.5,10)
    for row in range(5):
        for col in range(6):
            active=(row*6+col)/30 < progress
            d.dot(x+14+col*(w-28)/5,y+13+row*(h-26)/4,4,ROLE_COLORS['measurement'] if active else ROLE_FILLS['measurement'])


def overview(d,t=4,focus=None):
    """Fig. 1 overview; the structure box follows the figure's abstraction.

    Methods p33 additionally specifies ESM3 as the structure feature encoder;
    Fast 3D prediction is the distinct coordinate-generation stage (p36–37).
    """
    centers=[230,640,1050]
    titles=['Sequence','MSA','Structure']
    encoders=['ESM-2','MSA Transformer','Fast 3D prediction']
    for i,cx in enumerate(centers):
        old=d.alpha
        if focus is not None and focus!=i:d.alpha*=.25
        d.text(cx,73,titles[i],28,MOD_COLORS[i],True)
        if i==0:sequence(d,cx,202,22,mutate=ease((t-1.1)/.7))
        elif i==1:msa(d,cx,170,20)
        else:ribbon(d,cx,174,240,145,t)
        flow(d,[(cx,[230,227,245][i]),(cx,280)],ease((t-.6)/.7),MOD_COLORS[i],2.5)
        d.rect((cx-130,292,260,64),MOD_FILLS[i],MOD_COLORS[i],1.5,7)
        if i==2:
            d.text(cx,315,encoders[i],22,MOD_COLORS[i])
            d.text(cx,338,'Physics-constrained',18,MOD_COLORS[i])
        else:d.text(cx,324,encoders[i],24,MOD_COLORS[i])
        for j in range(12):
            x=cx-117+j*20
            d.rect((x,373,15,21),fade(MOD_COLORS[i],.45+.5*((j*7+i)%11)/10),radius=1)
        d.alpha=old
    q=ease((t-2.1)/1.9)
    for i,cx in enumerate(centers):
        for j in range(12):
            x0=cx-109+j*20;x1=315+j*55
            points=bezier((x0,400),(x0,445),(x1,435),(x1,475),35)
            if j in (0,3,11):flow(d,points,q,fade(MOD_COLORS[i],.4),1.2,False)
            if 2.1<t<4.7:moving_bead(d,points,ease((t-2.1-j*.018)/1.9),MOD_COLORS[i],3)
    d.rect((490,419,300,36),'white')
    d.text(640,437,'Position-wise sum',23,P['muted'])
    for j in range(12):token(d,300+j*55,483,30,j,j==3,ease((t-2.9-j*.03)/.8))
    d.text(640,540,'Aligned residue features',24,P['muted'])
    flow(d,[(640,558),(640,583)],ease((t-4)/.8),ROLE_COLORS['model'],2.5)
    d.rect((455,592,370,57),ROLE_FILLS['model'],ROLE_COLORS['model'],1.5,7)
    d.text(640,619,'MutationFormer',29,ROLE_COLORS['model'],True)
    d.text(857,619,'×12',23,P['muted'])


@functools.lru_cache(maxsize=1)
def native_logo_geometry():
    return json.loads(LOGO_GEOMETRY.read_text())


def native_path(items):
    points=[]
    for item in items:
        if item[0]=='l':
            if not points: points.append(item[1])
            points.append(item[2])
        elif item[0]=='c':
            curve=bezier(*item[1:],steps=24)
            points.extend(curve if not points else curve[1:])
    return points


def resample_path(points,count=240):
    a=np.asarray(points,float)
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))]
    samples=np.linspace(0,distance[-1],count)
    return np.column_stack([np.interp(samples,distance,a[:,j]) for j in range(2)])


@functools.lru_cache(maxsize=1)
def native_ribbon_path():
    # Include all three turns of the original left helix, using the midpoint
    # between each native polygon's paired boundary paths, then its front loop.
    drawings=native_logo_geometry()['drawings'];parts=[]
    for index in (45,53,46,55,47,57):
        points=np.array(native_path(drawings[index]['items']))
        first=resample_path(points[:101],100)
        second=resample_path(points[102:][::-1],100)
        parts.append((first+second)/2)
    start=np.array(native_path(drawings[42]['items']))
    front=np.array(native_path(drawings[43]['items']))
    nearest=np.argmin(np.linalg.norm(start-front[0],axis=1))
    parts.extend([start[:nearest+1],front,np.array(native_path(drawings[61]['items']))])
    return resample_path(np.concatenate(parts),420)


@functools.lru_cache(maxsize=1)
def logo_montage():
    # The complete project symbol stays present, including its left helix.
    return logo_image()


@functools.lru_cache(maxsize=1)
def cycle_path():
    # Straight sides and modest corner radii form an unambiguous hexagon.
    # Stage centers remain exact route anchors rather than guessed fractions.
    top=rounded_orthogonal([(265,350),(360,190),(920,190),(1015,350)],20)
    right=rounded_orthogonal([(1015,350),(920,575),(640,575)],20)
    left=rounded_orthogonal([(640,575),(360,575),(265,350)],20)
    return resample_path(top+right[1:]+left[1:],420)


@functools.lru_cache(maxsize=1)
def cycle_stations():
    path=cycle_path()
    return (0.,*(float(np.argmin(np.linalg.norm(path-np.array(point),axis=1)))/(len(path)-1)
                  for point in [(1015,350),(640,575)]),1.)


def ribbon_band(d,points,phase=0,alpha=1,width_scale=1):
    # Two open, light tracks retain the native protein's unfolding centerline
    # without a wide filled belt. Boxed stages interrupt the tracks naturally.
    points=np.array(points)
    kernel=np.exp(-np.arange(-5,6,dtype=float)**2/(2*2.1**2));kernel/=kernel.sum()
    points=np.column_stack([np.convolve(np.pad(points[:,i],5,mode='edge'),kernel,mode='valid') for i in range(2)])
    diff=np.gradient(points,axis=0)
    normals=np.column_stack([-diff[:,1],diff[:,0]])
    normals/=np.maximum(1e-5,np.linalg.norm(normals,axis=1))[:,None]
    old=d.alpha;d.alpha*=alpha
    bounds=[round(q*(len(points)-1)) for q in cycle_stations()]
    for j,color in enumerate(CYCLE_COLORS):
        a,b=bounds[j],bounds[j+1]
        d.line(points[a:b+1],color,max(1.4,2.6*width_scale))
        d.line((points+normals*6*width_scale)[a:b+1],fade(color,.35),1.1)
    if width_scale>.98:
        stations=cycle_stations()
        midpoints=[(a+b)/2 for a,b in zip(stations,stations[1:])]
        for q,color in zip(midpoints,CYCLE_COLORS):
            i=round(q*(len(points)-1));direction=points[i+2]-points[i-2]
            direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
            pt=points[i]
            d.polygon([pt+direction*9,pt-direction*7+normal*6,pt-direction*7-normal*6],color)
    d.alpha=old


def cycle_motion(q):
    # Predict, measure, refit, return: each semantic station gets a dwell.
    q=max(0,min(1,q))
    _,measurement,refit,_=cycle_stations()
    keys=[(0,0),(.08,0),(.31,measurement),(.40,measurement),(.64,refit),(.73,refit),(.96,1),(1,1)]
    for (u,a),(v,b) in zip(keys,keys[1:]):
        if u<=q<=v:return mix(a,b,ease((q-u)/(v-u)))
    return 1


def cycle_segment_color(progress):
    """Color by the current semantic segment; a completed cycle returns blue."""
    p=max(0,min(1,progress))
    _,measurement,refit,_=cycle_stations()
    role='prediction' if p<measurement or p>=1 else ('measurement' if p<refit else 'refit')
    return ROLE_COLORS[role]


def cycle_nodes(d,t,alpha=1):
    old=d.alpha;d.alpha*=alpha
    specs=[(265,350,320,160,('Mutation-effect','prediction'),'prediction'),
           (1015,350,320,160,('Wet-lab','measurement'),'measurement'),
           (640,575,420,160,('Target-specific','refitting'),'refit')]
    for index,(cx,cy,w,h,title,role) in enumerate(specs):
        col=ROLE_COLORS[role]
        active=(index==0 and t<.12) or (index==1 and .29<t<.42) or (index==2 and .62<t<.76)
        d.rect((cx-w/2+2,cy-h/2+3,w,h),'#F2F5F7',radius=11)
        d.rect((cx-w/2,cy-h/2,w,h),ROLE_FILLS[role] if active else 'white',col if active else fade(col,.65),2.4 if active else 1.6,11)
        d.text(cx,cy-h/2+26,title[0],22,col,True)
        d.text(cx,cy-h/2+52,title[1],22,col,True)
        if index==0:
            for j,ch in enumerate('MTYKLILN'):
                x=cx-115+j*16
                d.rect((x-7,cy+20,14,23),ROLE_FILLS['mutation'] if j==3 else ROLE_FILLS['sequence'],radius=1)
                d.text(x,cy+31,'W' if j==3 else ch,13,ROLE_COLORS['mutation'] if j==3 else ROLE_COLORS['prediction'])
            flow(d,[(cx+20,cy+32),(cx+49,cy+32)],1,ROLE_COLORS['prediction'],1.8)
            d.line([(cx+65,cy+56),(cx+65,cy+5)],P['line'],1.1)
            d.line([(cx+65,cy+56),(cx+130,cy+56)],P['line'],1.1)
            d.line([(cx+66,cy+46),(cx+88,cy+31),(cx+109,cy+23),(cx+128,cy+11)],ROLE_COLORS['prediction'],2.3)
        elif index==1:plate(d,cx-73,cy+4,146,66,ease(t*2))
        else:
            update=ease((t-.60)/.12)
            for j in range(3):
                yy=cy+17+mix(j,2-j,update)*16
                width=mix([109,82,53][j],[73,107,136][j],update)
                tone=fade(ROLE_COLORS['refit'],.58+j*.18)
                d.dot(cx-81,yy,4,tone)
                d.line([(cx-63,yy),(cx-63+width,yy)],tone,6)
    d.alpha=old


def logo_cycle(t,outro=False):
    d=Draw();grow=1 if outro else ease((t-.8)/2.2)
    d.text(640,62,'A mutation-resolved world model',36,P['ink'],True)
    d.text(640,111,'Closed-loop protein evolution',28,ROLE_COLORS['model'])
    width=mix(500,350,grow);cy=mix(346,335,grow)
    im=logo_montage();hh=width*im.height/im.width
    d.image_at(im,640-width/2,cy-hh/2,width,hh,1)
    native=(native_ribbon_path()-np.array([255,30]))*5+np.array([640,346])
    points=native*(1-grow)+cycle_path()*grow
    ribbon_band(d,points,phase=t*.17,alpha=ease((t-.35)/.7)*mix(.45,1,grow) if not outro else 1,width_scale=mix(.16,1,grow**2.5))
    # The full symbol stays visible beneath the unfolding ribbon, then returns
    # as the unblurred original symbol. No logo component is recolored or cut.

    d.alpha=ease((grow-.65)/.35)
    d.text(640,472,'ProTEUS',30,ROLE_COLORS['model'],True)
    d.alpha=1
    q=max(0,min(1,(t if outro else t-3.0)/(11.5 if outro else 4.8)))
    cycle_nodes(d,q,1 if outro else ease((t-2.8)/.6))
    if grow>.99:
        progress=cycle_motion(q)
        moving_bead(d,cycle_path(),progress,cycle_segment_color(progress),6)
    return d.image


def rounded_orthogonal(points,radius=22):
    result=[points[0]]
    for i in range(1,len(points)-1):
        prev=np.array(points[i-1],float);corner=np.array(points[i],float);nxt=np.array(points[i+1],float)
        incoming=(corner-prev)/max(1e-9,np.linalg.norm(corner-prev));outgoing=(nxt-corner)/max(1e-9,np.linalg.norm(nxt-corner))
        r=min(radius,np.linalg.norm(corner-prev)/2,np.linalg.norm(nxt-corner)/2)
        a=corner-incoming*r;b=corner+outgoing*r
        result.append(tuple(a));result.extend(bezier(a,a+incoming*r*.5523,b-outgoing*r*.5523,b,20)[1:])
    result.append(points[-1]);return result


def function_motif(d,cx,cy,t,w=160):
    # A qualitative function readout, deliberately without invented scores.
    points=[(cx-w/2,cy+35),(cx-w*.2,cy+20),(cx+w*.1,cy-10),(cx+w*.36,cy-27),(cx+w/2,cy-31)]
    d.line([(cx-w/2,cy+47),(cx-w/2,cy-49)],P['line'],1.3)
    d.line([(cx-w/2,cy+47),(cx+w/2,cy+47)],P['line'],1.3)
    flow(d,bezier(points[0],points[1],points[2],points[-1]),ease(t),ROLE_COLORS['prediction'],3,False)
    if t>.1:
        moving_bead(d,bezier(points[0],points[1],points[2],points[-1]),ease(t),ROLE_COLORS['prediction'],6)
    d.text(cx,cy+71,'Function score',18,ROLE_COLORS['prediction'])


def world_model_intro(t):
    return logo_cycle(t)


def world_model_loop(t):
    d=Draw()
    # The three modalities form one compact protein-state cluster.
    a=ease(t/.7);d.alpha=a
    d.text(185,190,'Sequence',25,ROLE_COLORS['sequence'])
    sequence(d,215,235,20)
    msa(d,185,349,18)
    ribbon(d,360,362,128,180,t)
    d.text(155,430,'MSA',25,ROLE_COLORS['msa'])
    d.text(360,473,'Structure',25,ROLE_COLORS['structure'])
    logo(d,650,320,242)
    d.text(650,411,'ProTEUS',35,ROLE_COLORS['model'],True)
    path=bezier((431,334),(474,334),(487,334),(520,334))
    flow(d,path,ease((t-.8)/1.1),ROLE_COLORS['model'],2.6)
    if 1<t<3.4:moving_bead(d,path,ease((t-1)/1.2),ROLE_COLORS['model'],4)
    d.alpha=ease((t-1.4)/.65)
    d.rect((524,139,252,61),ROLE_FILLS['mutation'],ROLE_COLORS['mutation'],1.4,9)
    d.text(650,169,'Single mutation',24,ROLE_COLORS['mutation'],True)
    flow(d,[(650,208),(650,248)],ease((t-1.8)/.7),ROLE_COLORS['mutation'],2.3)
    d.alpha=ease((t-2.4)/.7)
    d.text(1040,226,'Predicted function',29,ROLE_COLORS['prediction'],True)
    function_motif(d,1040,328,(t-3)/1.2)
    flow(d,[(784,334),(916,334)],ease((t-2.8)/.8),ROLE_COLORS['prediction'],3)
    d.alpha=ease((t-4.8)/.7)
    flow(d,[(1040,430),(1040,486)],ease((t-4.9)/.8),ROLE_COLORS['measurement'],3)
    plate(d,965,502,150,74,ease((t-5.3)/1.4))
    d.text(1040,612,'Measure in wet lab',27,ROLE_COLORS['measurement'],True)
    # This arrow returns observations to target-specific refitting, not to
    # pretraining or to the input encoders. The outer path closes the cycle.
    path=rounded_orthogonal([(954,540),(650,540),(650,449)],22)
    d.alpha=ease((t-6.7)/.6)
    flow(d,path,ease((t-6.8)/1.65),ROLE_COLORS['refit'],4)
    if t>6.9:moving_bead(d,path,((t-6.9)/2.3)%1,ROLE_COLORS['refit'],7)
    d.text(800,586,'Refit',27,ROLE_COLORS['refit'],True)
    d.alpha=1
    return d.image


@functools.lru_cache(maxsize=1)
def embedding_homography():
    # A single projective transform governs the plane, every tile border, and
    # every microcell. Tiles cannot float upright over a tilted plane.
    src=[(0,0),(1,0),(1,1),(0,1)];dst=[(215,215),(755,145),(1060,350),(470,430)]
    matrix=[];values=[]
    for (u,v),(x,y) in zip(src,dst):
        matrix.extend([[u,v,1,0,0,0,-x*u,-x*v],[0,0,0,u,v,1,-y*u,-y*v]])
        values.extend([x,y])
    return np.r_[np.linalg.solve(matrix,values),1].reshape(3,3)


def plane_point(u,v):
    point=embedding_homography()@np.array([u,v,1]);return tuple(point[:2]/point[2])


def planar_box(u,v,w,h):
    return [plane_point(u,v),plane_point(u+w,v),plane_point(u+w,v+h),plane_point(u,v+h)]


def projected_tile(d,u,v,index):
    w,h=.105,.26
    d.polygon(planar_box(u-.006,v-.012,w+.012,h+.024),'white',ROLE_COLORS['mutation'] if index==3 else fade(P['blue'],.4),1.4)
    for row in range(3):
        for col in range(3):
            color=feature_cell_color(index,row,col)
            d.polygon(planar_box(u+col*w/3+.002,v+row*h/3+.006,w/3-.004,h/3-.012),color)


def flying_feature(d,index,progress):
    """Move each exact residue feature grid into its projected plane tile."""
    p=ease(progress);u=.055+(index%6)*.155;v=.16+(index//6)*.42;w,h=.105,.26
    if p>=1:
        projected_tile(d,u,v,index)
        return
    arc=(-55 if index<6 else 25)*math.sin(math.pi*p)
    sx,sy=300+index*55,483
    target=planar_box(u-.006,v-.012,w+.012,h+.024)
    source=[(sx-3,sy-3),(sx+33,sy-3),(sx+33,sy+33),(sx-3,sy+33)]
    border=[(mix(a[0],b[0],p),mix(a[1],b[1],p)+arc) for a,b in zip(source,target)]
    if p>.02 or index==3:d.polygon(border,'white',ROLE_COLORS['mutation'] if index==3 else fade(P['blue'],.4),1.4)
    for row in range(3):
        for col in range(3):
            x,y=sx+col*10,sy+row*10
            src=[(x,y),(x+8.9,y),(x+8.9,y+8.9),(x,y+8.9)]
            dst=planar_box(u+col*w/3+.002,v+row*h/3+.006,w/3-.004,h/3-.012)
            quad=[(mix(a[0],b[0],p),mix(a[1],b[1],p)+arc) for a,b in zip(src,dst)]
            d.polygon(quad,feature_cell_color(index,row,col))


def architecture(t):
    # Three seconds to read the complete pipeline, then the residue features
    # themselves move into the joint plane. MutationFormer never becomes a zoom target.
    if t<=3:
        d=Draw();overview(d,4.8);return d.image
    d=Draw();p=ease((t-3)/2.4)
    context=Draw();overview(context,4.8)
    d.image_at(context.image.crop((0,0,LW*AA,477*AA)),0,-130*p,LW,477,1-ease((t-3)/1.0))
    d.alpha=ease((t-3.8)/.8)
    d.text(640,74,'Joint embedding space',33,ROLE_COLORS['model'],True)
    shadow=[(x+2,y+7) for x,y in planar_box(0,0,1,1)]
    d.polygon(shadow,'#E7EEF3')
    d.polygon(planar_box(0,0,1,1),'#F5F9FC','#BCD0DD',1.4)
    for u in np.linspace(0,1,7):d.line([plane_point(u,0),plane_point(u,1)],'#DFEAF1',.8)
    for v in np.linspace(0,1,4):d.line([plane_point(0,v),plane_point(1,v)],'#DFEAF1',.8)
    d.alpha=1
    for j in range(12):flying_feature(d,j,(t-3.15-j*.018)/2.05)
    d.alpha=1-ease((t-3)/.85)
    d.text(640,540,'Aligned residue features',24,P['muted'])
    d.alpha=1
    bx=mix(455,465,p);by=mix(592,493,p);bw=mix(370,350,p);bh=mix(57,76,p)
    for j in range(3):d.rect((bx+j*5*p,by-j*5*p,bw,bh),ROLE_FILLS['model'],ROLE_COLORS['model'],1.5,7)
    d.text(mix(640,650,p),mix(619,526,p),'MutationFormer',mix(29,31,p),ROLE_COLORS['model'],True)
    flow(d,[(640,440),(640,480)],ease((t-5.5)/.7),ROLE_COLORS['model'],2.5)
    d.alpha=ease((t-6)/.7)
    d.rect((112,495,230,65),ROLE_FILLS['mutation'],ROLE_COLORS['mutation'],1.5,7)
    d.text(227,526,'Mutations',28,ROLE_COLORS['mutation'],True)
    flow(d,[(353,526),(452,526)],ease((t-6.3)/.7),ROLE_COLORS['mutation'],2.5)
    d.alpha=ease((t-7.2)/.7)
    flow(d,[(838,526),(936,526)],ease((t-7.2)/.7),ROLE_COLORS['prediction'],2.5)
    d.text(1050,465,'Predicted function',26,ROLE_COLORS['prediction'],True)
    function_motif(d,1050,562,(t-7.1)/1.2,w=150)
    d.alpha=1
    return d.image


def zoom_recall(t,duration=3):
    q=ease(t/duration)
    target=(640,619)
    scale=mix(1,2.4,q)
    ox=mix(0,640-target[0]*2.4,q)
    oy=mix(0,335-target[1]*2.4,q)
    d=Draw(camera=(scale,ox,oy))
    overview(d,4.8)
    return d.image


def scaling_architecture(t):
    # Timing is a full, stable pipeline hold followed by a separate zoom.
    if t<3:
        d=Draw();overview(d,4.8);return d.image
    if t<4.2:return zoom_recall(t-3,duration=1.2)
    if t<5.7:return zoom_recall(1.2,duration=1.2)
    u=t-5.7;d=Draw()
    for i,(cx,title) in enumerate([(290,'Sequence'),(640,'MSA'),(990,'Structure')]):
        d.rect((cx-115,112,230,58),MOD_FILLS[i],MOD_COLORS[i],1.3,7)
        d.text(cx,140,title,26,MOD_COLORS[i])
        branch=[(cx,183),(cx,231)] if cx==640 else rounded_orthogonal([(cx,183),(cx,231),(640,231)],14)
        flow(d,branch,1,MOD_COLORS[i],1.7,False)
    flow(d,[(640,231),(640,285)],1,ROLE_COLORS['model'],1.7)
    d.rect((421,300,438,112),ROLE_FILLS['model'],ROLE_COLORS['model'],2,9)
    d.text(640,355,'MutationFormer',38,ROLE_COLORS['model'],True)
    p=ease(u/1.3)
    for j in range(8):
        d.alpha=ease((p*8-j)/.7)
        d.rect((209+j*11,552-j*8,155,92),'white',ROLE_COLORS['measurement'],1.5,4)
        for r in range(3):d.line([(226+j*11,576-j*8+r*18),(344+j*11,576-j*8+r*18)],ROLE_FILLS['measurement'],5)
    d.alpha=1
    path=bezier((443,557),(528,557),(569,488),(585,429))
    flow(d,path,ease(u/.8),ROLE_COLORS['measurement'],2.6)
    if u>.3:moving_bead(d,path,((u-.3)/1.6)%1,ROLE_COLORS['measurement'],6)
    d.text(862,545,'Pretraining corpus',31,ROLE_COLORS['model'])
    d.text(862,598,'1.32M variant measurements',43,ROLE_COLORS['model'],True)
    return d.image


def world_model_outro(t):
    return logo_cycle(t,True)


CLIPS={
    'world-model-intro':(8,world_model_intro,[.3,1.8,4,7.6]),
    'world-model-loop':(12,world_model_loop,[2,5.7,10.5]),
    'architecture':(12,architecture,[2.5,3.7,5.5,11.3]),
    'scaling-architecture':(9,scaling_architecture,[2.5,3.7,5.2,8.6]),
    'world-model-outro':(16,world_model_outro,[1,4.2,8,15.5]),
}


def prepare_inputs():
    TEMP.mkdir(parents=True,exist_ok=True)
    with PRETRAINING_CORPUS.open(newline='') as source:
        rows=list(csv.DictReader(source))
    if len(rows)!=182 or sum(int(row['n_mutants']) for row in rows)!=PRETRAINING_MEASUREMENTS:
        raise ValueError('The Figure 2A pretraining corpus count changed.')
    cache=TEMP/'ribbon';cache.mkdir(exist_ok=True)
    if len(list(cache.glob('*.png')))!=48:
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(ASSETS/'protein-state.mp4'),
                        '-vf','fps=6','-frames:v','48','-start_number','0',str(cache/'%03d.png')],check=True)
    if PDB.exists():
        residues=[];seen=set()
        names={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
               'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
        for row in PDB.read_text().splitlines():
            if row.startswith('ATOM') and row[21:22]=='A':
                key=row[22:27]
                if key not in seen:seen.add(key);residues.append(names.get(row[17:20],'X'))
        if ''.join(residues)[:12] != SEQUENCE:raise ValueError('1PGA sequence motif changed.')


def render(name,seconds,fn,samples,args):
    args.output.mkdir(parents=True,exist_ok=True)
    for index,t in enumerate(samples):
        im=fn(t).resize((W,H),Image.Resampling.LANCZOS)
        im.save(args.preview_dir/f'{name}-{index+1}.png')
    fn(samples[-1]).resize((W,H),Image.Resampling.LANCZOS).save(args.output/f'{name}-poster.webp',quality=93)
    if args.preview:return
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
             '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/f'{name}.mp4')]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    for frame in range(round(seconds*FPS)):
        im=fn(frame/FPS).resize((W,H),Image.Resampling.LANCZOS)
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError(f'FFmpeg failed: {name}')
    print(f'{name}: {seconds}s / {round(seconds*FPS)} frames',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--only',choices=list(CLIPS))
    parser.add_argument('--output',type=Path,default=ASSETS)
    parser.add_argument('--preview-dir',type=Path,default=TEMP/'previews')
    args=parser.parse_args()
    args.preview_dir.mkdir(parents=True,exist_ok=True)
    prepare_inputs()
    for name,(seconds,fn,samples) in CLIPS.items():
        if args.only and args.only!=name:continue
        render(name,seconds,fn,samples,args)
    provenance={
      'renderer':'tools/video/render_model_story.py','width':W,'height':H,'fps':FPS,'antialias_render_size':[LW*AA,LH*AA],
      'sources':[{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [LOGO,LOGO_GEOMETRY,ASSETS/'protein-state.mp4',PDB,PRETRAINING_CORPUS] if p.exists()],
      'dependencies':{'Pillow':'image compositing','NumPy':'projective geometry','aggdraw':'antialiased vector paths','FFmpeg':'H.264 encoding'},
      'scientific_limits':['MSA motifs, projected feature tiles and qualitative function readouts are explanatory diagrams, not additional measurements or a measured low-dimensional embedding.',
        'Target-specific refitting updates the target predictor; pretrained representations remain fixed during the campaign.',
        'Native 1PGA ribbon preserves structure coordinates and serves only as an input motif.',
        'Logo motion traces the original protein ribbon including its complete left helix; the complete original symbol remains sharp and retains its original pixel colors.',
        'All feature-tile corners and cells use one planar homography; identical aligned residue grids move into the plane.',
        'Fig. 1E names the structure stage Fast 3D prediction / Physics-constrained. Methods p33 separately specifies ESM3 as the structural feature encoder; p36–37 explains the coordinate predictor and differentiable Gauss–Seidel projection. These are different functions.'],
      'pretraining_corpus':{'variant_measurements':PRETRAINING_MEASUREMENTS,'assays':182,
        'source':str(PRETRAINING_CORPUS),'count_column':'n_mutants',
        'scope':'Full Figure 2A corpus, including train/validation/test; not the number used in the training split.',
        'training_split_count':None,'display':'Pretraining corpus / 1.32M variant measurements'},
      'cycle_shape':'Rounded hexagon with stage dwell positions measured along the path',
      'cycle_station_fractions':list(cycle_stations()),
      'pipeline_hold_seconds':3,
      'pretrain_zoom_hold_seconds':1.5,
      'architecture_final_hold_seconds':3.7,
      'outro_final_hold_seconds':4.5,
      'palette':P,
      'semantic_color_roles':ROLE_COLORS,
      'semantic_role_fills':ROLE_FILLS,
      'cycle_color_order':list(CYCLE_ROLES),
      'clips':{name:{'seconds':v[0],'video':name+'.mp4','poster':name+'-poster.webp'} for name,v in CLIPS.items()}}
    (args.output/'model-story.source.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
