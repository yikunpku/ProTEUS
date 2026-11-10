#!/usr/bin/env python3
"""Compose WT reference structures with exact WT / final-variant sequences.

Reads supplied paper FASTAs, validates every mutation's original residue,
and stores the complete WT and final strings in enzyme-sequences.json.
Only WT reference geometry is shown. Sequence windows are aligned and retain
full-precursor numbering, including SacB G23 in the unresolved signal peptide.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
W,H,FPS,SECONDS=1920,1080,24,8
INK,BLUE,MUTED,LINE,ROSE='#2c3e50','#5290D3','#77838d','#dce3e9','#E74C3C'
FONT=Path('/System/Library/Fonts/Supplemental/Georgia.ttf')
CONFIG={
    'sacb':dict(title='Levansucrase · SacB',accession='P05655',length=473,
                fasta='Levensrucase_sequence.fasta',mutations=['G23A','Q34N','G240W'],
                movie='sacb-final-reference.mp4',structure='WT reference · 1OYG',
                note='G23 is in the signal peptide and is unresolved in 1OYG.',
                crop=(220,5,1060,705)),
    'aldh':dict(title='Aldehyde dehydrogenase · ALDH',accession='P46368',length=506,
                fasta='ALDH_sequence.fasta',mutations=['S103V','A116K','H332N','L358I'],
                movie='aldh-structure.mp4',structure='WT reference model · AF-P46368-F1',
                note='Highlighted sites are mapped onto the WT reference model.',
                crop=(100,5,1180,705)),
}


def ease(v):
    v=max(0,min(1,v)); return v*v*(3-2*v)


def font(size,bold=False):
    return ImageFont.truetype(str(FONT.with_name('Georgia Bold.ttf') if bold else FONT),size)


def text(d,xy,value,size=27,fill=INK,anchor='mm',bold=False):
    d.text(xy,value,font=font(size,bold),fill=fill,anchor=anchor)


def read_sequence(path):
    lines=path.read_text().splitlines()
    sequence=''.join(line.strip() for line in lines if not line.startswith('>')).upper()
    assert sequence and set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY')
    return sequence, '\n'.join(line for line in lines if line.startswith('>'))


def record(name,config,source_root):
    path=source_root/config['fasta']; wt,header=read_sequence(path)
    assert len(wt)==config['length']
    final=list(wt); substitutions=[]
    for mutation in config['mutations']:
        position=int(mutation[1:-1]); before,after=mutation[0],mutation[-1]
        assert wt[position-1]==before, (name,mutation,wt[position-1])
        final[position-1]=after
        lo,hi=max(1,position-5),min(len(wt),position+5)
        substitutions.append(dict(label=mutation,position=position,wt=before,final=after,start=lo,end=hi))
    final=''.join(final)
    assert [i+1 for i,(a,b) in enumerate(zip(wt,final)) if a!=b]==[s['position'] for s in substitutions]
    return dict(name=name,accession=config['accession'],length=len(wt),source=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fasta_header=header,
                numbering='1-based full sequence; SacB includes its signal peptide.',
                wild_type=wt,final_variant=final,substitutions=substitutions)


def decode(path,crop):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height','-of','json',str(path)]))['streams'][0]
    w,h=p['width'],p['height']; size=w*h*3
    process=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    result=[]
    while True:
        raw=process.stdout.read(size)
        if not raw:break
        if len(raw)!=size:raise RuntimeError('Incomplete molecular frame')
        result.append(Image.frombytes('RGB',(w,h),raw).crop(crop))
    if process.wait():raise RuntimeError('Molecular decode failed')
    assert len(result)==192
    return result


def strip(d,sequence,x,y,width,substitutions=None,t=8):
    n=len(sequence)
    for i,_letter in enumerate(sequence):
        px=x+i*width/(n-1)
        d.line((px,y-5,px,y+5),fill='#b8c6cf',width=1)
    for index,sub in enumerate(substitutions or []):
        alpha=ease((t-.8-index*.72)/.65)
        if alpha<=0:continue
        px=x+(sub['position']-1)*width/(n-1)
        color=tuple(round(255*(1-alpha)+int(ROSE[j:j+2],16)*alpha) for j in (1,3,5))
        d.line((px,y-10,px,y+10),fill=color,width=4)


def sequence_window(record,sub,index,t,top):
    canvas=Image.new('RGB',(850,139),'white');d=ImageDraw.Draw(canvas)
    text(d,(15,14),sub['label'],27,ROSE,anchor='lm',bold=True)
    text(d,(815,14),f"Residues {sub['start']}–{sub['end']}",20,MUTED,anchor='rm')
    start,end=sub['start']-1,sub['end']
    for row,(label,sequence) in enumerate([('WT',record['wild_type']),('Final',record['final_variant'])]):
        y=58+row*45
        text(d,(88,y),label,25,MUTED,anchor='rm')
        for j,letter in enumerate(sequence[start:end]):
            px=134+j*49
            changed=start+j+1==sub['position']
            if changed:
                fill='#FDEBE9' if row else '#edf1f4'
                d.rounded_rectangle((px-21,y-20,px+21,y+21),radius=4,fill=fill)
            text(d,(px,y),letter,31,ROSE if changed and row else INK,bold=changed)
        text(d,(104,y),'…',21,MUTED)
        text(d,(682,y),'…',21,MUTED)
    # Reveal complete matched windows, never fabricated intermediate sequences.
    alpha=ease((t-.8-index*.72)/.65)
    return Image.blend(Image.new('RGB',canvas.size,'white'),canvas,alpha)


def frame(t,name,config,record,clips):
    canvas=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(canvas)
    text(d,(960,51),config['title'],40,INK,bold=True)
    text(d,(960,103),f"{config['accession']} · {record['length']} residues",27,MUTED)
    text(d,(480,178),'WT reference structure',30,BLUE)
    text(d,(1440,178),'WT and final variant sequences',30,BLUE)
    d.line((960,166,960,976),fill=LINE,width=1)
    protein=clips[int(t*FPS)%len(clips)]
    width,height=protein.size; scale=min(850/width,700/height)
    protein=protein.resize((round(width*scale),round(height*scale)),Image.Resampling.LANCZOS)
    canvas.paste(protein,(round(480-protein.width/2),round(565-protein.height/2)))
    text(d,(480,942),config['structure'],25,INK)
    text(d,(480,985),config['note'],20,MUTED)
    # Entire sequence position overview: every residue is represented by a tick.
    text(d,(1090,231),'WT',23,MUTED,anchor='rm')
    text(d,(1090,267),'Final',23,MUTED,anchor='rm')
    strip(d,record['wild_type'],1125,231,690)
    strip(d,record['final_variant'],1125,267,690,record['substitutions'],t)
    for pos in [1,100,200,300,400,record['length']]:
        if pos>record['length']:continue
        x=1125+(pos-1)*690/(record['length']-1)
        text(d,(x,296),str(pos),18,MUTED)
    text(d,(1470,331),'Position in the full sequence',20,MUTED)
    spacing=175 if name=='sacb' else 145
    for index,sub in enumerate(record['substitutions']):
        window=sequence_window(record,sub,index,t,0)
        canvas.paste(window,(1010,369+index*spacing))
    text(d,(1440,1010),f"Final variant: {len(record['substitutions'])} substitutions; all other residues unchanged",22,MUTED)
    return canvas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,default=Path('/Users/mhg/Dropbox/PROTEUS/Main Figure/fig3_4_protein_structure'))
    ap.add_argument('--assets',type=Path,default=ROOT/'docs/assets/video')
    ap.add_argument('--preview-only',action='store_true')
    args=ap.parse_args();args.assets.mkdir(parents=True,exist_ok=True)
    records={name:record(name,config,args.source_root) for name,config in CONFIG.items()}
    (ROOT/'tools/video/enzyme-sequences.json').write_text(json.dumps(records,indent=2))
    preview=Path('/private/tmp/proteus-sequence-story');preview.mkdir(exist_ok=True)
    for name,config in CONFIG.items():
        clips=decode(args.assets/config['movie'],config['crop'])
        for t in [0,2,5,7]:frame(t,name,config,records[name],clips).save(preview/f'{name}-{t:02}.png')
        frame(6,name,config,records[name],clips).save(args.assets/f'{name}-sequence-poster.webp',quality=93,method=6)
        if not args.preview_only:
            process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
                '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(args.assets/f'{name}-sequence.mp4')],stdin=subprocess.PIPE)
            for i in range(SECONDS*FPS):process.stdin.write(frame(i/FPS,name,config,records[name],clips).tobytes())
            process.stdin.close()
            if process.wait():raise RuntimeError(f'Encoding failed: {name}')
        (args.assets/f'{name}-sequence.source.json').write_text(json.dumps(dict(
            source=records[name]['source'],source_sha256=records[name]['source_sha256'],
            sequence_snapshot='tools/video/enzyme-sequences.json',molecular_clip=config['movie'],
            molecular_clip_sha256=hashlib.sha256((args.assets/config['movie']).read_bytes()).hexdigest(),
            duration=SECONDS,fps=FPS,size=[W,H],mutation_positions=[s['position'] for s in records[name]['substitutions']],
            geometry='WT reference only; no mutant geometry is shown.',
            semantic_palette=dict(prediction=BLUE,mutation=ROSE)),indent=2))
        print(name,'sequence ready',flush=True)


if __name__=='__main__':main()
