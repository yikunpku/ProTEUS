"""Render the real 7Y9A GS trajectory used in ScientificWorldmodel.pptx.

Run in ChimeraX with --notools --nocolor --exit --script
'tools/video/render_projection_molecules.py --preview'. All rendered coordinate
sets come from the original DCD without coordinate interpolation. Fixed shared
cameras map the overview box exactly to the local view. The read-only source is
a distance-constraint demonstration, not the paper's seven-family benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zipfile
import numpy as np
from chimerax.atomic import AtomicStructure
from chimerax.core.commands import run
from chimerax.geometry import Place
from chimerax.graphics.opengl import Silhouette
from chimerax.md_crds.read_coords import read_coords

PALETTE=dict(helix='#7BD0AC',strand='#6B9BD1',coil='#E6DED3',shell='#BFC7D1',
             carbon='#AD966E',nitrogen='#5290D3',oxygen='#DA6457',phosphorus='#D2AB59')


def cmd(text):return run(session,text,log=False)
def digest(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def opaque_outline(self,render):
    fb=render.pop_framebuffer()
    render.current_framebuffer().copy_from_framebuffer(fb,depth=False)


def lighting():
    for text in ['set bgColor white','lighting gentle','lighting multiShadow 32 msMapSize 128 msDepthBias 0.05',
                 'lighting ambientIntensity 0.80 intensity 0.75 fillIntensity 0.20',
                 'lighting direction 1,-1,-1 fillDirection -0.4,-0.2,-1',
                 'lighting shadows false depthCue false','material dull',
                 'material reflectivity 0.80 ambientReflectivity 0.80 specularReflectivity 0.10 exponent 25 transparentCastShadows false',
                 'graphics silhouettes true width 0.65 color #586B76 depthJump 0.01']:
        cmd(text)
    Silhouette.finish_silhouette_drawing=opaque_outline


def setup(topology,trajectory,local):
    cmd('close all');cmd(f'open "{topology}"')
    model=session.models.list(type=AtomicStructure)[0]
    # Preserve the topology's native secondary structure throughout the solve.
    ss_hash=digest(model.residues.ss_types)
    read_coords(session,str(trajectory),model,'dcd',replace=True)
    ids=sorted(model.coordset_ids)
    assert len(ids)==355 and model.num_atoms==1592
    hashes={i:digest(model.coordset(ids[i]).xyzs) for i in range(len(ids))}
    cmd('hide atoms; hide pbonds; hide cartoons; clip off; camera ortho')
    for group in model.pbg_map.values():
        if group.name=='chain trace':group.display=False
    cmd('show protein cartoons')
    cmd(f'color protein {PALETTE["coil"]} target c')
    cmd(f'color protein & helix {PALETTE["helix"]} target c')
    cmd(f'color protein & strand {PALETTE["strand"]} target c')
    cmd('cartoon style protein modeHelix default xsection oval width 1.5 thickness 0.26')
    cmd('cartoon style protein & strand arrows true width 1.7 thickness 0.24')
    cmd('cartoon suppress false')
    cmd('show ~protein & ~solvent atoms; style ~protein & ~solvent ball')
    cmd('size ~protein & ~solvent ballScale 0.27 stickRadius 0.12')
    cmd(f'color ~protein & ~solvent {PALETTE["carbon"]} target ab')
    if local:
        cmd('transparency protein 87 target c')
        cmd('show #1 atoms; style #1 stick; size #1 stickRadius 0.035')
        cmd('color #1 #D5DEE2 target ab; transparency #1 48 target ab')
        foreground='(#1/A:71-80 | #1/C:1)'
        cmd(f'style {foreground} ball; size {foreground} ballScale 0.25 stickRadius 0.10')
        cmd(f'transparency {foreground} 0 target ab')
        for element,key in [('C','carbon'),('N','nitrogen'),('O','oxygen'),('P','phosphorus')]:
            cmd(f'color {foreground} & {element} {PALETTE[key]} target ab')
    else:
        cmd('surface protein enclose protein probeRadius 1.4 gridSpacing 0.7')
        cmd(f'color protein {PALETTE["shell"]} target s; transparency protein 85 target s')
    lighting()
    return model,ids,hashes,ss_hash


def selected_snapshot(t):
    # Initial hold, real recorded snapshots, final 2.2-second hold. This is
    # time compression, never invented positions between solver updates.
    return round(min(1,max(0,(t-1.2)/7.4))*354)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-root',type=Path,default=Path('/Users/mhg/Projects/protein-rendering/gs-rendering'))
    parser.add_argument('--frames-dir',type=Path,default=Path('/private/tmp/proteus-projection-molecules'))
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--seconds',type=float,default=10.8)
    parser.add_argument('--fps',type=int,default=24)
    args=parser.parse_args(sys.argv[1:])
    root=args.trajectory_root;case_dir=root/'tmp/case';solve_dir=root/'tmp/cartesian'
    paths=dict(topology=case_dir/'topology.cif',trajectory=solve_dir/'trajectory.dcd',
               frame_plan=solve_dir/'frame_plan.json',summary=solve_dir/'summary.json',
               case=case_dir/'case.npz',atom_order=case_dir/'atom_order.json',
               config=root/'configs/7y9a_gauss_seidel.json',solver=root/'scripts/gauss_seidel.py',
               source_cif=root/'raw_data/7y9a.cif',trace=solve_dir/'trace.json')
    config=json.loads(paths['config'].read_text())
    records=json.loads(paths['frame_plan'].read_text())['frames']
    summary=json.loads(paths['summary'].read_text())
    case=np.load(paths['case']);perm=np.asarray(json.loads(paths['atom_order'].read_text())['perm'])
    hero=perm[case['hero_atoms']]
    z=np.asarray(config['camera']['zoom_direction'],float);z/=np.linalg.norm(z)
    x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x);basis=np.column_stack([x,y,z])
    # Fit the union of the entire trajectory, then keep both cameras fixed.
    model,ids,_,_=setup(paths['topology'],paths['trajectory'],False)
    coordinates=np.stack([model.coordset(i).xyzs for i in ids])
    host_coordinates=coordinates[:,perm,:].astype(float)
    pairs,targets=case['pairs'],case['targets']
    residual_error=0.
    for xyz,record in zip(host_coordinates,records):
        residual=np.linalg.norm(xyz[pairs[:,0]]-xyz[pairs[:,1]],axis=1)-targets
        residual_error=max(residual_error,abs(float(abs(residual).max())-record['max_violation']),
                           abs(float(np.sqrt(np.mean(residual*residual)))-record['rms_violation']))
    initial=case['coords']+np.random.default_rng(101).normal(0,.3,size=case['coords'].shape)
    initial_error=float(np.abs(initial-host_coordinates[0]).max())
    # Re-execute the source's pure forward pair update, not a reimplementation.
    # exec avoids creating bytecode inside the read-only source project.
    solver_namespace={}
    exec(compile(paths['solver'].read_text(),str(paths['solver']),'exec'),solver_namespace)
    flat=host_coordinates[0].reshape(-1).tolist();projection_error=0.
    for k in range(150):
        solver_namespace['project'](flat,pairs,targets,k,1.)
        projection_error=max(projection_error,float(np.abs(np.asarray(flat).reshape(-1,3)-host_coordinates[8+k]).max()))
    assert max(residual_error,initial_error,projection_error)<1e-5,'Original GS trajectory verification failed'
    validation=dict(snapshot_residuals_verified=355,max_residual_recompute_error_A=residual_error,
                    initial_perturbation_max_error_A=initial_error,
                    original_GS_pair_updates_replayed=150,max_replayed_coordinate_error_A=projection_error)
    pptx=Path('/Users/mhg/Desktop/ScientificWorldmodel.pptx')
    if pptx.exists():
        with zipfile.ZipFile(pptx) as archive:
            matches={}
            for member,clip in [('media37.mp4','7y9a_gs_cartesian_overview_20s.mp4'),
                                ('media38.mp4','7y9a_gs_cartesian_local_20s.mp4')]:
                embedded=hashlib.sha256(archive.read('ppt/media/'+member)).hexdigest()
                native=hashlib.sha256((root/'outputs/clips'/clip).read_bytes()).hexdigest()
                assert embedded==native,'Slides and native source movie differ'
                matches[member]=dict(native_clip=clip,sha256=embedded)
            validation['supplied_slides_source_identity']=matches
    centers={};viewports={}
    for kind,pts,size,padding in [('global',coordinates,[600,680],1.13),
                                  ('local',coordinates[:,hero,:],[1000,510],1.30)]:
        projected=pts.reshape(-1,3)@basis
        low,high=projected.min(0),projected.max(0)
        centers[kind]=basis@((low+high)/2)
        fw=max(high[0]-low[0],(high[1]-low[1])*size[0]/size[1])*padding
        viewports[kind]=dict(size=size,field_width_A=float(fw))
    count=round(args.seconds*args.fps)
    indices=[0,round(4*args.fps),count-1] if args.preview else range(count)
    metadata=dict(structure='7Y9A · real GS distance-constraint restoration demonstration',
        fps=args.fps,frames=count,seconds=args.seconds,palette=PALETTE,viewports=viewports,preview=args.preview,
        coordinate_motion='Exact original DCD snapshots; no interpolation, rigid alignment or camera animation.',
        trajectory_case=dict(atoms=1592,snapshots=355,perturbation_sigma_A=.3,seed=101,
            constraint_tiers=['1–2 bond distance','1–3 angle-derived distance','1–4 torsion-derived distance'],
            constraints=6418,source_summary=summary),
        scientific_scope='A real sequential GS restoration demo used in the supplied slides, distinct from the paper seven-family benchmark. The endpoint is not claimed to pass all seven checks.',
        source_validation=validation,
        sources=[dict(role=k,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()],
        frames_metadata={})
    args.frames_dir.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    for kind,view in viewports.items():
        folder=args.frames_dir/kind;folder.mkdir(exist_ok=True)
        model,ids,hashes,ss_hash=setup(paths['topology'],paths['trajectory'],kind=='local')
        center=centers[kind];camera=session.main_view.camera
        camera.position=Place(matrix=np.column_stack([basis,center+basis[:,2]*200]))
        camera.field_width=view['field_width_A']
        width,height=view['size'];field_height=camera.field_width*height/width
        def project(xyz):
            delta=np.asarray(xyz)-center
            return [float(.5+delta@basis[:,0]/camera.field_width),float(.5-delta@basis[:,1]/field_height)]
        previous_snapshot=None;previous_path=None
        for index in indices:
            snapshot=selected_snapshot(index/args.fps)
            model.active_coordset_id=ids[snapshot]
            record=records[snapshot]
            info=metadata['frames_metadata'].setdefault(str(index),dict(source_snapshot=snapshot,
                phase='perturbed' if index/args.fps<1.2 else ('projected' if index/args.fps>=8.6 else 'projecting'),
                rms_residual_A=record['rms_violation'],max_residual_A=record['max_violation'],
                source_sweep=record['sweep'],source_constraint=record['constraint']))
            pair=record.get('pair')
            points=[project(p) for p in model.atoms.coords[perm[np.asarray(pair)]]] if pair is not None else []
            info[kind]=dict(active_pair=points,camera_matrix=camera.position.matrix.tolist(),coordinates_sha256=hashes[snapshot])
            if kind=='global':
                local=viewports['local'];hw=local['field_width_A']/2;hh=hw*local['size'][1]/local['size'][0]
                info['region_corners']=[project(centers['local']+sx*hw*basis[:,0]+sy*hh*basis[:,1])
                    for sx,sy in [(-1,1),(1,1),(1,-1),(-1,-1)]]
            target=folder/f'{index:04d}.png'
            if snapshot==previous_snapshot:target.write_bytes(previous_path.read_bytes())
            else:cmd(f'save "{target}" width {width} height {height} supersample 2')
            assert hashes[snapshot]==digest(model.atoms.coords),'Rendered coordinates differ from source trajectory'
            assert ss_hash==digest(model.residues.ss_types),'Secondary structure assignment changed'
            previous_snapshot=snapshot;previous_path=target
            if index%48==0:print(f'GS_TRAJECTORY {kind} {index}/{count} snapshot={snapshot} {time.monotonic()-start:.1f}s',flush=True)
        metadata[kind+'_secondary_structure_sha256']=ss_hash
    metadata['trajectory_integrity_verified']=True
    metadata['elapsed_seconds']=time.monotonic()-start
    (args.frames_dir/'source.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('GS_TRAJECTORY_RENDER_DONE',metadata['elapsed_seconds'],flush=True)
    cmd('exit')


main()
