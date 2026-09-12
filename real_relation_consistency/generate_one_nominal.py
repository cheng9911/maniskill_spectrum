"""One real nominal assembly + independent pickup connectors + frozen pitch law.
Offline Cartesian candidates only; no IK, collision model or robot commands.
Figure contract: 180 mm quantitative grid, Python, deterministic trajectories.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation,Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from analyze_event_alignment import detect,GRID
from predict_real_transfer import source_alpha
BASE=Path(__file__).resolve().parent

def smooth(t):return 10*t**3-15*t**4+6*t**5

def interpolate(p0,R0,p1,R1,duration,rate):
    t=np.linspace(0,duration,int(np.ceil(duration*rate))+1);u=smooth(t/duration)
    p=p0+u[:,None]*(p1-p0)
    R=Slerp([0,1],Rotation.from_matrix([R0,R1]))(u).as_matrix()
    return t,p,R

def save_pose(path,t,p,R,stage):
    q=Rotation.from_matrix(R).as_quat()
    for i in range(1,len(q)):
        if np.dot(q[i],q[i-1])<0:q[i]*=-1
    frame=pd.DataFrame(dict(time_s=t,x_m=p[:,0],y_m=p[:,1],z_m=p[:,2],qx=q[:,0],qy=q[:,1],qz=q[:,2],qw=q[:,3],stage=stage))
    frame.to_csv(path,index=False)
    assert np.all(np.diff(t)>0) and np.isfinite(p).all() and np.allclose(np.linalg.det(R),1)
    v=np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(t)
    w=Rotation.from_matrix(R[:-1].transpose(0,2,1)@R[1:]).magnitude()/np.diff(t)
    return dict(samples=len(t),duration_s=float(t[-1]),max_sample_linear_speed_m_s=float(v.max()),max_sample_angular_speed_deg_s=float(np.rad2deg(w.max())))

def append(parts):
    times=[];poses=[];rots=[];stages=[];elapsed=0.
    for j,(t,p,R,label) in enumerate(parts):
        begin=0 if j==0 else 1
        if j:
            assert np.allclose(poses[-1][-1],p[0],atol=1e-10)
            assert np.allclose(rots[-1][-1],R[0],atol=1e-10)
        times.append(t[begin:]+elapsed);poses.append(p[begin:]);rots.append(R[begin:]);stages.extend([label]*(len(t)-begin));elapsed+=t[-1]
    return np.concatenate(times),np.concatenate(poses),np.concatenate(rots),np.asarray(stages)

def draw(out,pickups,nom,tilt,alpha,t):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.086614,6.4))
    fig.subplots_adjust(left=.10,right=.89,bottom=.08,top=.91,hspace=.60,wspace=.55)
    ax=fig.add_subplot(2,2,1,projection='3d')
    for name,pack in pickups.items():
        p=pack[1]*1000;ax.plot(*p.T,lw=1.1,label=name);ax.scatter(*p[0],s=12)
    ax.scatter(*(nom[0]*1000),color='black',s=18,label='Common entry')
    ax.set(title='a  Separate pickup connectors',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)');ax.legend(fontsize=5.5,frameon=False)
    ax=fig.add_subplot(2,2,2,projection='3d')
    ax.plot(*(nom*1000).T,color='#777777',label='Vertical nominal')
    ax.plot(*(tilt*1000).T,color='#327DA8',label='Predicted +10°')
    ax.scatter(*(nom[0]*1000),color='black',s=18)
    ax.set(title='b  One shared assembly trajectory',xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)');ax.legend(fontsize=5.5,frameon=False)
    # Equal coordinate scaling within each 3-D panel prevents geometric distortion.
    for a in fig.axes[:2]:
        limits=np.asarray([a.get_xlim(),a.get_ylim(),a.get_zlim()]);center=limits.mean(1);radius=np.ptp(limits,axis=1).max()/2
        a.set_xlim(center[0]-radius,center[0]+radius);a.set_ylim(center[1]-radius,center[1]+radius);a.set_zlim(center[2]-radius,center[2]+radius);a.set_box_aspect((1,1,1));a.tick_params(labelsize=6,pad=0)
        for component in [a.xaxis,a.yaxis,a.zaxis]:component.set_major_locator(MaxNLocator(4));component.labelpad=1;component.label.set_fontsize(6)
    ax=fig.add_subplot(2,2,3);ax.plot(t,10*alpha,color='#327DA8');ax.axhline(10,color='#999999',ls='--',lw=.8)
    ax.set(title='c  Applied frozen-law rotation',xlabel='Assembly time (s)',ylabel='Applied rotation (°)')
    ax=fig.add_subplot(2,2,4)
    for i,label in enumerate(['x','y','z']):ax.plot(t,(tilt[:,i]-nom[:,i])*1000,label=label)
    ax.set(title='d  Predicted position change',xlabel='Assembly time (s)',ylabel='Tilted minus nominal (mm)');ax.legend(fontsize=6,frameon=False)
    fig.suptitle('Generated Cartesian candidates — no execution validation',fontsize=8)
    fig.savefig(out/'generated_trajectories.png',dpi=600);fig.savefig(out/'generated_trajectories.pdf');fig.savefig(out/'generated_trajectories.svg');fig.savefig(out/'generated_trajectories.tiff',dpi=600);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',type=int,default=0);ap.add_argument('--rate',type=float,default=30);ap.add_argument('--assembly-duration',type=float,default=15);ap.add_argument('--offsets-json',type=Path);ap.add_argument('--out',type=Path,default=BASE/'one_nominal_generation');args=ap.parse_args()
    assert args.rate>0 and args.assembly_duration>0
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');r=table[(table.dataset=='insert_jc_fix_3')&(table.episode_index==args.episode)].iloc[0]
    source=BASE/f'fk_results/trajectories/insert_jc_fix_3_{args.episode:03d}.npz';raw=np.load(source);Rraw=Rotation.from_quat(raw['tcp_quaternion_xyzw']).as_matrix()
    rec=dict(p=raw['tcp_position_m'],axis=-Rraw[:,:,2],start=int(r.grasp_start),end=int(r.window_end_exclusive)-1,rate=30.,terminal_slice=slice(int(r.window_start),int(r.window_end_exclusive)))
    result,status=detect(rec,.03);assert status=='ok';events=result[1]
    grip=events['grip_frame'];entry=events['apex_frame'];near=events['near_target_frame'];end=events['terminal_frame']
    # Select the original nominal apex as a shared free-space assembly entry.
    # Not a contact event or guaranteed clearance pose.
    ppre=rec['p'][entry];Rpre=Rraw[entry];pgrip=rec['p'][grip];Rgrip=Rraw[grip]
    t=np.linspace(0,args.assembly_duration,int(np.ceil(args.assembly_duration*args.rate))+1)
    frame=entry+(end-entry)*smooth(t/t[-1]);assert np.all(np.diff(frame)>0)
    indices=np.arange(len(rec['p']));assert np.all(np.diff(indices)>0)
    nominal=np.column_stack([np.interp(frame,indices,rec['p'][:,j]) for j in range(3)])
    nominalR=Slerp(indices,Rotation.from_matrix(Rraw))(frame).as_matrix()
    event_s=np.interp(frame,[entry,near,end],[1,2,3]);source_profiles,_=source_alpha(.03)
    assert np.all(np.diff(GRID)>0)
    alpha=np.interp(event_s,GRID,source_profiles.mean(0))
    geometry_path=BASE/'real_predictive_transfer/geometry.json';geo=json.loads(geometry_path.read_text());axis=np.asarray(geo['axis_base']);pivot=np.asarray(geo['effective_pivot_base_m'])
    D=Rotation.from_rotvec(alpha[:,None]*np.deg2rad(10)*axis).as_matrix();tiltR=D@nominalR;tilt=pivot+np.einsum('nij,nj->ni',D,nominal-pivot)
    # Never silently force a nonzero law to start at a common pose.
    assert abs(alpha[0])<1e-10,'Law changes entry pose: require a separate transition to predicted entry'
    assert np.allclose(tilt[0],ppre) and np.allclose(tiltR[0],Rpre)
    offsets=np.array([[0,0,0],[.02,0,0],[0,.02,0]],float) if args.offsets_json is None else np.asarray(json.loads(args.offsets_json.read_text()),float)
    assert offsets.shape==(3,3) and np.isfinite(offsets).all()
    pickups={};qa={};conditions=[]
    for i,offset in enumerate(offsets,1):
        name=f'E{i}';pg=pgrip+offset;above=pg+[0,0,.05];height=max(pg[2],ppre[2])+.05
        lifted=np.array([pg[0],pg[1],height]);transferred=np.array([ppre[0],ppre[1],height])
        specs=[(above,Rgrip,pg,Rgrip,4.,'approach_open'),(pg,Rgrip,pg,Rgrip,1.,'close_gripper_event'),(pg,Rgrip,lifted,Rgrip,3.,'lift_closed'),(lifted,Rgrip,transferred,Rpre,4.,'transfer_closed'),(transferred,Rpre,ppre,Rpre,3.,'entry_closed')]
        parts=[(*interpolate(p0,R0,p1,R1,dur,args.rate),stage) for p0,R0,p1,R1,dur,stage in specs]
        pack=append(parts);pickups[name]=pack;qa[f'{name}_pickup']=save_pose(out/f'{name}_pickup.csv',*pack)
        for cname,p,rot in [('vertical',nominal,nominalR),('tilt10',tilt,tiltR)]:
            combo=append([(pack[0],pack[1],pack[2],'pickup'),(t,p,rot,'assembly')])
            # Retain the detailed gripper-event labels in the combined file.
            stages=np.concatenate([pack[3],np.full(len(t)-1,'assembly_closed')])
            qa[f'{name}_{cname}']=save_pose(out/f'{name}_{cname}.csv',combo[0],combo[1],combo[2],stages)
        conditions.append(dict(pickup_id=name,object_translation_from_reference_m=offset.tolist(),grasp_tcp_base_m=pg.tolist(),approach_tcp_base_m=above.tolist()))
    qa['assembly_vertical']=save_pose(out/'assembly_vertical.csv',t,nominal,nominalR,np.full(len(t),'assembly_closed'))
    qa['assembly_tilt10']=save_pose(out/'assembly_tilt10.csv',t,tilt,tiltR,np.full(len(t),'assembly_closed'))
    pd.DataFrame(dict(time_s=t,nominal_frame=frame,event_progress=event_s,alpha_pitch=alpha,applied_rotation_deg=alpha*10,delta_x_mm=(tilt[:,0]-nominal[:,0])*1000,delta_y_mm=(tilt[:,1]-nominal[:,1])*1000,delta_z_mm=(tilt[:,2]-nominal[:,2])*1000)).to_csv(out/'assembly_response.csv',index=False)
    manifest=dict(status='Offline Cartesian candidates; not execution validated',nominal_dataset='insert_jc_fix_3',nominal_episode=args.episode,nominal_frame_events=events,common_entry_tcp_m=ppre.tolist(),common_entry_quaternion_xyzw=Rotation.from_matrix(Rpre).as_quat().tolist(),conditions=conditions,geometry=geo,rate_hz=args.rate,assembly_duration_s=args.assembly_duration,coordinate_frame='Franka robot base',quaternion_order='xyzw',position_unit='m',pickup_method='Independent quintic Cartesian waypoint connectors; no obstacle or IK solving',nominal_retiming='Global quintic frame clock, endpoint speed zero; internal geometric corners retained',gripper_semantics='Stage labels only: close_gripper_event requests closure; no physical aperture/force command encoded',common_entry_note='Nominal height apex; not independently established contact/pre-insertion pose',law_start_alpha=float(alpha[0]),law_end_alpha=float(alpha[-1]),law_refit=False,real_geometry_calibration_episodes=geo['calibration_episodes'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'kinematic_checks.json').write_text(json.dumps(qa,indent=2)+'\n')
    inputs=[source,geometry_path,BASE/'tilt_sweep/frozen_law.npz',BASE/'predict_real_transfer.py',BASE/'analyze_event_alignment.py',Path(__file__)]+[BASE/f'tilt_sweep/circular_seed_{s}.h5' for s in [20260910,20270910,20280910]]
    (out/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
    draw(out,pickups,nominal,tilt,alpha,t)
    print(json.dumps(dict(episode=args.episode,common_entry=ppre.tolist(),conditions=conditions,alpha_start=float(alpha[0]),alpha_end=float(alpha[-1]),qa=qa),indent=2))
if __name__=='__main__':main()
