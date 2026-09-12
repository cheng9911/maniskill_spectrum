"""Offline multi-cue real-stage proposals, with explicit abstention and video QA."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;OUT=BASE/'real_phase_identification';DATA=Path('/home/rocos/sia/Sim/robomimic/dataset/data/SunJincheng');NAMES=['insert_jc_fix_3','insert_10degree']

def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def runs(mask):
 changes=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
def first(mask,start,end,dwell,rate=30):
 n=int(np.ceil(dwell*rate))+1
 for a,b in runs(mask[start:end]):
  if b-a>=n:return start+a
 return None

def proposals(r,f,pref,aref,axis_deg=5,settle_mm=20):
 p=f['p'];axis=f['axis'];v=f['v'];speed=f['speed'];angular=f['angular'];g=int(r.grasp_start);release=int(r.release_frame);stop=int(r.window_end_exclusive)
 delta=p-pref;depth=delta@aref;lateral=np.linalg.norm(delta-depth[:,None]*aref,axis=1);distance=np.linalg.norm(delta,axis=1);axis_error=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(axis,aref),axis=1),axis@aref));axial=v@aref;lat_speed=np.linalg.norm(v-axial[:,None]*aref,axis=1)
 lift=first(p[:,2]-p[g,2]>=.02,g,stop,.15)
 region=first(distance<=.08,lift if lift is not None else g,stop,.2)
 advance=None if region is None else first((distance<=.06)&(axis_error<=axis_deg)&(axial<=-.002)&(lat_speed<=.02),region,stop,.15)
 stable=(distance<=settle_mm/1000)&(axis_error<=axis_deg)&(speed<=.01)&(angular<=5)
 settle=None
 if region is not None:
  for a,b in runs(stable):
   a=max(a,region,advance if advance is not None else region);b=min(b,release)
   if b-a>=10 and b>int(r.window_start) and a<stop:
    settle=a;break
 events={k:None if v is None else int(v) for k,v in dict(grasp=g,lift_clearance=lift,assembly_region=region,axial_advance_candidate=advance,terminal_settle_candidate=settle,release=release).items()}
 stage=np.full(len(p),'approach',dtype=object);stage[g:release]='grasp_or_transport_unresolved'
 if lift is not None:stage[g:lift]='grasp_lift';stage[lift:release]='transport'
 if region is not None:stage[region:release]='alignment_or_advance_unresolved'
 if advance is not None:stage[region:advance]='prealignment';stage[advance:release]='axial_advance_candidate'
 if settle is not None:
  held=True;stable_start=None
  for i in range(settle,release):
   departure=distance[i]>(settle_mm+5)/1000 or axis_error[i]>axis_deg+2 or speed[i]>.015 or angular[i]>8
   if departure:held=False;stable_start=None
   if not held:
    if stable[i]:
     if stable_start is None:stable_start=i
     if i-stable_start>=9:
      held=True;stage[stable_start:i+1]='terminal_settle_candidate'
    else:stable_start=None
   stage[i]='terminal_settle_candidate' if held else 'post_settle_motion_unresolved'
 stage[release:]='released'
 features=dict(reference_distance_m=distance,reference_axis_error_deg=axis_error,axial_distance_proxy_m=depth,lateral_distance_proxy_m=lateral,axial_velocity_m_s=axial,lateral_speed_m_s=lat_speed)
 return events,stage,features

def main():
 OUT.mkdir(exist_ok=True);(OUT/'frame_labels').mkdir(exist_ok=True)
 windows=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');refs={};data={};meta={};inputs=[BASE/'fk_results/terminal_pose_per_episode.csv',Path(__file__),OUT/'PROTOCOL.md']
 for name in NAMES:
  c=windows[(windows.dataset==name)&(windows.episode_index%5==0)];refs[name]=dict(position=c[['tcp_x_m','tcp_y_m','tcp_z_m']].mean().values,axis=unit(c[['axis_x','axis_y','axis_z']].mean().values),episodes=c.episode_index.tolist())
  files=sorted((DATA/name/'data').glob('*/*.parquet'));inputs+=files;raw=pd.concat([pd.read_parquet(p) for p in files]);data[name]={int(ep):g.sort_values('frame_index') for ep,g in raw.groupby('episode_index')}
  files=sorted((DATA/name/'meta/episodes').glob('*/*.parquet'));inputs+=files;meta[name]=pd.concat([pd.read_parquet(p) for p in files]).set_index('episode_index')
 rows=[];sensitivity=[];review=[];aggregate=[];examples={};fixed={'insert_jc_fix_3':[0,10,20,49],'insert_10degree':[0,12,31,60]}
 for _,r in windows.iterrows():
  name=r.dataset;ep=int(r.episode_index);file=BASE/'fk_results/trajectories'/f'{name}_{ep:03d}.npz';inputs.append(file);z=np.load(file);n=len(z['tcp_position_m']);time=z['timestamp'];assert np.all(np.diff(time)>0)
  p=savgol_filter(z['tcp_position_m'],9,2,axis=0,mode='interp');axis=unit(median_filter(z['upward_hand_axis'],size=(7,1),mode='nearest'));vel=np.gradient(p,time,axis=0);R=Rotation.from_quat(z['tcp_quaternion_xyzw']).as_matrix();angular=np.r_[0,np.rad2deg(Rotation.from_matrix(R[:-1].transpose(0,2,1)@R[1:]).magnitude())/np.diff(time)];angular=median_filter(angular,7)
  f=dict(p=p,axis=axis,v=vel,speed=np.linalg.norm(vel,axis=1),angular=angular);ref=refs[name]
  events,stage,features=proposals(r,f,ref['position'],ref['axis']);variant_events=[]
  for variant,deg,mm,pos,ax in [('strict',3,15,ref['position'],ref['axis']),('main',5,20,ref['position'],ref['axis']),('loose',7,25,ref['position'],ref['axis']),('own_terminal_offline',5,20,r[['tcp_x_m','tcp_y_m','tcp_z_m']].values.astype(float),unit(r[['axis_x','axis_y','axis_z']].values.astype(float)))]:
   es,_,_=proposals(r,f,pos,ax,deg,mm);variant_events.append(es)
   for event,frame in es.items():sensitivity.append(dict(dataset=name,episode=ep,variant=variant,event=event,frame=frame,time_s=None if frame is None else float(time[frame])))
  raw=data[name][ep];assert len(raw)==n and np.array_equal(raw.frame_index.values,z['frame_index'])
  act=np.stack(raw.action);load=np.linalg.norm(act[:,8:15],axis=1);g=int(r.grasp_start);free_end=events['assembly_region'] or int(r.window_start);free_start=events['lift_clearance'] or g
  baseline=load[free_start:free_end];med=float(np.median(baseline)) if len(baseline) else 0.;mad=float(np.median(np.abs(baseline-med))) if len(baseline) else 0.;load_z=(load-med)/max(1.4826*mad,.1)
  frame=pd.DataFrame(dict(frame_index=z['frame_index'],time_s=time,stage=stage,contact_state='unknown_no_ground_truth',tcp_x_m=p[:,0],tcp_y_m=p[:,1],tcp_z_m=p[:,2],speed_m_s=f['speed'],angular_speed_deg_s=angular,gripper_state=z['gripper_state'],aux_action_tail_norm=load,aux_action_tail_robust_z=load_z,**features))
  frame.to_csv(OUT/'frame_labels'/f'{name}_{ep:03d}.csv',index=False)
  me=meta[name].loc[ep];duration=me['videos/observation.images.head/to_timestamp']-me['videos/observation.images.head/from_timestamp'];mismatch=abs(duration-n/30)>1/30
  flags=[]
  for event,idx in events.items():
   candidates=[es[event] for es in variant_events[:3] if es[event] is not None];spread=None if len(candidates)<2 else (max(candidates)-min(candidates))/30
   if idx is None:flags.append('missing_'+event)
   if spread is not None and spread>.5:flags.append('threshold_sensitive_'+event)
   rows.append(dict(dataset=name,episode=ep,reference_split='calibration' if ep%5==0 else 'reference_heldout',event=event,frame=idx,time_s=None if idx is None else float(time[idx]),status='missing' if idx is None else 'kinematic_proxy',threshold_time_spread_s=spread,threshold_detection_count=len(candidates),video_duration_mismatch=mismatch))
  if mismatch:flags.append('video_duration_mismatch')
  aggregate.append(dict(dataset=name,episode=ep,frames=n,flags=';'.join(flags),missing_event_count=sum(v is None for v in events.values()),unresolved_fraction=float(np.mean(np.char.find(stage.astype(str),'unresolved')>=0)),contact_verified=False))
  if ep in fixed[name]:
   start=max(g-30,0);end=min(int(r.release_frame)+30,n-1);chunk=int(me['videos/observation.images.head/chunk_index']);vf=int(me['videos/observation.images.head/file_index']);video=DATA/name/f'videos/observation.images.head/chunk-{chunk:03d}/file-{vf:03d}.mp4'
   review.append(dict(dataset=name,episode=ep,video=str(video),video_start_s=float(me['videos/observation.images.head/from_timestamp']+start/30),clip_duration_s=(end-start)/30,clip_start_frame=start,fps=30,source_duration_mismatch=bool(mismatch),events={k:v for k,v in events.items()},clip=f'clips/{name}_{ep:03d}.mp4'));examples[name,ep]=frame
 eventsdf=pd.DataFrame(rows);eventsdf.to_csv(OUT/'events.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False);pd.DataFrame(aggregate).to_csv(OUT/'episode_audit.csv',index=False)
 (OUT/'reference_geometry.json').write_text(json.dumps({k:{'position_m':v['position'].tolist(),'upward_hand_axis':v['axis'].tolist(),'calibration_episodes':v['episodes']} for k,v in refs.items()},indent=2)+'\n');(OUT/'video_review_manifest.json').write_text(json.dumps(review,indent=2)+'\n');(OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
 coverage=eventsdf.groupby(['dataset','event']).agg(n=('episode','size'),detected=('frame','count'),median_threshold_spread_s=('threshold_time_spread_s','median'));coverage.to_csv(OUT/'event_coverage.csv');print(coverage.to_string());print('Flags',pd.DataFrame(aggregate).missing_event_count.value_counts().to_dict(),flush=True)
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(2,2,figsize=(7.086614,4.8),layout='constrained')
 for col,name in enumerate(NAMES):
  frame=examples[name,0];es=eventsdf[(eventsdf.dataset==name)&(eventsdf.episode==0)];t=frame.time_s
  axs[0,col].plot(t,frame.reference_distance_m*1000,color='#327DA8',label='Distance to reference TCP');axs[0,col].plot(t,frame.speed_m_s*1000,color='#C07747',label='TCP speed (mm/s)')
  axs[1,col].plot(t,frame.reference_axis_error_deg,color='#327DA8',label='Axis discrepancy');axs[1,col].plot(t,frame.angular_speed_deg_s,color='#C07747',label='Angular speed (°/s)')
  for i,(_,ev) in enumerate(es.iterrows()):
   if pd.notna(ev.time_s):
    for row in range(2):axs[row,col].axvline(ev.time_s,color='#aaaaaa',ls=':',lw=.7)
  axs[0,col].set(title=('a  Upright' if col==0 else 'b  Tilted')+' episode 0',ylabel='Distance (mm) / speed (mm/s)');axs[1,col].set(title=('c' if col==0 else 'd')+'  Orientation cues',ylabel='Angle (°) / speed (°/s)',xlabel='Episode time (s)')
  for row in range(2):axs[row,col].legend(fontsize=5.5,frameon=False)
 fig.suptitle('Real kinematic stage proposals; dotted lines are not verified contact events',fontsize=8)
 fig.savefig(OUT/'phase_features.png',dpi=600);fig.savefig(OUT/'phase_features.pdf');fig.savefig(OUT/'phase_features.svg');fig.savefig(OUT/'phase_features.tiff',dpi=600);plt.close(fig)
if __name__=='__main__':main()
