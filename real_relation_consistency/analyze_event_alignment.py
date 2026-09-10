"""Offline kinematic-event alignment, not calibrated contact/online estimation.

Fixed before running: grip -> height apex -> sustained endpoint-neighborhood
entry -> terminal reference. Distances use observed terminal TCP position,
not a calibrated hole or peg tip. Main radius 30 mm; sensitivity 20/40 mm.
No orientation is used to locate events. Figure: 180x125 mm quantitative grid,
measured axis response, event timing and alignment sensitivity, Python only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.transform import Rotation
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
GRID=np.linspace(0,3,151)
LABELS=['Grip','Height apex','Near target','Terminal']


def unit(a):
    a=np.asarray(a,dtype=float)
    n=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(n<1e-10):raise ValueError('Degenerate mean axis')
    return a/n


def angle(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))


def first_run(mask,length):
    for i in range(max(len(mask)-length+1,0)):
        if mask[i:i+length].all():return i
    return None


def detect(record,radius):
    p=record['p'];start=record['start'];end=record['end'];rate=record['rate']
    if end-start<8:return None,'short_grasp_interval'
    # Terminal position inferred from the observation, not independently known.
    target=p[record['terminal_slice']].mean(0)
    distance=np.linalg.norm(p-target,axis=1)
    hold=int(np.ceil(.15*rate))+1
    entry=first_run(distance[start:end+1]<=radius,hold)
    if entry is None:return None,'no_sustained_target_entry'
    entry+=start
    if entry<=start+2:return None,'target_entry_before_lift'
    filter_size=2*int(round(.1*rate))+1
    height=median_filter(p[:,2],size=filter_size,mode='nearest')
    apex=start+int(np.argmax(height[start:entry]))
    if apex<=start or entry<=apex or end<=entry:return None,'nonordered_events'
    if height[apex]-height[start]<.02:return None,'lift_less_than_20mm'
    # Piecewise time within physically defined event intervals, no orientation warp.
    anchors=[start,apex,entry,end]
    samples=np.interp(GRID,[0,1,2,3],anchors)
    indices=np.arange(len(p),dtype=float)
    assert np.all(np.diff(indices)>0) and np.all(np.diff(anchors)>0)
    axes=unit(np.column_stack([np.interp(samples,indices,record['axis'][:,j]) for j in range(3)]))
    info=dict(grip_frame=start,apex_frame=apex,near_target_frame=entry,terminal_frame=end,
        grip_to_apex_s=(apex-start)/rate,apex_to_near_s=(entry-apex)/rate,
        near_to_terminal_s=(end-entry)/rate,lift_height_m=float(height[apex]-height[start]),
        terminal_x_m=float(target[0]),terminal_y_m=float(target[1]),terminal_z_m=float(target[2]),
        near_frame_solver_phase=int(record['phases'][entry]) if 'phases' in record else None,
        apex_frame_solver_phase=int(record['phases'][apex]) if 'phases' in record else None)
    return (axes,info),'ok'


def load_real():
    windows=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');records=[]
    for _,r in windows.iterrows():
        p=BASE/'fk_results/trajectories'/f'{r.dataset}_{int(r.episode_index):03d}.npz'
        data=np.load(p)
        records.append(dict(domain='real',condition='upright' if r.dataset=='insert_jc_fix_3' else 'tilted',
            dataset=r.dataset,episode=int(r.episode_index),seed=-1,source=str(p),
            p=data['tcp_position_m'],axis=data['upward_hand_axis'],rate=30.,
            start=int(r.grasp_start),end=int(r.window_end_exclusive)-1,
            terminal_slice=slice(int(r.window_start),int(r.window_end_exclusive))))
    return records


def load_sim():
    records=[]
    for path in sorted((BASE/'sim_10degree').glob('circular_seed_*.h5')):
        seed=int(path.stem.split('_')[-1])
        with h5py.File(path) as f:
            for key in f:
                g=f[key];gen=str(g.attrs['generator'])
                if gen not in ['baseline','pitch']:continue
                pose=np.asarray(g['tcp_pose']);phase=np.asarray(g['solver_phase'])
                q=np.asarray(g['qpos']);finger=q[:,-2:].sum(1)
                # Detect mid-closure from measured opening; search in grasp command interval.
                grasp=np.flatnonzero(phase==1);hold=np.flatnonzero(phase==2)
                if not len(grasp) or not len(hold):raise ValueError('Missing simulation grasp/lift phase')
                opening=float(np.max(finger[grasp]));closing=float(np.median(finger[hold]))
                threshold=.5*(opening+closing)
                hits=grasp[finger[grasp]<threshold]
                if not len(hits):raise ValueError('Cannot locate measured gripper closure')
                terminal=np.flatnonzero(phase==6)
                axis=-Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()[:,:,2]
                c=np.array(g['causal_delta'])
                records.append(dict(domain='sim',condition='upright' if gen=='baseline' else ('pitch_plus' if c[4]>0 else 'pitch_minus'),
                    dataset='sim_10degree',episode=int(key.split('_')[-1]),seed=seed,source=str(path)+'::'+key,
                    p=pose[:,:3],axis=axis,rate=20.,start=int(hits[0]),end=int(terminal[-1]),
                    terminal_slice=terminal,phases=phase,sim_grip_threshold=threshold))
    return records


def compare_curves(curves,nboot=2000):
    rng=np.random.default_rng(20260914);rows=[];milestones=[]
    real0=[x for x in curves if x['domain']=='real' and x['condition']=='upright']
    real1=[x for x in curves if x['domain']=='real' and x['condition']=='tilted']
    a=np.array([x['curve'] for x in real0]);b=np.array([x['curve'] for x in real1])
    point=angle(unit(a.mean(0)),unit(b.mean(0)))
    boot=[]
    for _ in range(nboot):
        aa=unit(a[rng.integers(0,len(a),len(a))].mean(0));bb=unit(b[rng.integers(0,len(b),len(b))].mean(0))
        boot.append(angle(aa,bb))
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    for s,m,l,h in zip(GRID,point,lo,hi):rows.append(dict(domain='real',condition='tilted',seed=-1,progress=s,response_deg=m,ci_low=l,ci_high=h,n_upright=len(a),n_tilt=len(b)))
    for cond in ['pitch_minus','pitch_plus']:
        paired_base=[];paired_tilt=[]
        for tilt in [x for x in curves if x['domain']=='sim' and x['condition']==cond]:
            baseline=[x for x in curves if x['domain']=='sim' and x['condition']=='upright' and x['seed']==tilt['seed']]
            if len(baseline)!=1:continue
            paired_base.append(baseline[0]['curve']);paired_tilt.append(tilt['curve'])
            vals=angle(baseline[0]['curve'],tilt['curve'])
            for s,m in zip(GRID,vals):rows.append(dict(domain='sim',condition=cond,seed=tilt['seed'],progress=s,response_deg=m,ci_low=None,ci_high=None,n_upright=1,n_tilt=1))
        aggregate=angle(unit(np.mean(paired_base,axis=0)),unit(np.mean(paired_tilt,axis=0)))
        for s,m in zip(GRID,aggregate):rows.append(dict(domain='sim',condition=cond,seed=-1,progress=s,response_deg=m,ci_low=None,ci_high=None,n_upright=len(paired_base),n_tilt=len(paired_tilt)))
    return pd.DataFrame(rows)


def make_plot(response,events,sensitivity,out):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7.086614,4.921260),layout='constrained')
    real=response[response.domain=='real'];axs[0,0].plot(real.progress,real.response_deg,color='#C07747',label='Real: event-aligned')
    axs[0,0].fill_between(real.progress,real.ci_low,real.ci_high,color='#C07747',alpha=.2)
    old=pd.read_csv(BASE/'fk_results/real_response_time_diagnostic.csv')
    axs[0,1].plot(old.normalized_grasp_time,old.mean_axis_separation_deg,color='#888888',label='Whole-interval time')
    axs[0,1].plot(real.progress/3,real.response_deg,color='#C07747',label='Equal event intervals')
    for cond,style in [('pitch_minus','--'),('pitch_plus','-')]:
        g=response[(response.domain=='sim')&(response.condition==cond)]
        mean=g[g.seed==-1].set_index('progress').response_deg;paired=g[g.seed!=-1];lo=paired.groupby('progress').response_deg.min();hi=paired.groupby('progress').response_deg.max()
        axs[0,0].plot(mean.index,mean,color='#427A9E',ls=style,label='Sim pitch −10°' if cond=='pitch_minus' else 'Sim pitch +10°')
        axs[0,0].fill_between(mean.index,lo,hi,color='#427A9E',alpha=.1)
    for ax in axs[0]:ax.axhline(10,color='#999999',ls=':',lw=.8);ax.set_ylabel('Between-condition hand-axis angle (°)');ax.legend(fontsize=5.5,frameon=False)
    axs[0,0].set(xticks=[0,1,2,3],xticklabels=LABELS,title='a  Kinematic-event response',xlabel='Piecewise event progress')
    axs[0,1].set(title='b  Dependence on alignment',xlabel='Normalized coordinate (definitions differ)')
    for radius,part in sensitivity[sensitivity.domain=='real'].groupby('radius_mm'):
        axs[1,0].plot(part.progress,part.response_deg,label=f'{radius:g} mm')
    axs[1,0].set(xticks=[0,1,2,3],xticklabels=LABELS,ylabel='Real hand-axis response (°)',xlabel='Piecewise event progress',title='c  Target-neighborhood sensitivity');axs[1,0].legend(fontsize=6,frameon=False)
    timing=['grip_to_apex_s','apex_to_near_s','near_to_terminal_s'];tick=['Grip → apex','Apex → near','Near → end']
    for i,(dataset,color) in enumerate([('insert_jc_fix_3','#427A9E'),('insert_10degree','#C07747')]):
        g=events[(events.dataset==dataset)&(events.status=='ok')]
        vals=g[timing];x=np.arange(3)+(i-.5)*.22;med=vals.median();lo=vals.quantile(.25);hi=vals.quantile(.75)
        axs[1,1].errorbar(x,med,yerr=[med-lo,hi-med],fmt='o',capsize=3,color=color,label='Upright' if i==0 else 'Tilted')
    axs[1,1].set(xticks=range(3),xticklabels=tick,ylabel='Duration (s), median and IQR',title='d  Real event interval durations');axs[1,1].legend(fontsize=6,frameon=False)
    fig.savefig(out/'event_alignment.png',dpi=600);fig.savefig(out/'event_alignment.pdf');fig.savefig(out/'event_alignment.svg');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'event_alignment');args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    records=load_real()+load_sim();all_events=[];all_response=[]
    for radius in [.02,.03,.04]:
        curves=[];events=[]
        for r in records:
            result,status=detect(r,radius);meta={k:r[k] for k in ['domain','condition','dataset','episode','seed','source']}
            meta.update(radius_mm=radius*1000,status=status)
            if result:
                curve,info=result;meta.update(info);curves.append(dict(**{k:r[k] for k in ['domain','condition','seed']},curve=curve))
            events.append(meta)
        response=compare_curves(curves);response['radius_mm']=radius*1000
        all_events.extend(events);all_response.append(response)
    events=pd.DataFrame(all_events);responses=pd.concat(all_response,ignore_index=True)
    events.to_csv(args.out/'event_frames.csv',index=False);responses.to_csv(args.out/'response_curves.csv',index=False)
    points=responses[np.isclose(responses.progress%1,0)];points.to_csv(args.out/'event_response_summary.csv',index=False)
    counts=events.groupby(['radius_mm','domain','condition','status']).size().reset_index(name='n');counts.to_csv(args.out/'extraction_counts.csv',index=False)
    mainresp=responses[responses.radius_mm==30];mainevents=events[events.radius_mm==30]
    make_plot(mainresp,mainevents,responses,args.out)
    print(counts.to_string(index=False))
    print(points[(points.radius_mm==30)&(points.seed==-1)].groupby(['domain','condition','progress']).response_deg.agg(['mean','min','max']).round(3).to_string())
    summary=dict(event_rules='Grip, maximum filtered height before sustained neighborhood entry, sustained neighborhood entry, terminal endpoint',
        real_n=111,sim_n=9,primary_radius_mm=30,sensitivity_radii_mm=[20,30,40],neighborhood_dwell_s=.15,
        terminal_reference='Measured terminal TCP mean; retrospective position proxy, not calibrated hole or tip',
        inference='Primary seed=-1: angle between group-mean axes in both domains; sim seed rows: paired diagnostics; real episode bootstrap only',
        height_filter='Centered median, 0.2 s endpoint span: 7 real frames, 5 sim frames',
        dwell_samples='ceil(0.15*rate)+1: real 6 frames (0.167 s), sim 4 frames (0.15 s)',
        limitations=['No contact labels inferred','Apex is a kinematic landmark, not guaranteed lift-controller completion',
        'Real trajectories are human demonstrations; simulation trajectories are planner-generated: domain and policy are confounded',
        'Pooled gradual response does not establish within-episode gradual adaptation',
        'Endpoint-derived geometry and future apex require offline observations',
        'Pickup context labels and real session pairing unavailable','Physical tilt axis unknown; magnitude comparison only',
        'Axes use hand TCP, not calibrated peg axes','Progress differs from frozen-law four solver phases'])
    (args.out/'protocol.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
