"""Exploratory terminal-axis consistency; never estimates unobserved hole poses.

Figure contract: quantify whether terminal hand-axis tilt differs with the
reported hole tilt; expose position mixing and window sensitivity. Quantitative
grid, 180 x 135 mm, Python only, editable PDF/SVG plus PNG, all episodes kept.
The terminal window is a gripper-based proxy, not a verified insertion label.
"""
from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XYZ = [[0,0,.333],[0,0,0],[0,-.316,0],[.0825,0,0],[-.0825,.384,0],[0,0,0],[.088,0,0]]
ROLL = [0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2]


def fk(q):
    """Base -> default hand TCP. URDF origin transform precedes each Rz(q)."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Expected finite (N,7) joint angles in radians')
    t = np.tile(np.eye(4), (len(q),1,1))
    for j in range(7):
        origin = np.eye(4)
        origin[:3,:3] = Rotation.from_euler('x', ROLL[j]).as_matrix()
        origin[:3,3] = XYZ[j]
        z = np.tile(np.eye(4), (len(q),1,1))
        z[:,:3,:3] = Rotation.from_euler('z', q[:,j]).as_matrix()
        t = t @ origin @ z
    tool = np.eye(4)
    tool[:3,:3] = Rotation.from_euler('z', -np.pi/4).as_matrix()
    tool[2,3] = .107 + .1034
    return t @ tool


def fk_checks():
    """Independent modified-DH formulation, plus rigid transform invariants."""
    q = np.random.default_rng(17).uniform(-1,1,(20,7))
    out = fk(q)
    a = [0,0,0,.0825,-.0825,0,.088,0]
    d = [.333,0,.316,0,.384,0,0,.107]
    errors=[]
    for qi, ti in zip(q,out):
        t=np.eye(4)
        for ai,di,al,th in zip(a,d,ROLL+[0],list(qi)+[0]):
            ca,sa,ct,st=np.cos(al),np.sin(al),np.cos(th),np.sin(th)
            t=t@np.array([[ct,-st,0,ai],[st*ca,ct*ca,-sa,-di*sa],
                         [st*sa,ct*sa,ca,di*ca],[0,0,0,1]])
        tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('z',-np.pi/4).as_matrix();tool[2,3]=.1034
        errors.append(np.max(np.abs(t@tool-ti)))
    assert max(errors)<1e-12
    assert np.allclose(out[:,:3,:3].transpose(0,2,1)@out[:,:3,:3],np.eye(3))
    assert np.allclose(np.linalg.det(out[:,:3,:3]),1)
    return dict(max_urdf_mdh_difference=float(max(errors)), tested_configurations=len(q))


def mean_axis(axes):
    v=np.mean(axes,axis=0)
    return v/np.linalg.norm(v)


def angle(a,b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))


def terminal_window(state, pos, fps, threshold=.5, window=.5, guard=.2):
    # Close short gaps caused by small grip adjustments, preserve original edges.
    raw=state[:,7]<threshold
    closed=raw | binary_closing(raw,structure=np.ones(int(fps*.4)+1))
    starts=np.flatnonzero(np.diff(np.r_[False,closed].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[closed,False].astype(int))==-1)+1
    segments=[(s,e) for s,e in zip(starts,ends) if e-s>=fps and e<len(state)]
    if not segments:
        return None
    # Choose the grasp interval with largest endpoint displacement, independently of tilt.
    s,e=max(segments,key=lambda se:np.linalg.norm(pos[se[1]-1]-pos[se[0]]))
    end=e-round(guard*fps); start=end-round(window*fps)
    if start<s or end<=start:
        return None
    return dict(grasp_start=int(s),release_frame=int(e),window_start=int(start),
                window_end_exclusive=int(end),n_grasp_segments=len(segments),
                transport_distance_m=float(np.linalg.norm(pos[e-1]-pos[s])))


def bootstrap_means(values,rng,n=5000):
    values=np.asarray(values)
    return values[rng.integers(0,len(values),(n,len(values)))].mean(1)


def collect_real(roots,out):
    rows,sensitivity,curves=[],[],[]
    for root in roots:
        df=pd.concat([pd.read_parquet(p) for p in sorted((root/'data').glob('*/*.parquet'))],ignore_index=True)
        for ep,g in df.groupby('episode_index',sort=True):
            g=g.sort_values('frame_index'); state=np.stack(g['observation.state'])
            t=fk(state[:,:7]);pos=t[:,:3,3]; axes=-t[:,:3,2]
            quat=Rotation.from_matrix(t[:,:3,:3]).as_quat()
            tilt=angle(axes,np.array([0,0,1]))
            key=f'{root.name}_{ep:03d}'
            np.savez_compressed(out/'trajectories'/f'{key}.npz',
                frame_index=g.frame_index.to_numpy(),timestamp=g.timestamp.to_numpy(),
                tcp_position_m=pos,tcp_quaternion_xyzw=quat,upward_hand_axis=axes,
                tilt_deg=tilt,gripper_state=state[:,7])
            w=terminal_window(state,pos,30)
            row=dict(dataset=root.name,episode_index=int(ep),frames=len(g),
                     extraction_status='ok' if w else 'no_terminal_window',
                     hole_tilt_reported_deg=0 if root.name=='insert_jc_fix_3' else 10)
            if w:
                sl=slice(w['window_start'],w['window_end_exclusive']);ax=mean_axis(axes[sl]);p=pos[sl].mean(0)
                row.update(w);row.update(dict(zip(['axis_x','axis_y','axis_z'],ax)))
                row.update(dict(zip(['tcp_x_m','tcp_y_m','tcp_z_m'],p)))
                row['terminal_tilt_deg']=float(angle(ax,np.array([0,0,1])))
                row['window_axis_spread_max_deg']=float(angle(axes[sl],ax).max())
                row.update(dict(zip(['pickup_tcp_x_m','pickup_tcp_y_m','pickup_tcp_z_m'],pos[w['grasp_start']])))
                # Grasp-relative time, not a claimed simulation phase mapping.
                idx=np.arange(w['grasp_start'],w['window_end_exclusive'])
                xp=np.linspace(0,1,len(idx))
                assert len(xp)>1 and np.all(np.diff(xp)>0)
                for progress,value in zip(np.linspace(0,1,101),np.interp(np.linspace(0,1,101),xp,tilt[idx])):
                    curves.append(dict(dataset=root.name,episode_index=int(ep),grasp_progress=progress,tilt_deg=value))
            rows.append(row)
            for th in [.4,.5,.6]:
                for width in [.25,.5,1.0]:
                    ww=terminal_window(state,pos,30,threshold=th,window=width)
                    if ww:
                        ax=mean_axis(axes[ww['window_start']:ww['window_end_exclusive']])
                        sensitivity.append(dict(dataset=root.name,episode_index=int(ep),threshold=th,window_s=width,
                                                terminal_tilt_deg=float(angle(ax,np.array([0,0,1]))),
                                                **dict(zip(['axis_x','axis_y','axis_z'],ax))))
    real=pd.DataFrame(rows);real.to_csv(out/'terminal_pose_per_episode.csv',index=False)
    sens=pd.DataFrame(sensitivity);sens.to_csv(out/'window_sensitivity.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'grasp_progress_curves.csv',index=False)
    return real,sens,pd.DataFrame(curves)


def collect_sim(simdir,out):
    rows=[]
    for p in sorted(simdir.glob('circular_honest_seed_*.h5')):
        seed=int(p.stem.split('_')[-1])
        with h5py.File(p) as f:
            for key in f:
                g=f[key];gen=str(g.attrs.get('generator'))
                if gen not in ['baseline','roll','pitch']:continue
                inds=np.flatnonzero(np.array(g['solver_phase'])==6)
                if not len(inds):continue
                pose=np.array(g['peg_pose'])[inds];q=pose[:,3:]
                axes=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()[:,:,2]
                ax=mean_axis(axes)
                c=np.array(g['causal_delta'])
                rows.append(dict(seed=seed,episode=key,generator=gen,
                    input_tilt_deg=float(np.degrees(np.linalg.norm(c[3:5]))),
                    output_tilt_deg=float(angle(ax,np.array([0,0,1]))),
                    success=bool(np.array(g['success']).any()),phase6_frames=len(inds)))
    sim=pd.DataFrame(rows);sim.to_csv(out/'simulation_terminal_tilt.csv',index=False)
    return sim


def summarize(real,sens,sim):
    rng=np.random.default_rng(20260910);summary={}
    for name,g in real.groupby('dataset'):
        values=g.terminal_tilt_deg.dropna().to_numpy();b=bootstrap_means(values,rng)
        summary[name]=dict(n_recorded=len(g),n_extracted=len(values),mean_deg=float(values.mean()),
            median_deg=float(np.median(values)),sd_deg=float(values.std(ddof=1)),
            min_deg=float(values.min()),max_deg=float(values.max()),
            exploratory_episode_bootstrap_mean_ci95_deg=np.quantile(b,[.025,.975]).tolist())
    u=real[real.dataset=='insert_jc_fix_3'].terminal_tilt_deg.dropna().to_numpy()
    t=real[real.dataset=='insert_10degree'].terminal_tilt_deg.dropna().to_numpy()
    diff=bootstrap_means(t,rng)-bootstrap_means(u,rng)
    means=[mean_axis(g[['axis_x','axis_y','axis_z']].dropna().to_numpy()) for _,g in real.groupby('dataset')]
    axis_boot=[]
    for _,g in real.groupby('dataset'):
        axes=g[['axis_x','axis_y','axis_z']].dropna().to_numpy()
        boot=axes[rng.integers(0,len(axes),(5000,len(axes)))].mean(1)
        axis_boot.append(boot/np.linalg.norm(boot,axis=1,keepdims=True))
    summary['contrast']=dict(mean_tilt_magnitude_difference_deg=float(t.mean()-u.mean()),
        exploratory_episode_bootstrap_ci95_deg=np.quantile(diff,[.025,.975]).tolist(),
        mean_axis_separation_deg=float(angle(means[0],means[1])),
        mean_axis_separation_exploratory_ci95_deg=np.quantile(angle(axis_boot[0],axis_boot[1]),[.025,.975]).tolist(),
        magnitude_difference_over_reported_10deg=float((t.mean()-u.mean())/10),
        interpretation='Descriptive terminal hand-axis contrast, not fitted alpha or equivalence test')
    sm=sens.groupby(['threshold','window_s','dataset']).terminal_tilt_deg.mean().unstack('dataset')
    summary['sensitivity_difference_range_deg']=[float((sm.insert_10degree-sm.insert_jc_fix_3).min()),float((sm.insert_10degree-sm.insert_jc_fix_3).max())]
    separations=[]
    for _,g in sens.groupby(['threshold','window_s']):
        ax=[mean_axis(v[['axis_x','axis_y','axis_z']].to_numpy()) for _,v in g.groupby('dataset')]
        separations.append(float(angle(ax[0],ax[1])))
    summary['sensitivity_axis_separation_range_deg']=[min(separations),max(separations)]
    selected=sim[sim.generator!='baseline']
    summary['simulation']=dict(n_tilt_attempts=len(selected),input_degrees=sorted(selected.input_tilt_deg.unique().tolist()),
        mean_output_tilt_deg=float(selected.output_tilt_deg.mean()),
        mean_output_input_ratio=float((selected.output_tilt_deg/selected.input_tilt_deg).mean()),
        role='Empirical simulated peg-axis response at +/-15 degrees, not a matched 10-degree trial or learned P')
    summary['limitations']=['Unverified terminal phase and success labels', 'Uncalibrated hand-to-peg axis alignment',
        'Hole tilt direction and base-to-table leveling not supplied', 'Pickup-position and collection-session confounding; hole position fixed per user',
        'Bootstrap treats episodes as exchangeable within a dataset; no cross-session inference',
        'No full generator law or phase-dependent equivalence claim']
    return summary


def plot(real,sens,curves,summary,out):
    plt.rcParams.update({'font.size':7,'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7.086614,5.314961),layout='constrained')
    names=['insert_jc_fix_3','insert_10degree'];labels=['Upright hole','10° tilted hole'];colors=['#427A9E','#C07747']
    rng=np.random.default_rng(19)
    for i,(name,label,color) in enumerate(zip(names,labels,colors)):
        g=real[real.dataset==name]
        axs[0,0].scatter(g.tcp_x_m*1000,g.tcp_y_m*1000,s=12,alpha=.7,color=color,label=f'{label} (n={g.terminal_tilt_deg.notna().sum()})')
        axs[0,1].scatter(i+rng.uniform(-.16,.16,len(g)),g.terminal_tilt_deg,s=11,alpha=.6,color=color)
        stats=summary[name];ci=stats['exploratory_episode_bootstrap_mean_ci95_deg'];m=stats['mean_deg']
        axs[0,1].errorbar(i+.25,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='o',color='black',capsize=3,ms=4)
        c=curves[curves.dataset==name].groupby('grasp_progress').tilt_deg
        med=c.median();lo=c.quantile(.25);hi=c.quantile(.75)
        axs[1,0].plot(med.index,med,color=color,label=label)
        axs[1,0].fill_between(med.index,lo,hi,color=color,alpha=.18)
    axs[0,0].set(xlabel='Terminal hand TCP x (mm)',ylabel='Terminal hand TCP y (mm)',title='a  Pre-release TCP positions');axs[0,0].legend(fontsize=6)
    axs[0,0].set_aspect('equal',adjustable='datalim')
    axs[0,1].set(xticks=[0,1],xticklabels=labels,ylabel='Hand-axis tilt from base vertical (°)',title='b  Terminal orientation')
    axs[0,1].axhline(10,color='#999999',ls='--',lw=.8)
    axs[1,0].set(xlabel='Normalized grasp-to-pre-release time',ylabel='Hand-axis tilt (°)',title='c  Median and interquartile range')
    axs[1,0].legend(fontsize=6)
    for th,style in zip([.4,.5,.6],['o-','s--','^-']):
        widths=[];values=[]
        for width,g in sens[sens.threshold==th].groupby('window_s'):
            axes=[mean_axis(v[['axis_x','axis_y','axis_z']].to_numpy()) for _,v in g.groupby('dataset')]
            widths.append(width);values.append(float(angle(axes[0],axes[1])))
        axs[1,1].plot(widths,values,style,ms=4,lw=1,label=f'Grip threshold {th}')
    axs[1,1].axhline(10,color='#999999',ls=':',lw=.8)
    axs[1,1].set(xlabel='Pre-release window (s)',ylabel='Angle between group mean axes (°)',title='d  Segmentation sensitivity')
    axs[1,1].legend(fontsize=6)
    fig.savefig(out/'terminal_axis_consistency.png',dpi=600)
    fig.savefig(out/'terminal_axis_consistency.pdf')
    fig.savefig(out/'terminal_axis_consistency.svg')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',type=Path,nargs='+',required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sim-dir',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'trajectories').mkdir(exist_ok=True)
    checks=fk_checks();real,sens,curves=collect_real(args.datasets,args.out)
    sim=collect_sim(args.sim_dir,args.out);summary=summarize(real,sens,sim);summary['fk_checks']=checks
    (args.out/'fk_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(real,sens,curves,summary,args.out);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
