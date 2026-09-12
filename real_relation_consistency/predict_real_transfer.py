"""Retrospective, TCP-calibrated real transfer; not independently calibrated transfer.

Figure contract: quantitative grid, 180 mm, Python; all episodes retained.
Geometry calibration and law fitting are distinct. Target event timing is offline.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_event_alignment import load_real, detect, GRID, unit, angle
BASE=Path(__file__).resolve().parent
OUT=BASE/'real_predictive_transfer'
SEEDS=[20260910,20270910,20280910]
METHODS=['No adaptation','Identity','Event ramp','Frozen law']
COLORS=['#999999','#8064A2','#679C91','#327DA8']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def export(fig,stem):
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/f'{stem}.tiff',dpi=600)

def resample_real(record,radius):
    result,status=detect(record,radius)
    if status!='ok':raise ValueError((record['episode'],radius,status))
    axes,info=result
    anchors=[info[k] for k in ['grip_frame','apex_frame','near_target_frame','terminal_frame']]
    assert np.all(np.diff(anchors)>0)
    t=np.interp(GRID,[0,1,2,3],anchors)
    raw=np.load(record['source']);idx=np.arange(len(record['p']))
    assert np.all(np.diff(idx)>0)
    pos=np.column_stack([np.interp(t,idx,record['p'][:,j]) for j in range(3)])
    rot=Slerp(idx,Rotation.from_quat(raw['tcp_quaternion_xyzw']))(t).as_matrix()
    # Derive axes from the same interpolated SO(3) used for predicted poses.
    return dict(p=pos,R=rot,axis=-rot[:,:,2],events=info)

def source_alpha(radius):
    law=np.load(BASE/'tilt_sweep/frozen_law.npz');all_alpha=[];rows=[]
    for seed in SEEDS:
        with h5py.File(BASE/f'tilt_sweep/circular_seed_{seed}.h5') as f:
            groups=[g for g in f.values() if str(g.attrs['generator'])=='baseline']
            assert len(groups)==1
            g=groups[0];phase=np.array(g['solver_phase']);p=np.array(g['tcp_pose']);q=np.array(g['qpos']);finger=q[:,-2:].sum(1)
            grasp=np.flatnonzero(phase==1);lift=np.flatnonzero(phase==2)
            threshold=.5*(finger[grasp].max()+np.median(finger[lift]));start=grasp[finger[grasp]<threshold][0]
            terminal=np.flatnonzero(phase==6)
            rec=dict(p=p[:,:3],axis=-Rotation.from_quat(p[:,[4,5,6,3]]).as_matrix()[:,:,2],start=int(start),end=int(terminal[-1]),rate=20.,terminal_slice=terminal,phases=phase)
            result,status=detect(rec,radius);assert status=='ok'
            anchors=[result[1][k] for k in ['grip_frame','apex_frame','near_target_frame','terminal_frame']]
            query=np.interp(GRID,[0,1,2,3],anchors)
            times=np.concatenate([np.linspace(np.flatnonzero(phase==k)[0],np.flatnonzero(phase==k)[-1],25) for k in [3,4,5,6]])
            assert np.all(np.diff(times)>0)
            a=np.interp(query,times,law[f'seed_{seed}_alpha_pitch'],left=0.)
            all_alpha.append(a)
            rows.extend(dict(radius_mm=radius*1000,seed=seed,progress=s,alpha=v) for s,v in zip(GRID,a))
    return np.asarray(all_alpha),rows

def predict(nominal,axis,pivot,alpha):
    delta=Rotation.from_rotvec(np.deg2rad(10)*np.asarray(alpha)[:,None]*axis).as_matrix()
    rot=delta@nominal['R'];p=pivot+np.einsum('nij,nj->ni',delta,nominal['p']-pivot)
    return dict(p=p,R=rot,axis=-rot[:,:,2])

def score(pred,target,mask):
    ae=angle(pred['axis'],target['axis']);pe=np.linalg.norm(pred['p']-target['p'],axis=1)*1000
    return float(np.sqrt(np.mean(ae[mask]**2))),float(np.sqrt(np.mean(pe[mask]**2)))

def plot_results(table,matches,nominal_rows,tilted_rows,archive,representative,sensitivity,alpha):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7.086614,5.3),layout='constrained')
    ax=axs[0,0]
    ax.scatter(nominal_rows.pickup_tcp_x_m*1000,nominal_rows.pickup_tcp_y_m*1000,s=13,facecolors='none',edgecolors='#666666',label='Nominal (50)')
    for split,marker in [('Calibration','s'),('Evaluation','o')]:
        ids=matches[matches.split==split].target_episode
        g=tilted_rows.loc[ids];ax.scatter(g.pickup_tcp_x_m*1000,g.pickup_tcp_y_m*1000,s=12,marker=marker,label=f'{split} ({len(g)})')
    ax.set(title='a  Observed pickup support',xlabel='TCP x (mm)',ylabel='TCP y (mm)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6,frameon=False)
    ax=axs[0,1];ep=representative;target=archive[f'e{ep}_target_axis'];base=archive[f'e{ep}_nominal_axis']
    ax.plot(GRID,angle(base,target),color='#C07747',lw=1.8,label='Real demonstration')
    for method,color in zip(METHODS,COLORS):ax.plot(GRID,angle(base,archive[f'e{ep}_{method}_axis']),color=color,lw=1.1,label=method)
    ax.set(title=f'b  Representative response: episode {ep}',xlabel='Event progress',ylabel='Axis change from nominal (°)',xticks=[0,1,2,3],xticklabels=['Grip','Apex','Near','End']);ax.legend(fontsize=5.5,frameon=False,loc='lower right',bbox_to_anchor=(1,.12))
    ax=axs[1,0]
    for i,(method,color) in enumerate(zip(METHODS,COLORS)):
        vals=table[(table.method==method)&(table.segment=='Apex to end')].axis_rmse_deg.values
        ax.scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=5,alpha=.55,color=color)
        q=np.quantile(vals,[.25,.5,.75]);ax.errorbar(i,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',color='black',capsize=4,ms=3)
    ax.set(title='c  All 48 evaluation episodes',ylabel='Axis RMSE (°)',xticks=range(4),xticklabels=['No adapt','Identity','Ramp','Frozen'])
    ax=axs[1,1]
    for segment,style in [('Apex to end','-'),('Near to end','--')]:
        g=sensitivity[(sensitivity.factor=='Axis azimuth')&(sensitivity.segment==segment)]
        ax.plot(g.value,g.axis_rmse_deg,color='#327DA8',ls=style,marker='o',label=segment)
    ax.set(title='d  Dependence on estimated tilt direction',xlabel='Azimuth offset from calibration (°)',ylabel='Mean axis RMSE (°)');ax.legend(fontsize=6,frameon=False)
    fig.suptitle('TCP-calibrated retrospective prediction; simulation law frozen',fontsize=8)
    export(fig,'transfer_summary')
    plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(7.086614,4.7),layout='constrained')
    for j,label in enumerate(['x','y','z']):
        for typ in ['target','nominal']:
            axs[0,j].plot(GRID,archive[f'e{ep}_{typ}_p'][:,j]*1000,color='#C07747' if typ=='target' else '#999999',label='Real demo' if typ=='target' else 'Nominal')
        for method,color in zip(METHODS[1:],COLORS[1:]):axs[0,j].plot(GRID,archive[f'e{ep}_{method}_p'][:,j]*1000,color=color,label=method)
        axs[0,j].set(title=f'{chr(97+j)}  TCP {label}',ylabel='Position (mm)',xlabel='Event progress')
        axs[1,j].plot(GRID,target[:,j],color='#C07747',label='Real demo')
        for method,color in zip(METHODS,COLORS):axs[1,j].plot(GRID,archive[f'e{ep}_{method}_axis'][:,j],color=color,label=method)
        axs[1,j].set(title=f'{chr(100+j)}  Hand-axis {label}',ylabel='Unit-axis component',xlabel='Event progress')
    axs[0,0].legend(fontsize=5.5,frameon=False)
    fig.suptitle(f'Episode {ep}: conditional TCP trajectory, not an executable command',fontsize=8)
    export(fig,'trajectory_components')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.086614,3.7),layout='constrained')
    for j,(ax,dims) in enumerate(zip(axs,[(0,1),(0,2)])):
        for typ,color,label in [('target','#C07747','Real demo'),('nominal','#999999','Nominal')]+[(m,c,m) for m,c in zip(METHODS[1:],COLORS[1:])]:
            points=archive[f'e{ep}_{typ}_p']*1000
            ax.plot(points[:,dims[0]],points[:,dims[1]],color=color,label=label,lw=1.3)
            ax.scatter(points[[0,50,100,150],dims[0]],points[[0,50,100,150],dims[1]],color=color,s=9)
        ax.set(xlabel='TCP x (mm)',ylabel='TCP '+('y' if j==0 else 'z')+' (mm)',title=f'{chr(97+j)}  '+('Top view' if j==0 else 'Side view'))
        ax.set_aspect('equal',adjustable='datalim')
    axs[0].legend(fontsize=6,frameon=False,loc='upper right')
    fig.suptitle(f'Episode {ep}: nominal, predicted and demonstrated TCP paths',fontsize=8)
    export(fig,'spatial_trajectories');plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    lawpath=BASE/'tilt_sweep/frozen_law.npz';before=sha(lawpath)
    rows=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv')
    n=rows[rows.dataset=='insert_jc_fix_3'].set_index('episode_index');t=rows[rows.dataset=='insert_10degree'].set_index('episode_index')
    xy=['pickup_tcp_x_m','pickup_tcp_y_m'];distance=np.linalg.norm(t[xy].values[:,None,:]-n[xy].values[None,:,:],axis=2);nearest=distance.argmin(1)
    matches=pd.DataFrame(dict(target_episode=t.index,nominal_episode=n.index.values[nearest],pickup_distance_mm=distance.min(1)*1000))
    matches['split']=np.where(matches.target_episode%5==0,'Calibration','Evaluation');matches.to_csv(OUT/'matches.csv',index=False)
    cal=matches[matches.split=='Calibration'];test=matches[matches.split=='Evaluation'];assert len(cal)==13 and len(test)==48
    ac=['axis_x','axis_y','axis_z'];pc=['tcp_x_m','tcp_y_m','tcp_z_m']
    a=unit(n.loc[cal.nominal_episode,ac].values.mean(0));b=unit(t.loc[cal.target_episode,ac].values.mean(0));axis=unit(np.cross(a,b))
    rot=Rotation.from_rotvec(np.deg2rad(10)*axis).as_matrix();A=np.eye(3)-rot
    p0=n.loc[cal.nominal_episode,pc].values;p1=t.loc[cal.target_episode,pc].values;rhs=p1-p0@rot.T
    center=p0.mean(0);pivot=np.linalg.lstsq(A,rhs.mean(0),rcond=1e-10)[0];pivot+=axis*np.dot(axis,center-pivot)
    residual=rhs-pivot@A.T
    geometry=dict(method='Calibration TCP minimal-axis + fixed 10 degree + rank-2 effective-pivot least squares',calibration_episodes=cal.target_episode.tolist(),evaluation_episodes=test.target_episode.tolist(),axis_base=axis.tolist(),input_angle_deg=10,observed_calibration_angle_deg=float(angle(a,b)),effective_pivot_base_m=pivot.tolist(),nominal_terminal_center_m=center.tolist(),pivot_fit_rms_mm=float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))*1000),matrix_rank=int(np.linalg.matrix_rank(A)),limitations=['Response-derived geometry, not independent hole calibration','Fixed grasp and fixed physical pivot assumed','Pivot component along rotation axis is unobservable and gauge fixed','All real data were explored previously; split is retrospective','Source law identified on peg poses, target evaluated on hand axes under fixed grasp'])
    (OUT/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    records={(r['condition'],r['episode']):r for r in load_real()}
    all_scores=[];alpha_rows=[];archive={'progress':GRID};sensitivity=[];sample_rows=[]
    for radius in [.02,.03,.04]:
        alphas,ar=source_alpha(radius);alpha_rows+=ar;alpha=alphas.mean(0)
        sampled={key:resample_real(rec,radius) for key,rec in records.items()}
        for _,m in test.iterrows():
            nominal=sampled['upright',int(m.nominal_episode)];target=sampled['tilted',int(m.target_episode)];ep=int(m.target_episode)
            profiles=[np.zeros_like(GRID),np.ones_like(GRID),np.clip(GRID/2,0,1),alpha]
            if radius==.03:
                for typ,obj in [('nominal',nominal),('target',target)]:
                    for k in ['p','axis']:archive[f'e{ep}_{typ}_{k}']=obj[k]
                    archive[f'e{ep}_{typ}_quaternion_xyzw']=Rotation.from_matrix(obj['R']).as_quat()
            for method,profile in zip(METHODS,profiles):
                pred=predict(nominal,axis,pivot,profile)
                if method=='No adaptation':assert np.allclose(pred['p'],nominal['p']) and np.allclose(pred['R'],nominal['R'])
                assert np.allclose(np.linalg.det(pred['R']),1) and np.allclose(np.linalg.norm(pred['axis'],axis=1),1)
                if radius==.03:
                    for k in ['p','axis']:archive[f'e{ep}_{method}_{k}']=pred[k]
                    archive[f'e{ep}_{method}_quaternion_xyzw']=Rotation.from_matrix(pred['R']).as_quat()
                    for i,s in enumerate(GRID):sample_rows.append(dict(episode=ep,method=method,progress=s,axis_error_deg=float(angle(pred['axis'][i],target['axis'][i])),position_error_mm=float(np.linalg.norm(pred['p'][i]-target['p'][i])*1000)))
                for segment,start in [('Whole',0),('Apex to end',1),('Near to end',2)]:
                    ae,pe=score(pred,target,GRID>=start);all_scores.append(dict(episode=ep,nominal_episode=int(m.nominal_episode),radius_mm=radius*1000,method=method,segment=segment,axis_rmse_deg=ae,position_rmse_mm=pe))
            if radius==.03:
                # All sensitivity settings retained; no target-driven selection.
                options=[('Axis azimuth',offset,Rotation.from_rotvec([0,0,np.deg2rad(offset)]).apply(axis),pivot) for offset in [-15,0,15]]
                options += [('Pivot lever arm',mm,axis,center-mm/1000*a) for mm in [0,50,100]]
                options += [('Source seed',seed,axis,pivot) for seed in SEEDS]
                for factor,value,ax,piv in options:
                    profile=alphas[SEEDS.index(value)] if factor=='Source seed' else alpha
                    pred=predict(nominal,ax,piv,profile)
                    for segment,start in [('Apex to end',1),('Near to end',2)]:
                        ae,pe=score(pred,target,GRID>=start);sensitivity.append(dict(episode=ep,factor=factor,value=value,segment=segment,axis_rmse_deg=ae,position_rmse_mm=pe))
    scores=pd.DataFrame(all_scores);scores.to_csv(OUT/'episode_metrics.csv',index=False);main_scores=scores[scores.radius_mm==30]
    summary=main_scores.groupby(['method','segment'])[['axis_rmse_deg','position_rmse_mm']].agg(['mean','median','std']);summary.to_csv(OUT/'summary_metrics.csv')
    sens=pd.DataFrame(sensitivity);sens.to_csv(OUT/'sensitivity_episode_metrics.csv',index=False);sens_summary=sens.groupby(['factor','value','segment'])[['axis_rmse_deg','position_rmse_mm']].mean().reset_index();sens_summary.to_csv(OUT/'sensitivity_summary.csv',index=False)
    pd.DataFrame(alpha_rows).to_csv(OUT/'source_event_alpha.csv',index=False);pd.DataFrame(sample_rows).to_csv(OUT/'pointwise_errors.csv',index=False)
    np.savez_compressed(OUT/'predicted_trajectories.npz',**archive)
    representative=int(test.sort_values(['pickup_distance_mm','target_episode']).iloc[len(test)//2].target_episode)
    (OUT/'representative.json').write_text(json.dumps(dict(episode=representative,selection='Middle ranked evaluation episode by pickup matching distance; not prediction error'),indent=2)+'\n')
    plot_results(main_scores,matches,n,t,archive,representative,sens_summary,alpha)
    inputs=[lawpath,BASE/'tilt_sweep/frozen_law.json',BASE/'fk_results/terminal_pose_per_episode.csv',Path(__file__),BASE/'analyze_event_alignment.py',OUT/'PROTOCOL.md']+[Path(r['source']) for r in records.values()]+[BASE/f'tilt_sweep/circular_seed_{s}.h5' for s in SEEDS]
    (OUT/'provenance.json').write_text(json.dumps({str(p):sha(p) for p in inputs},indent=2)+'\n');assert sha(lawpath)==before
    print(json.dumps(geometry,indent=2));print(summary.round(3).to_string());print('Matching:',matches.groupby('split').pickup_distance_mm.agg(['min','median','max']).to_string());print('PASS: frozen law unchanged; all 48 evaluation episodes scored under all 3 radii.')
if __name__=='__main__':main()
