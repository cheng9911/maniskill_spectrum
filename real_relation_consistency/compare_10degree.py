"""Amplitude-matched exploratory terminal axis comparison, no equivalence claim.

Figure contract: compare the same terminal hand-axis response magnitude at 10°,
show all four simulated directions and all seeds, quantify real-minus-sim gaps.
Two-panel quantitative grid, Python, 180 x 85 mm, PDF/SVG/600-dpi PNG.
"""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_fk import mean_axis, angle

CONDITIONS=['roll_minus','roll_plus','pitch_minus','pitch_plus']


def axes_from_pose(pose, sign=1):
    # SAPIEN serializes quaternions as wxyz.
    return sign*Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()[:,:,2]


def extract(root):
    rows=[]
    for file in sorted(root.glob('circular_seed_*.h5')):
        seed=int(file.stem.split('_')[-1])
        with h5py.File(file) as f:
            for key in sorted(f,key=lambda s:int(s.split('_')[-1])):
                g=f[key];c=np.array(g['causal_delta']);gen=str(g.attrs['generator'])
                cid='baseline' if gen=='baseline' else gen+('_plus' if c[3:5].sum()>0 else '_minus')
                indices=np.flatnonzero(np.array(g['solver_phase'])==6)
                row=dict(seed=seed,episode=key,condition=cid,
                    input_tilt_deg=float(np.degrees(np.linalg.norm(c[3:5]))),
                    success=bool(np.array(g['success'])[-1]),phase6_frames=len(indices),
                    has_phase6=bool(len(indices)),solver_error=str(g.attrs.get('solver_error','')))
                if len(indices):
                    hand=axes_from_pose(np.array(g['tcp_pose'])[indices],-1)
                    peg=axes_from_pose(np.array(g['peg_pose'])[indices])
                    for label,sl in [('all',slice(None)),('last_half',slice(len(indices)//2,None)),('last_frame',slice(-1,None))]:
                        for body,data in [('hand',hand),('peg',peg)]:
                            for component,value in zip('xyz',mean_axis(data[sl])):
                                row[f'{body}_{label}_{component}']=value
                    row['mean_hand_peg_axis_gap_deg']=float(angle(hand,peg).mean())
                rows.append(row)
    data=pd.DataFrame(rows);response=[]
    for seed,g in data.groupby('seed'):
        base=g[g.condition=='baseline']
        if len(base)!=1 or not bool(base.iloc[0].has_phase6):
            continue
        base=base.iloc[0]
        for _,r in g[g.condition!='baseline'].iterrows():
            if not r.has_phase6:continue
            row=dict(seed=seed,condition=r.condition,input_tilt_deg=r.input_tilt_deg,success=r.success)
            for body in ['hand','peg']:
                for window in ['all','last_half','last_frame']:
                    cols=[f'{body}_{window}_{j}' for j in 'xyz']
                    row[f'{body}_{window}_response_deg']=float(angle(r[cols].to_numpy(float),base[cols].to_numpy(float)))
            response.append(row)
    return data,pd.DataFrame(response)


def real_bootstrap(real,n=10000):
    rng=np.random.default_rng(20260911);means=[];boots=[]
    for name in ['insert_jc_fix_3','insert_10degree']:
        g=real[real.dataset==name][['axis_x','axis_y','axis_z']].to_numpy()
        assert np.isfinite(g).all()
        means.append(mean_axis(g))
        a=g[rng.integers(0,len(g),(n,len(g)))].mean(1)
        boots.append(a/np.linalg.norm(a,axis=1,keepdims=True))
    return float(angle(means[0],means[1])),angle(boots[0],boots[1])


def compare(real,response):
    point,boot=real_bootstrap(real);rng=np.random.default_rng(20260912);rows=[]
    for cond in CONDITIONS:
        g=response[response.condition==cond]
        if len(g)==0:continue
        values=g.hand_all_response_deg.to_numpy()
        simboot=values[rng.integers(0,len(values),(len(boot),len(values)))].mean(1)
        diffboot=boot-simboot;ci=np.quantile(diffboot,[.025,.975])
        rows.append(dict(condition=cond,n_sim=len(g),sim_mean_hand_response_deg=float(values.mean()),
            sim_sd_hand_response_deg=float(values.std(ddof=1)) if len(values)>1 else None,
            sim_mean_peg_response_deg=float(g.peg_all_response_deg.mean()),
            sim_bootstrap_ci95_low_deg=float(np.quantile(simboot,.025)),sim_bootstrap_ci95_high_deg=float(np.quantile(simboot,.975)),
            real_mean_axis_response_deg=point,real_minus_sim_deg=point-float(values.mean()),
            difference_bootstrap_ci95_low_deg=float(ci[0]),difference_bootstrap_ci95_high_deg=float(ci[1])))
    table=pd.DataFrame(rows)
    summary=dict(real_axis_response_deg=point,real_episode_bootstrap_ci95_deg=np.quantile(boot,[.025,.975]).tolist(),
        bootstrap_draws=len(boot),simulation_terminal='phase6 hand axis; paired nominal per seed',
        real_terminal='0.5 second pre-release hand axis window with 0.2 second guard',
        interpretation='Amplitude-matched, direction-agnostic terminal hand-axis response evidence',
        statistical_equivalence='NOT_TESTED: no independently justified margin; no calibration/session error budget',
        limitations=['True physical tilt direction unknown; four simulated cardinal directions do not exhaust all directions',
            'Terminal stages are proxies, not time-resolved phase correspondence',
            'Human success labels user-confirmed; no failure-rate experiment',
            'Three simulation seeds; bootstrap does not create new independent trials',
            'Real bootstrap conditional on exchangeability within datasets, not calibrated uncertainty',
            'Simulated planner explicitly uses socket tilt: this validates executed response, not autonomous law discovery or frozen law transfer'])
    return table,summary


def plot(table,response,summary,out):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(7.086614,3.346457),layout='constrained')
    labels=['Roll −10°','Roll +10°','Pitch −10°','Pitch +10°']
    for i,cond in enumerate(CONDITIONS):
        g=response[response.condition==cond];r=table[table.condition==cond].iloc[0]
        # Horizontal offsets expose all observed seeds; no synthetic observations.
        axs[0].scatter(i+np.linspace(-.12,.12,len(g)),g.hand_all_response_deg,s=18,color='#427A9E')
        axs[0].plot([i-.18,i+.18],[r.sim_mean_hand_response_deg]*2,color='#1D4055',lw=1.4)
        m=r.real_minus_sim_deg;lo=r.difference_bootstrap_ci95_low_deg;hi=r.difference_bootstrap_ci95_high_deg
        axs[1].errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color='#555555',ms=4)
    m=summary['real_axis_response_deg'];lo,hi=summary['real_episode_bootstrap_ci95_deg']
    axs[0].errorbar(4,m,yerr=[[m-lo],[hi-m]],fmt='D',capsize=3,color='#C07747',ms=5)
    axs[0].axhline(10,color='#999999',ls='--',lw=.8)
    axs[0].set(xticks=range(5),xticklabels=labels+['Real 10°'],ylabel='Terminal hand-axis response (°)',
               title='a  Matched 10° amplitude')
    axs[1].axhline(0,color='#999999',ls='--',lw=.8)
    axs[1].set(xticks=range(4),xticklabels=labels,ylabel='Real − simulation (°)',title='b  Response discrepancy and uncertainty')
    for ax in axs:ax.tick_params(axis='x',labelsize=6)
    fig.savefig(out/'comparison_10degree.png',dpi=600)
    fig.savefig(out/'comparison_10degree.pdf');fig.savefig(out/'comparison_10degree.svg');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sim-dir',type=Path,default=Path('real_relation_consistency/sim_10degree'))
    ap.add_argument('--real-csv',type=Path,default=Path('real_relation_consistency/fk_results/terminal_pose_per_episode.csv'));args=ap.parse_args()
    data,response=extract(args.sim_dir)
    data.to_csv(args.sim_dir/'episode_axis_audit.csv',index=False);response.to_csv(args.sim_dir/'sim_axis_responses.csv',index=False)
    real=pd.read_csv(args.real_csv);table,summary=compare(real,response)
    summary.update(sim_attempts=len(data),sim_successes=int(data.success.sum()),sim_with_terminal_phase=int(data.has_phase6.sum()))
    table.to_csv(args.sim_dir/'comparison.csv',index=False)
    (args.sim_dir/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(table,response,summary,args.sim_dir)
    files=sorted(args.sim_dir.glob('*.h5'))+[args.real_csv,Path(__file__),args.sim_dir/'contexts.json',args.sim_dir/'protocol.json']
    manifest=[]
    for p in files:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        manifest.append(dict(path=str(p.resolve()),sha256=h.hexdigest()))
    (args.sim_dir/'comparison_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(table.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
