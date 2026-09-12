"""Fit and freeze all six diagonal finite SE3 channels on existing circular data."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import h5py
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase_switch_symmetry'))
from benchmark_phase_switch_baselines import usable,progress_grid
from benchmark_se3_transfer import task_curve_se3,nominal_frame_se3
from phase_switch_se3_baselines import SE3SmoothFinitePDiagModel,SE3FrameWeightedModel,se3_from_pose6_batched
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;OUT=BASE/'full_se3_law';SEEDS=[20260818,20270818,20280818];NAMES=['du','dv','dw','roll','pitch','yaw'];PHASE=['Align','Enter','Unlock label','Insert']
CONFIG=dict(alpha_max=1.25,n_basis=24,basis_width=.065,smoothness_weight=.1,nominal_iterations=3)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(pred,actual):
 P=se3_from_pose6_batched(pred.reshape(-1,6)).reshape(*pred.shape[:-1],4,4);A=se3_from_pose6_batched(actual.reshape(-1,6)).reshape(*actual.shape[:-1],4,4)
 dist=np.linalg.norm(P[...,:3,3]-A[...,:3,3],axis=-1)*1000;rel=np.swapaxes(P[...,:3,:3],-1,-2)@A[...,:3,:3];rot=np.rad2deg(Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude()).reshape(dist.shape);u=P[...,:3,2];v=A[...,:3,2];axis=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1)))
 return [np.sqrt(np.mean(x*x,axis=1)) for x in [dist,rot,axis]]
def main():
 OUT.mkdir(exist_ok=True);saved={};audit=[];rows=[];summary={};files=[];progress,phase=progress_grid(25)
 for seed in SEEDS:
  path=ROOT/f'phase_switch_symmetry_rollouts_se3/circular_honest_seed_{seed}.h5';files.append(path)
  with h5py.File(path) as f:
   selected={}
   for key in sorted(f,key=lambda k:int(k.split('_')[-1])):
    g=f[key];cid=int(g.attrs['condition_id']);ok=usable(g);chosen=ok and cid not in selected
    audit.append(dict(seed=seed,episode_key=key,condition_id=cid,generator=str(g.attrs['generator']),attempt_id=int(g.attrs.get('attempt_id',0)),usable=ok,selected=chosen,success=bool(np.asarray(g['success'])[-1]),solver_error=str(g.attrs.get('solver_error',''))))
    if chosen:selected[cid]=key
   mixed=sorted(c for c,k in selected.items() if str(f[k].attrs['generator'])=='mixed');assert len(mixed)==60
   test_mixed=set(mixed[::4]);train=sorted(c for c in selected if c==0 or (c in mixed and c not in test_mixed));test=sorted(set(selected)-set(train));assert len(train)==46 and len(test)==29
   keys=[selected[c] for c in train];ctx=np.array([f[k]['causal_delta'] for k in keys]);curves=np.array([task_curve_se3(f[k],25) for k in keys]);nominal=nominal_frame_se3(f,keys,ctx)
   testctx=np.array([f[selected[c]]['causal_delta'] for c in test]);testcurves=np.array([task_curve_se3(f[selected[c]],25) for c in test]);testgen=[str(f[selected[c]].attrs['generator']) for c in test]
  model=SE3SmoothFinitePDiagModel(nominal_frame_pose=nominal,**CONFIG);scale=np.array([.012]*3+[np.deg2rad(15)]*2+[np.deg2rad(30)])
  design=ctx/scale;twists=model._twists(ctx)/scale
  assert np.linalg.matrix_rank(design)==np.linalg.matrix_rank(twists)==6
  print(f'Fitting seed {seed}: 46 train / 29 test, rank6, scaled condition {np.linalg.cond(design):.3f}',flush=True)
  model.fit(ctx,curves,progress,phase);assert model.optimization_success
  scalar=SE3FrameWeightedModel().fit(ctx,curves,progress,phase)
  methods={'Frozen diagonal':model.predict(testctx),'No adaptation':model._predict_with(testctx,model.nominal_curve,np.zeros_like(model.diagonal)),'Identity':model._predict_with(testctx,model.nominal_curve,np.ones_like(model.diagonal)),'Frame scalar affine':scalar.predict(testctx)}
  for method,pred in methods.items():
   for label,mask in [('All assembly',np.ones(100,bool)),('Align',phase==0),('Insert',phase==3)]:
    scores=metrics(pred[:,mask],testcurves[:,mask])
    for i,c in enumerate(test):rows.append(dict(seed=seed,condition_id=c,generator=testgen[i],method=method,segment=label,position_rmse_mm=scores[0][i],orientation_rmse_deg=scores[1][i],axis_rmse_deg=scores[2][i]))
  for key,val in dict(diagonal=model.diagonal,parameters=model.parameters,nominal_curve=model.nominal_curve,nominal_frame_pose=nominal,train_contexts=ctx,train_curves=curves,test_contexts=testctx,test_curves=testcurves,train_condition_ids=np.array(train),test_condition_ids=np.array(test),test_predictions=methods['Frozen diagonal']).items():saved[f'seed_{seed}_{key}']=val
  summary[str(seed)]=dict(n_train=46,n_test=29,train_condition_ids=train,test_condition_ids=test,scaled_context_rank=6,scaled_context_condition=float(np.linalg.cond(design)),scaled_twist_rank=6,scaled_twist_condition=float(np.linalg.cond(twists)),optimization_success=bool(model.optimization_success),optimization_cost=float(model.optimization_cost),optimization_nfev=int(model.optimization_nfev),phase_mean=model.diagonal.reshape(4,25,6).mean(1).tolist(),terminal=model.diagonal[-1].tolist())
  # Checkpoint each completed fit; all-seed final marker only below.
  np.savez_compressed(OUT/f'checkpoint_{seed}.npz',**{k:v for k,v in saved.items() if k.startswith(f'seed_{seed}_')})
  print(f'Finished {seed}; phase means: {np.round(np.array(summary[str(seed)]["phase_mean"]),3).tolist()}',flush=True)
 np.savez_compressed(OUT/'frozen_full_se3.npz',progress=progress,phase_codes=phase,generator_names=np.array(NAMES),**saved)
 pd.DataFrame(audit).to_csv(OUT/'attempt_audit.csv',index=False);scores=pd.DataFrame(rows);scores.to_csv(OUT/'heldout_metrics.csv',index=False)
 summary.update(model='SE3SmoothFinitePDiagModel',config=CONFIG,generator_names=NAMES,finite_action='C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0',scope='Circular task; all six diagonal coefficients; no real fit; no dense operator identification',source_files={str(p):sha(p) for p in files},source_code={str(p):sha(p) for p in [Path(__file__),ROOT/'phase_switch_symmetry/phase_switch_se3_baselines.py',ROOT/'phase_switch_symmetry/benchmark_se3_transfer.py',ROOT/'phase_switch_symmetry/benchmark_phase_switch_baselines.py',OUT/'PROTOCOL.md']})
 (OUT/'frozen_full_se3.json').write_text(json.dumps(summary,indent=2)+'\n')
 alpha=np.array([saved[f'seed_{s}_diagonal'] for s in SEEDS]);csv=[]
 for k,name in enumerate(NAMES):
  for i in range(100):csv.append(dict(sample_index=i,source_progress=progress[i],phase=PHASE[phase[i]],phase_index=int(phase[i]),phase_progress=(i%25)/24,generator=name,mean=alpha[:,i,k].mean(),sd=alpha[:,i,k].std(ddof=1),**{f'seed_{s}':alpha[j,i,k] for j,s in enumerate(SEEDS)}))
 pd.DataFrame(csv).to_csv(OUT/'pr_all_generators.csv',index=False)
 phase_summary=pd.DataFrame([dict(phase=PHASE[p],generator=name,mean=alpha[:,p*25:(p+1)*25,k].mean(),seed_phase_mean_sd=alpha[:,p*25:(p+1)*25,k].mean(1).std(ddof=1)) for p in range(4) for k,name in enumerate(NAMES)]);phase_summary.to_csv(OUT/'phase_summary.csv',index=False)
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(3,2,figsize=(7.086614,5.8),layout='constrained');x=np.arange(100)
 for k,ax in enumerate(axs.flat):
  mean=alpha[:,:,k].mean(0);sd=alpha[:,:,k].std(0,ddof=1)
  ax.fill_between(x,mean-sd,mean+sd,color='#327DA8',alpha=.2);ax.plot(x,mean,color='#327DA8',lw=1.5)
  for line in alpha[:,:,k]:ax.plot(x,line,color='#327DA8',lw=.45,alpha=.4)
  ax.axhline(1,color='#888888',ls=':',lw=.7);ax.axhline(0,color='#aaaaaa',lw=.5)
  for b in [24.5,49.5,74.5]:ax.axvline(b,color='#dddddd',lw=.6)
  ax.set(title=f'{chr(97+k)}  {NAMES[k]}',ylabel='Response coefficient',xticks=[12,37,62,87],xticklabels=['Align','Enter','Unlock*','Insert'],ylim=(-.15,1.4))
 fig.suptitle('Circular-task six-generator law: three simulation fits, mean ± SD',fontsize=8)
 fig.savefig(OUT/'pr_all_generators.png',dpi=600);fig.savefig(OUT/'pr_all_generators.pdf');fig.savefig(OUT/'pr_all_generators.svg');fig.savefig(OUT/'pr_all_generators.tiff',dpi=600);plt.close(fig)
 print(scores[(scores.segment=='All assembly')].groupby('method')[['position_rmse_mm','orientation_rmse_deg','axis_rmse_deg']].mean().round(3).to_string(),flush=True)
if __name__=='__main__':main()
