"""Verify actual fitted artifacts plus software contracts; no manual labels invented."""
from pathlib import Path
import sys,json,hashlib,tempfile,subprocess
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase_switch_symmetry'))
from phase_switch_se3_baselines import SE3SmoothFinitePDiagModel,se3_from_pose6_batched,se3_from_pose6,se3_inverse
from full_se3_adapter import apply_relation
from real_phase_tracker import CausalAssemblyPhaseTracker
BASE=Path(__file__).resolve().parent;S=BASE/'full_se3_law';P=BASE/'real_phase_identification'
def main():
 law=np.load(S/'frozen_full_se3.npz');meta=json.loads((S/'frozen_full_se3.json').read_text());scores=pd.read_csv(S/'heldout_metrics.csv');assert len(scores)==1044 and np.isfinite(scores[['position_rmse_mm','orientation_rmse_deg','axis_rmse_deg']]).all().all();maxdiff=0.
 for seed in [20260818,20270818,20280818]:
  prefix=f'seed_{seed}_';train=set(law[prefix+'train_condition_ids']);test=set(law[prefix+'test_condition_ids']);assert len(train)==46 and len(test)==29 and not train&test
  model=SE3SmoothFinitePDiagModel(nominal_frame_pose=law[prefix+'nominal_frame_pose'],**meta['config']);model.nominal_curve=law[prefix+'nominal_curve'];model.diagonal=law[prefix+'diagonal'];pred=model.predict(law[prefix+'test_contexts']);diff=np.max(np.abs(pred-law[prefix+'test_predictions']));maxdiff=max(maxdiff,float(diff));assert diff<1e-10
  X=se3_from_pose6_batched(model.nominal_curve);C0=se3_from_pose6(model.nominal_frame_pose);context=law[prefix+'test_contexts'][0];D=C0@se3_from_pose6(context)@se3_inverse(C0);result,xi=apply_relation(X,D,C0,model.diagonal);expected=se3_from_pose6_batched(pred[0]);assert np.allclose(result,expected,atol=1e-10)
  zero,_=apply_relation(X,D,C0,np.zeros_like(model.diagonal));one,_=apply_relation(X,D,C0,np.ones_like(model.diagonal));assert np.allclose(zero,X) and np.allclose(one,D@X)
 for group in ['source_files','source_code']:
  for path,h in meta[group].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 for path,h in json.loads((P/'provenance.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 events=pd.read_csv(P/'events.csv');assert len(events)==111*6
 totals=0;departure_errors=0
 for (name,ep),es in events.groupby(['dataset','episode']):
  f=pd.read_csv(P/'frame_labels'/f'{name}_{ep:03d}.csv');totals+=len(f);assert f.contact_state.eq('unknown_no_ground_truth').all();assert np.all(np.diff(f.time_s)>0)
  order=es.set_index('event').frame.reindex(['grasp','lift_clearance','assembly_region','axial_advance_candidate','terminal_settle_candidate','release']).dropna().values;assert np.all(np.diff(order)>=0)
  settled=f.stage=='terminal_settle_candidate';departure=(f.reference_distance_m>.025)|(f.reference_axis_error_deg>7)|(f.speed_m_s>.015)|(f.angular_speed_deg_s>8);departure_errors+=int((settled&departure).sum())
 assert totals==94752 and departure_errors==0
 refs=json.loads((P/'reference_geometry.json').read_text());assert all(all(ep%5==0 for ep in r['calibration_episodes']) for r in refs.values())
 z=np.load(BASE/'fk_results/trajectories/insert_jc_fix_3_000.npz');ref=refs['insert_jc_fix_3'];hist=[]
 for limit in [400,len(z['timestamp'])]:
  tr=CausalAssemblyPhaseTracker(ref['position_m'],ref['upward_hand_axis']);result=[]
  for i in range(limit):result.append(tr.update(z['timestamp'][i],z['tcp_position_m'][i],z['tcp_quaternion_xyzw'][i],z['gripper_state'][i]));assert result[-1]['contact']==result[-1]['insertion_success']=='unknown'
  hist.append(result)
 assert hist[0]==hist[1][:400]
 # Evaluator regression fixtures are synthetic software tests in a temp directory.
 with tempfile.TemporaryDirectory() as tmp:
  temp=Path(tmp);ann=temp/'fixture.json';out=temp/'eval';e=events[(events.dataset=='insert_jc_fix_3')&(events.episode==0)&(events.event=='grasp')].iloc[0]
  def sample(exposure):return dict(dataset='insert_jc_fix_3',episode=0,rater='SOFTWARE_TEST_ONLY',clip_start_frame=0,source_fps=30,predictions_shown=exposure,events={'grasp':dict(status='observed',lower_clip_s=float(e.time_s),upper_clip_s=float(e.time_s))})
  ann.write_text(json.dumps({'annotations':[sample(False),sample(True),sample(None)]}));cmd=[sys.executable,str(BASE/'evaluate_phase_annotations.py'),str(ann),'--out',str(out)]
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL);status=json.loads((out/'status.json').read_text());assert status['prediction_blind_observed_annotations']==1 and status['prediction_exposed_observed_annotations']==0
  subprocess.run(cmd+['--include_prediction_exposed'],check=True,stdout=subprocess.DEVNULL);status=json.loads((out/'status.json').read_text());assert status['prediction_blind_observed_annotations']==status['prediction_exposed_observed_annotations']==status['unknown_exposure_observed_annotations']==1
  ann.write_text(json.dumps({'annotations':[]}));subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL);assert not (out/'summary.csv').exists() and not (out/'per_annotation.csv').exists()
 result=dict(status='PASS',frozen_prediction_reconstruction_max_error=maxdiff,metric_rows=1044,real_episodes=111,real_frames=totals,settled_departure_label_errors=departure_errors,checks=['Train/test isolation','Reconstructed frozen model predictions','Six-channel adapter matches model','Zero/identity action','Source hashes','Phase ordering and explicit unknown contact','Calibration split','Causal prefix invariance','Annotation exposure accounting and stale-score removal'],phase_accuracy='NOT MEASURED: independent video event annotations absent')
 (S/'verification.json').write_text(json.dumps(result,indent=2)+'\n');(P/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
