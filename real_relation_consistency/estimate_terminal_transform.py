"""Estimate a full base-frame rigid transform from terminal Franka TCP poses.
No fixed angle or fixed-pivot constraint. A hole-transform interpretation requires
constant hole-to-TCP terminal relation, including insertion depth and axial spin.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;OUT=BASE/'terminal_transform'

def average(poses):
 T=np.eye(4);T[:3,:3]=Rotation.from_matrix(poses[:,:3,:3]).mean().as_matrix();T[:3,3]=poses[:,:3,3].mean(0);return T

def inverse(T):
 V=np.eye(4);V[:3,:3]=T[:3,:3].T;V[:3,3]=-T[:3,:3].T@T[:3,3];return V

def estimate(a,b):
 A=average(a);B=average(b);D=B@inverse(A);return A,B,D

def describe(a,b):
 A,B,D=estimate(a,b);axis0=A[:3,2];axis1=B[:3,2];v=np.cross(axis0,axis1);theta=np.arctan2(np.linalg.norm(v),np.dot(axis0,axis1));M=np.eye(4)
 if np.linalg.norm(v)>1e-10:M[:3,:3]=Rotation.from_rotvec(v/np.linalg.norm(v)*theta).as_matrix()
 M[:3,3]=B[:3,3]-M[:3,:3]@A[:3,3]
 return dict(n_nominal=len(a),n_tilt=len(b),nominal_mean_tcp=A.tolist(),tilted_mean_tcp=B.tolist(),delta_base_left=D.tolist(),rotation_vector_deg=np.rad2deg(Rotation.from_matrix(D[:3,:3]).as_rotvec()).tolist(),full_rotation_angle_deg=float(np.rad2deg(Rotation.from_matrix(D[:3,:3]).magnitude())),translation_base_m=D[:3,3].tolist(),mean_tcp_position_difference_m=(B[:3,3]-A[:3,3]).tolist(),axis_tilt_deg=float(np.rad2deg(theta)),axis_only_minimal_rotation_delta_base=M.tolist(),matrix_convention='X_tilt = delta_base_left @ X_upright; homogeneous column vectors; position in m',averaging='Each episode: mean position + quaternion chordal rotation mean over pre-release window; group: equally weighted episode position and rotation means')

def main():
 OUT.mkdir(exist_ok=True);table=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');poses={};files=[];rows=[]
 for _,r in table.iterrows():
  p=BASE/'fk_results/trajectories'/f'{r.dataset}_{int(r.episode_index):03d}.npz';files.append(p);z=np.load(p);w=slice(int(r.window_start),int(r.window_end_exclusive));T=np.eye(4);T[:3,:3]=Rotation.from_quat(z['tcp_quaternion_xyzw'][w]).mean().as_matrix();T[:3,3]=z['tcp_position_m'][w].mean(0);poses[r.dataset,int(r.episode_index)]=T
 matches=pd.read_csv(BASE/'real_predictive_transfer/matches.csv')
 a=np.asarray([poses['insert_jc_fix_3',int(m.nominal_episode)] for _,m in matches.iterrows()]);b=np.asarray([poses['insert_10degree',int(m.target_episode)] for _,m in matches.iterrows()]);cal=matches.split.values=='Calibration';ev=~cal
 result={'primary_calibration13':describe(a[cal],b[cal]),'all_data_descriptive':describe(np.asarray([v for (d,e),v in poses.items() if d=='insert_jc_fix_3']),np.asarray([v for (d,e),v in poses.items() if d=='insert_10degree'])),'single_nominal_episode0_to_calibration_mean':describe(np.asarray([poses['insert_jc_fix_3',0]]),b[cal])}
 D=np.asarray(result['primary_calibration13']['delta_base_left']);A=np.asarray(result['primary_calibration13']['nominal_mean_tcp']);B=np.asarray(result['primary_calibration13']['tilted_mean_tcp']);assert np.allclose(D@A,B,atol=1e-12);assert np.allclose(D[:3,:3].T@D[:3,:3],np.eye(3)) and np.isclose(np.linalg.det(D[:3,:3]),1)
 rng=np.random.default_rng(20260911);boot=[]
 for _ in range(2000):
  idx=rng.integers(0,int(cal.sum()),int(cal.sum()));_,_,db=estimate(a[cal][idx],b[cal][idx]);boot.append(np.r_[np.rad2deg(Rotation.from_matrix(db[:3,:3]).as_rotvec()),db[:3,3],np.rad2deg(Rotation.from_matrix(db[:3,:3]).magnitude())])
 names=['rotation_vector_x_deg','rotation_vector_y_deg','rotation_vector_z_deg','translation_x_m','translation_y_m','translation_z_m','full_rotation_angle_deg'];lo,hi=np.quantile(boot,[.025,.975],axis=0);result['calibration_bootstrap_pointwise_95pct']={k:[float(l),float(h)] for k,l,h in zip(names,lo,hi)}
 for i,(_,m) in enumerate(matches.iterrows()):
  di=b[i]@inverse(a[i]);pred=D@a[i];rot_err=Rotation.from_matrix(pred[:3,:3].T@b[i,:3,:3]).magnitude();u=pred[:3,2];v=b[i,:3,2]
  rows.append(dict(target_episode=int(m.target_episode),nominal_episode=int(m.nominal_episode),split=m.split,position_error_mm=float(np.linalg.norm(pred[:3,3]-b[i,:3,3])*1000),orientation_error_deg=float(np.rad2deg(rot_err)),axis_error_deg=float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u,v)),np.dot(u,v)))),**{f'delta_{j}{k}':float(di[j,k]) for j in range(4) for k in range(4)}))
 errors=pd.DataFrame(rows);errors.to_csv(OUT/'per_pair_transforms_and_errors.csv',index=False);result['evaluation48_terminal_error']=errors[errors.split=='Evaluation'][['position_error_mm','orientation_error_deg','axis_error_deg']].agg(['mean','median','max']).to_dict();result['limitations']=['Estimated TCP terminal pose transform, not independently calibrated hole pose','Hole interpretation requires constant hole-to-TCP terminal transform: grasp, depth, axial spin and compliance','Circular hole does not determine axial spin: full TCP rotation may include human-selected yaw','Transform translation t = p1 - R p0 is not simply p1 - p0','Calibration 13, evaluation 48; old retrospective split retained','Bootstrap resamples calibration pairs; repeated nominal references and unknown sessions not independent','No old trajectory artifacts overwritten or executed']
 (OUT/'transform.json').write_text(json.dumps(result,indent=2)+'\n');np.savetxt(OUT/'delta_base_left_calibration13.txt',D,fmt='%.10f');np.save(OUT/'delta_base_left_calibration13.npy',D)
 ref=np.asarray(result['single_nominal_episode0_to_calibration_mean']['delta_base_left']);np.savetxt(OUT/'delta_for_nominal_episode0.txt',ref,fmt='%.10f')
 files += [Path(__file__),BASE/'fk_results/terminal_pose_per_episode.csv',BASE/'real_predictive_transfer/matches.csv'];(OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
