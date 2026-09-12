from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
m=json.loads((OUT/'manifest.json').read_text());cols=['x_m','y_m','z_m','qx','qy','qz','qw'];checks={}
a0=pd.read_csv(OUT/'assembly_vertical.csv');a1=pd.read_csv(OUT/'assembly_tilt10.csv');response=pd.read_csv(OUT/'assembly_response.csv')
assert len(a0)==len(a1)==451 and np.allclose(a0.iloc[0][cols].astype(float),a1.iloc[0][cols].astype(float))
for condition in m['conditions']:
 name=condition['pickup_id'];pick=pd.read_csv(OUT/f'{name}_pickup.csv')
 assert len(pick)==451 and np.allclose(pick.iloc[-1][cols].astype(float),a0.iloc[0][cols].astype(float))
 assert np.allclose(pick.loc[pick.stage=='close_gripper_event',['x_m','y_m','z_m']],condition['grasp_tcp_base_m'])
 for label,assembly in [('vertical',a0),('tilt10',a1)]:
  data=pd.read_csv(OUT/f'{name}_{label}.csv');assert len(data)==901 and np.all(np.diff(data.time_s)>0)
  assert np.allclose(data[cols].iloc[450:].values,assembly[cols].values)
  assert (data.stage=='close_gripper_event').sum()==30
  q=data[['qx','qy','qz','qw']].values;assert np.allclose(np.linalg.norm(q,axis=1),1)
  assert (np.sum(q[1:]*q[:-1],axis=1)>=0).all()
  assert np.isfinite(data[cols].values).all()
  checks[f'{name}_{label}']='PASS: 901 samples, exact common assembly, monotone time, valid quaternion, gripper event'
R0=Rotation.from_quat(a0[['qx','qy','qz','qw']]).as_matrix();R1=Rotation.from_quat(a1[['qx','qy','qz','qw']]).as_matrix();axis=np.asarray(m['geometry']['axis_base']);pivot=np.asarray(m['geometry']['effective_pivot_base_m'])
D=Rotation.from_rotvec(response.alpha_pitch.values[:,None]*np.deg2rad(10)*axis).as_matrix()
assert np.allclose(R1,D@R0)
p0=a0[['x_m','y_m','z_m']].values;p1=a1[['x_m','y_m','z_m']].values
assert np.allclose(p1,pivot+np.einsum('nij,nj->ni',D,p0-pivot))
assert np.allclose(np.rad2deg(Rotation.from_matrix(R1@R0.transpose(0,2,1)).magnitude()),response.applied_rotation_deg)
for p,h in json.loads((OUT/'provenance.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
checks.update(finite_SE3='PASS',common_start='PASS: alpha(0)=0 without forcing',provenance='PASS',execution_validation='NOT PERFORMED: IK, collisions, controller dynamics and physical grasp')
(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
