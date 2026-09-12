"""Apply frozen six-channel alpha to a base-frame relation transform.
The caller MUST supply the target nominal task frame and a phase-aligned alpha.
No implicit adoption of robot-base axes or inference of real contact phases.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase_switch_symmetry'))
from phase_switch_se3_baselines import se3_log,se3_exp_batched,se3_inverse

def apply_relation(nominal_tcp_matrices,delta_base_left,nominal_task_frame,alpha):
 X=np.asarray(nominal_tcp_matrices,float);D=np.asarray(delta_base_left,float);C0=np.asarray(nominal_task_frame,float);a=np.asarray(alpha,float)
 if X.ndim!=3 or X.shape[1:]!=(4,4) or a.shape!=(len(X),6):raise ValueError('Expected X[N,4,4] and alpha[N,6]')
 for T in [D,C0]:
  if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3)) or not np.isclose(np.linalg.det(T[:3,:3]),1):raise ValueError('Invalid rigid transform')
 if not np.isfinite(X).all() or not np.isfinite(a).all():raise ValueError('Nonfinite input')
 C0inv=se3_inverse(C0);relative=C0inv@D@C0;xi=se3_log(relative)
 actions=se3_exp_batched(a*xi[None,:]);prediction=C0@actions@C0inv@X
 return prediction,xi
