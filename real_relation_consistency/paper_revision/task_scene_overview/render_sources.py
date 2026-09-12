from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'phase_switch_symmetry'))
import sqlite3
import numpy as np
import sapien
import gymnasium as gym
from mani_skill.utils import sapien_utils
from make_videos import load_episode,make_env
from make_scene_frames import render_frame
from make_symmetry_transfer_videos import first_success_condition
import imageio.v2 as imageio
OUT=Path(__file__).resolve().parent
manifest=[]
for name,file in [('keyed','rotated_Q1_seed_20260818.h5'),('circular','circular_honest_seed_20260818.h5')]:
 path=ROOT/'phase_switch_symmetry_rollouts_rotated'/file
 eid=first_success_condition(path,4)
 data,ea,ra=load_episode(path,eid)
 env=gym.make(str(ra['env_id']),num_envs=1,obs_mode='state_dict',control_mode='pd_joint_pos',reward_mode='sparse',sim_backend='physx_cpu',render_mode='rgb_array',orientation=np.asarray(ra['orientation']),task_anchor=np.asarray(ra['task_anchor']),human_render_camera_configs={'render_camera':{'width':1024,'height':1024}})
 env.reset(seed=int(ea.get('episode_seed',20260818)),options={'causal_delta':data['causal_delta'].tolist()})
 center=data['socket_pose'][0,:3]
 target=center+np.array([0,0,.075])
 eye=target+np.array([.26,-.34,.23])
 cam=env.unwrapped.scene.human_render_cameras['render_camera']
 pose=sapien_utils.look_at(eye,target)
 cam.camera.set_local_pose(sapien.Pose(np.asarray(pose.p).reshape(-1),np.asarray(pose.q).reshape(-1)))
 for phase in ([3,4,5,6] if name=='keyed' else [3]):
  hit=np.flatnonzero(data['solver_phase']==phase)
  # During enter, choose a frame before the key has fully cleared the gate.
  if phase==4:
   valid=hit[(data['peg_pose'][hit,2]-center[2]-.029)>.053]
   i=int(valid[len(valid)//2]) if len(valid) else int(hit[len(hit)//2])
  else:i=int(hit[-1])
  frame=render_frame(env,data,i);dest=OUT/'frames'/f'{name}_{phase}.png';imageio.imwrite(dest,frame)
  from transforms3d.quaternions import quat2mat
  R=quat2mat(data['peg_pose'][i,3:]); corners=np.array([[xx,yy,-.029] for xx in [-.021,.021] for yy in [-.014,.014]])
  topkey=float((corners@R.T+data['peg_pose'][i,:3]-center)[:,2].max())
  manifest.append(dict(image=str(dest.relative_to(OUT)),source=str(path),episode=eid,frame=i,solver_phase=phase,peg_center_local_z=float(data['peg_pose'][i,2]-center[2]),key_top_local_z=topkey if name=='keyed' else None,gate_bottom_z=.053,clearance_margin=.053-topkey if name=='keyed' else None,camera_eye=eye.tolist(),camera_target=target.tolist()))
 env.close()
(OUT/'render_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
