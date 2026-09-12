from pathlib import Path
import sys,json,sqlite3
import numpy as np
import h5py,torch,sapien,gymnasium as gym
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
import imageio.v2 as imageio
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'phase_switch_symmetry'))
import phase_switch_symmetry_planar_push_env
p=ROOT/'phase_switch_symmetry_rollouts_planar_push/heading_push_seed_20260818.h5'
with h5py.File(p) as f:
 g=f['episode_11'];ph=np.array(g['solver_phase']);hits=np.flatnonzero(ph==5);i=int(hits[len(hits)//2]);d={k:np.array(g[k][i]) for k in ['qpos','block_pose','target_pose']};delta=np.array(g['causal_delta']);seed=int(g.attrs['episode_seed']);assert bool(g['success'][-1])
env=gym.make('PlanarPush-v1',num_envs=1,obs_mode='state_dict',control_mode='pd_joint_pos',sim_backend='physx_cpu',render_mode='rgb_array',goal_heading=True,human_render_camera_configs={'render_camera':{'width':1024,'height':1024}})
env.reset(seed=seed,options={'causal_delta':delta});base=env.unwrapped
base.agent.robot.set_qpos(torch.tensor(d['qpos'],dtype=torch.float32,device=base.device).reshape(1,-1))
for obj,key in [(base.block,'block_pose'),(base.target,'target_pose')]:obj.set_pose(Pose.create_from_pq(torch.tensor(d[key][:3],dtype=torch.float32,device=base.device).reshape(1,3),torch.tensor(d[key][3:],dtype=torch.float32,device=base.device).reshape(1,4)))
target=d['block_pose'][:3]+[0,0,.04];eye=target+[.24,-.32,.23];cam=base.scene.human_render_cameras['render_camera'];pose=sapien_utils.look_at(eye,target);cam.camera.set_local_pose(sapien.Pose(np.asarray(pose.p).reshape(-1),np.asarray(pose.q).reshape(-1)))
frame=env.render();frame=frame.cpu().numpy() if hasattr(frame,'cpu') else np.asarray(frame);imageio.imwrite(OUT/'frames/planar_heading.png',np.squeeze(frame));env.close()
(OUT/'planar_manifest.json').write_text(json.dumps(dict(source=str(p),episode='episode_11',frame=i,solver_phase=5,goal_heading=True,causal_delta=delta.tolist(),evidence='closed-loop simulation replay',execution='guided grasp-and-slide, not non-prehensile point pushing'),indent=2))
print('rendered planar',i)
