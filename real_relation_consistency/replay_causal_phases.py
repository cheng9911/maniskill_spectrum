"""Replay causal prototype on real observations; coverage is not accuracy."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from real_phase_tracker import CausalAssemblyPhaseTracker
BASE=Path(__file__).resolve().parent;OUT=BASE/'real_phase_identification'
def main():
 refs=json.loads((OUT/'reference_geometry.json').read_text());windows=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');rows=[]
 for _,r in windows.iterrows():
  z=np.load(BASE/'fk_results/trajectories'/f'{r.dataset}_{int(r.episode_index):03d}.npz');ref=refs[r.dataset];tracker=CausalAssemblyPhaseTracker(ref['position_m'],ref['upward_hand_axis']);previous=None
  for i,(t,p,q,g) in enumerate(zip(z['timestamp'],z['tcp_position_m'],z['tcp_quaternion_xyzw'],z['gripper_state'])):
   result=tracker.update(t,p,q,g);assert result['contact']=='unknown' and result['insertion_success']=='unknown'
   if result['stage']!=previous:rows.append(dict(dataset=r.dataset,episode=int(r.episode_index),frame=i,time_s=float(t),stage=result['stage']));previous=result['stage']
 pd.DataFrame(rows).to_csv(OUT/'causal_transitions.csv',index=False)
 print(pd.DataFrame(rows).drop_duplicates(['dataset','episode','stage']).groupby('stage').size().to_string())
if __name__=='__main__':main()
