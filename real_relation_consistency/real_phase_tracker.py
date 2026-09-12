"""Causal, history-only deployment prototype. States are kinematic candidates.
Requires externally calibrated target TCP pose. No contact/success inference.
"""
import numpy as np
from scipy.spatial.transform import Rotation
class CausalAssemblyPhaseTracker:
 def __init__(self,target_position_m,target_up_axis):
  self.target=np.asarray(target_position_m,float);self.axis=np.asarray(target_up_axis,float);self.axis/=np.linalg.norm(self.axis);self.state='approach';self.last=None;self.timer={};self.grasp_z=None
 def dwell(self,name,condition,time,duration):
  if not condition:self.timer.pop(name,None);return False
  if name not in self.timer:self.timer[name]=time
  return time-self.timer[name]>=duration
 def update(self,time_s,position_m,quaternion_xyzw,gripper_state):
  p=np.asarray(position_m,float);rot=Rotation.from_quat(quaternion_xyzw).as_matrix();up=-rot[:,2];t=float(time_s)
  assert np.isfinite(p).all() and np.isfinite(t)
  if self.last is None:velocity=np.zeros(3);angular=0.
  else:
   dt=t-self.last[0]
   if dt<=0:raise ValueError('Timestamps must increase')
   velocity=(p-self.last[1])/dt;angular=np.rad2deg(Rotation.from_matrix(self.last[2].T@rot).magnitude())/dt
  self.last=(t,p.copy(),rot.copy());speed=np.linalg.norm(velocity);distance=np.linalg.norm(p-self.target);axial=velocity@self.axis;lateral=np.linalg.norm(velocity-axial*self.axis);angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(up,self.axis)),up@self.axis))
  closed=gripper_state<.5
  if self.state=='approach' and self.dwell('closed',closed,t,.15):self.state='grasp_lift';self.grasp_z=float(p[2])
  if self.state not in ['approach','released'] and self.dwell('open',not closed,t,.15):self.state='released'
  if self.state=='grasp_lift' and self.dwell('lift',p[2]-self.grasp_z>=.02,t,.15):self.state='transport'
  if self.state=='transport' and self.dwell('region',distance<=.08,t,.2):self.state='prealignment'
  if self.state=='prealignment' and self.dwell('advance',distance<=.06 and angle<=5 and axial<=-.002 and lateral<=.02,t,.15):self.state='axial_advance_candidate'
  if self.state=='axial_advance_candidate' and self.dwell('settle',distance<=.02 and angle<=5 and speed<=.01 and angular<=5,t,.3):self.state='terminal_settle_candidate'
  if self.state=='terminal_settle_candidate' and (distance>.025 or angle>7 or speed>.015 or angular>8):self.state='axial_advance_candidate';self.timer.pop('settle',None)
  return dict(stage=self.state,contact='unknown',insertion_success='unknown',distance_m=float(distance),axis_error_deg=float(angle),speed_m_s=float(speed))
