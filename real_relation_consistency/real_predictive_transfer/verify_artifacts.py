"""Independent numeric checks of saved prediction artifacts and metric accounting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from predict_real_transfer import predict
m=pd.read_csv(OUT/'matches.csv');s=pd.read_csv(OUT/'episode_metrics.csv');z=np.load(OUT/'predicted_trajectories.npz');g=json.loads((OUT/'geometry.json').read_text())
cal=set(m[m.split=='Calibration'].target_episode);test=set(m[m.split=='Evaluation'].target_episode)
assert len(cal)==13 and len(test)==48 and not cal&test and cal|test==set(range(61))
assert len(s)==48*4*3*3 and not s[['axis_rmse_deg','position_rmse_mm']].isna().any().any()
assert set(s.episode)==test and s.groupby(['radius_mm','method','segment']).size().eq(48).all()
progress=z['progress'];assert np.all(np.diff(progress)>0)
worst=0.
for ep in sorted(test):
    for method in ['No adaptation','Identity','Event ramp','Frozen law']:
        key=f'e{ep}_{method}';p=z[key+'_p'];axes=z[key+'_axis'];q=z[key+'_quaternion_xyzw']
        assert np.isfinite(p).all() and np.isfinite(q).all()
        assert np.allclose(np.linalg.norm(q,axis=1),1) and np.allclose(np.linalg.norm(axes,axis=1),1)
        assert np.allclose(axes,-Rotation.from_quat(q).as_matrix()[:,:,2])
        target=z[f'e{ep}_target_axis'];err=np.degrees(np.arctan2(np.linalg.norm(np.cross(axes,target),axis=1),np.sum(axes*target,axis=1)))
        pos=np.linalg.norm(p-z[f'e{ep}_target_p'],axis=1)*1000
        for segment,start in [('Whole',0),('Apex to end',1),('Near to end',2)]:
            row=s[(s.episode==ep)&(s.method==method)&(s.radius_mm==30)&(s.segment==segment)].iloc[0]
            diff=abs(np.sqrt(np.mean(err[progress>=start]**2))-row.axis_rmse_deg);worst=max(worst,diff)
            assert diff<1e-8
            assert abs(np.sqrt(np.mean(pos[progress>=start]**2))-row.position_rmse_mm)<1e-8
        if method=='No adaptation':
            assert np.allclose(p,z[f'e{ep}_nominal_p']) and np.allclose(axes,z[f'e{ep}_nominal_axis'])
axis=np.asarray(g['axis_base']);pivot=np.asarray(g['effective_pivot_base_m']);ep=min(test)
nom=dict(p=z[f'e{ep}_nominal_p'],R=Rotation.from_quat(z[f'e{ep}_nominal_quaternion_xyzw']).as_matrix())
a=np.linspace(0,1,len(progress));pred=predict(nom,axis,pivot,a);gauge=predict(nom,axis,pivot+axis*.25,a)
assert np.allclose(pred['p'],gauge['p'])
# Independent homogeneous transform, same one-generator scalar/diagonal action.
for i in [0,50,100,150]:
    D=Rotation.from_rotvec(axis*np.deg2rad(10)*a[i]).as_matrix()
    H=np.eye(4);H[:3,:3]=D;H[:3,3]=(np.eye(3)-D)@pivot
    X=np.eye(4);X[:3,:3]=nom['R'][i];X[:3,3]=nom['p'][i];Y=H@X
    assert np.allclose(Y[:3,3],pred['p'][i]) and np.allclose(Y[:3,:3],pred['R'][i])
    c=np.zeros(6);c[4]=np.deg2rad(10);diag=np.diag([0,0,0,0,a[i],0]);scalar=a[i]*np.eye(6)
    assert np.allclose(diag@c,scalar@c)
for p,h in json.loads((OUT/'provenance.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
result=dict(status='PASS',calibration_n=13,evaluation_n=48,metric_rows=len(s),recomputed_metric_max_difference_deg=worst,checks=['Split disjointness','No missing evaluation episodes or radius/method/segment combinations','Quaternion and axis consistency','Independent atan2 metric recomputation','No-adaptation identity','Homogeneous SE3 action equality','Pivot nullspace invariance','Single-generator scalar/diagonal equivalence','All input provenance hashes match'])
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
print('Distinct nominal references:',m.groupby('split').nominal_episode.nunique().to_dict())
print('Radius sensitivity:',s[(s.method=='Frozen law')&(s.segment=='Apex to end')].groupby('radius_mm')[['axis_rmse_deg','position_rmse_mm']].mean().to_string())
for segment in ['Apex to end','Near to end']:
    part=s[(s.radius_mm==30)&(s.segment==segment)].pivot(index='episode',columns='method',values='axis_rmse_deg')
    for comp in ['No adaptation','Identity','Event ramp']:
        delta=part['Frozen law']-part[comp];print(segment,comp,'mean paired delta',delta.mean(),'better n',int((delta<0).sum()))
