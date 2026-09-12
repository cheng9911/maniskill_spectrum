"""All-episode kinematic phase proposals, explicitly including unresolved spans."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
BASE=Path(__file__).resolve().parent;OUT=BASE/'real_phase_identification'
windows=pd.read_csv(BASE/'fk_results/terminal_pose_per_episode.csv');labels=['Unresolved','Grasp/lift','Transport','Prealignment','Axial candidate','Settle candidate'];colors=['#D3D3D3','#DDAD72','#90AFCA','#A798BE','#65A99A','#396B8D'];codes={'grasp_lift':1,'transport':2,'prealignment':3,'axial_advance_candidate':4,'terminal_settle_candidate':5}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(7.086614,4.7),layout='constrained');source=[]
for col,name in enumerate(['insert_jc_fix_3','insert_10degree']):
 rows=[];group=windows[windows.dataset==name]
 for _,r in group.iterrows():
  f=pd.read_csv(OUT/'frame_labels'/f'{name}_{int(r.episode_index):03d}.csv');idx=np.rint(np.linspace(int(r.grasp_start),int(r.release_frame)-1,301)).astype(int);v=np.array([codes.get(s,0) for s in f.stage.iloc[idx]]);rows.append(v)
  source.extend(dict(dataset=name,episode=int(r.episode_index),normalized_grasp_progress=i/300,stage_code=int(code)) for i,code in enumerate(v))
 axs[col].imshow(np.asarray(rows),aspect='auto',interpolation='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(7)-.5,6),extent=[0,1,len(rows)-.5,-.5]);axs[col].set(title=('a  Upright: 50 episodes' if col==0 else 'b  Tilted: 61 episodes'),xlabel='Normalized grasp-to-release time',ylabel='Episode index')
fig.suptitle('Offline stage proposals; gray intervals remain unresolved',fontsize=8);fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc='outside lower center',ncol=3,fontsize=6,frameon=False)
fig.savefig(OUT/'phase_timeline.png',dpi=600);fig.savefig(OUT/'phase_timeline.pdf');fig.savefig(OUT/'phase_timeline.svg');fig.savefig(OUT/'phase_timeline.tiff',dpi=600);plt.close(fig)
pd.DataFrame(source).to_csv(OUT/'phase_timeline_source.csv',index=False)
