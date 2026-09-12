from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.086614,5.62),facecolor='white')
cols=[.035,.282,.529,.776];w=.215
blue='#0072B2';orange='#D55E00';ink='#253344'
sources=[]
def picture(name,x,y,h,crop=None,title='',note='',box=None):
 img=plt.imread(OUT/'frames'/name);ny,nx=img.shape[:2]
 x0,y0,x1,y1=crop if crop else (0.,0.,1.,1.)
 # Center-crop to the physical panel ratio; never stretch a scene.
 ratio=(w*7.086614)/(h*5.62)
 bw=(x1-x0)*nx;bh=(y1-y0)*ny
 if bw/bh<ratio:
  keep=bw/ratio/ny;cy=(y0+y1)/2;y0=cy-keep/2;y1=cy+keep/2
 else:
  keep=bh*ratio/nx;cx=(x0+x1)/2;x0=cx-keep/2;x1=cx+keep/2
 img=img[int(y0*ny):int(y1*ny),int(x0*nx):int(x1*nx)]
 crop=(x0,y0,x1,y1)
 ax=fig.add_axes([x,y,w,h]);ax.imshow(img,extent=[0,1,0,1]);ax.set_aspect('auto');ax.set_axis_off()
 if box:
  bx,by,bw,bh=box;box=((bx-x0)/(x1-x0),(by-(1-y1))/(y1-y0),bw/(x1-x0),bh/(y1-y0))
 ax.set_title(title,pad=4,color=ink,fontweight='bold')
 if box:ax.add_patch(Rectangle((box[0],box[1]),box[2],box[3],fill=False,edgecolor='#F0B740',lw=1.1))
 if note:fig.text(x+w/2,y-.022,note,ha='center',va='top',fontsize=6.5,color=ink)
 sources.append(dict(image=name,crop_normalized_xyxy=crop,display_box=box))
 return ax
fig.text(.035,.968,'a  Task geometry',fontsize=9,fontweight='bold',color=ink)
# fig.text(.99,.968,'Actual scene closeups + sectional schematics',ha='right',fontsize=6.5,color='#555555')
picture('keyed_3.png',cols[0],.735,.178,crop=(.36,.34,.68,.70),title='Keyed peg and gate',note='Actual simulation geometry')
picture('circular_3.png',cols[2],.735,.178,crop=(.36,.32,.68,.68),title='Circular control',note='Actual simulation geometry')
def section(x,keyed):
 ax=fig.add_axes([x,.735,w,.178]);ax.set(xlim=(-.074,.074),ylim=(-.006,.144));ax.axis('off')
 ax.set_title('Key cleared: section' if keyed else 'Circular gate: section',pad=4,color=ink,fontweight='bold')
 # Actual nominal widths and gate heights; schematic not a simulation image.
 for sign in [-1,1]:
  left=-.055 if sign<0 else .030
  ax.add_patch(Rectangle((left,0),.025,.053,facecolor='#A5C6C3',edgecolor='#567D7A',lw=.6))
 gap=.023 if keyed else .015
 ax.add_patch(Rectangle((-.055,.053),.055-gap,.012,facecolor='#8396B7',edgecolor='#596B8A',lw=.6))
 ax.add_patch(Rectangle((gap,.053),.055-gap,.012,facecolor='#8396B7',edgecolor='#596B8A',lw=.6))
 center=.077 if keyed else .080
 ax.add_patch(Rectangle((-.013,center-.029),.026,.074,facecolor='#E39471',edgecolor='#AC5A3F',lw=.6))
 if keyed:ax.add_patch(Rectangle((-.021,center-.045),.042,.016,facecolor='#D77858',edgecolor='#AC5A3F',lw=.6))
 ax.annotate('',xy=(0,.128),xytext=(0,.143),arrowprops={'arrowstyle':'->','color':blue,'lw':1})
 ax.text(.010,.132,'Axis',fontsize=6,color=blue)
 ax.annotate('Gate',xy=(.040,.059),xytext=(.039,.090),ha='center',fontsize=6,arrowprops={'arrowstyle':'-','lw':.55})
 ax.text(0,.012,'Bore',ha='center',fontsize=6,color='#426664')
 if keyed:
  ax.annotate('Key',xy=(-.018,.041),xytext=(-.050,.096),fontsize=6,ha='center',arrowprops={'arrowstyle':'-','lw':.55})
  ax.plot([-.024,.024],[.053,.053],ls=':',color='#666666',lw=.65)
 fig.text(x+w/2,.713,'Schematic; yaw permitted below gate' if keyed else 'Schematic; no keyed yaw constraint',ha='center',va='top',fontsize=6.2,color=ink)
section(cols[1],True);section(cols[3],False)
fig.text(.035,.654,'b  Keyed-to-circular insertion',fontsize=9,fontweight='bold',color=ink)
# fig.text(.99,.654,'Closed-loop simulation · one recorded rollout',ha='right',fontsize=6.5,color='#555555')
man=json.loads((OUT/'render_manifest.json').read_text());phase_lookup={r['solver_phase']:r for r in man if 'keyed_' in r['image']}
for k,(phase,title,note) in enumerate([(3,'Align','Match the keyed opening'),(4,'Enter','Key has not fully cleared'),(5,'Clear / Unlock','Key below gate; yaw permitted'),(6,'Insert','Continue into circular bore')]):
 r=phase_lookup[phase]
 if phase in [5,6]:assert r['clearance_margin']>0
 if phase==4:assert r['clearance_margin']<0
 picture(f'keyed_{phase}.png',cols[k],.405,.19,crop=(.20,.15,.87,.82),title=title,note=note)
 if k<3:fig.add_artist(FancyArrowPatch((cols[k]+w+.003,.505),(cols[k+1]-.007,.505),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.8,color='#777777'))
fig.text(.035,.326,'c  Representative task relations',fontsize=9,fontweight='bold',color=ink)
# fig.text(.99,.326,'Evidence type is indicated for each scene',ha='right',fontsize=6.5,color='#555555')
scene=[('drawer_middle_open.png','Sliding','Drawer · state-level probe',(.10,.14,.37,.57)),('planar_heading.png','Planar transport','Heading-constrained · closed-loop',None),('microwave_door_revolute.png','Revolute','Microwave door · state-level probe',(.51,.19,.45,.57)),('bowl_on_stove.png','Support / placement','Bowl–stove · state-level probe',(.35,.30,.36,.32))]
for k,(name,title,note,box) in enumerate(scene):picture(name,cols[k],.082,.19,crop=((.15,.15,.9,.9) if name=='planar_heading.png' else (0,.28,1,1)),title=title,note=note,box=box)
# fig.text(.5,.018,'Geometry explains permitted motion; observed generator response also depends on the execution policy.',ha='center',fontsize=6.5,color='#555555')
fig.savefig(OUT/'task_scene_overview.png',dpi=600)
fig.savefig(OUT/'task_scene_overview.pdf')
fig.savefig(OUT/'task_scene_overview.svg')
fig.savefig(OUT/'task_scene_overview.tiff',dpi=600)
plt.close(fig)
(OUT/'layout_sources.json').write_text(json.dumps(sources,indent=2))
(OUT/'task_scene_overview.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{task_scene_overview.pdf}
\caption{Simulation geometry, insertion stages, and representative task relations. (a) Closeups replayed in the keyed and circular environments, paired with explicitly schematic sections of the gate and bore. The keyed section depicts the key after clearance; it is not a section of the adjacent alignment frame. Axial rotation becomes admissible below the keyed gate, whereas the circular control has no keyed yaw constraint. These geometric permissions do not prescribe a zero yaw-response coefficient. (b) Four states from one successful keyed rollout rendered with the same camera and crop. Key clearance is checked from the recorded pose and key geometry; Unlock denotes the controller phase following clearance. (c) Representative task relations. Planar transport replays the heading-constrained PlanarPush-v1 closed-loop benchmark, whose collector uses a guided grasp-and-slide procedure. Drawer, microwave and bowl scenes are controlled state-level probes; their visible robots are scene context rather than evidence of execution. Outlines locate objects in the state-level scenes. Panels are illustrative states, not quantitative success comparisons.}
\label{fig:task_scene_overview}
\end{figure*}
''')
paths=[OUT/'render_manifest.json',OUT/'video_manifest.json',OUT/'planar_manifest.json',Path(__file__).resolve()]+list((OUT/'frames').glob('*.png'))
(OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print('Exported task_scene_overview PNG PDF SVG TIFF + LaTeX and source manifests')
