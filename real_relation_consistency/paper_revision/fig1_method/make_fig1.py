from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Ellipse,FancyArrowPatch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
SRC=ROOT/'phase_switch_symmetry_baselines/phase_switch_baseline_profiles.csv'
df=pd.read_csv(SRC);d=df[df.model=='Pdiag finite'].copy();assert len(d)>0
d.to_csv(OUT/'source_response.csv',index=False)
W,H=180,96;ops=[];INK='#243442';BLUE='#0072B2';ORANGE='#C65A29';GRAY='#74818C';LIGHT='#E4E9ED';TEAL='#86AAA5'
def text(x,y,t,size=7,color=INK,bold=False,align='left',w=50,h=5):ops.append(dict(kind='text',x=x,y=y,text=t,size=size,color=color,bold=bold,align=align,w=w,h=h))
def rect(x,y,w,h,fill='#FFFFFF',edge=GRAY,lw=.7):ops.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,edge=edge,lw=lw))
def line(points,color=GRAY,lw=.7,dash=False,arrow=False):ops.append(dict(kind='line',points=np.asarray(points).tolist(),color=color,lw=lw,dash=dash,arrow=arrow))
def ellipse(x,y,w,h,fill='#FFFFFF',edge=GRAY,lw=.7):ops.append(dict(kind='ellipse',x=x,y=y,w=w,h=h,fill=fill,edge=edge,lw=lw))
# Three quiet columns, no decorative title strip.
for x in [59,119]:line([[x,5],[x,88]],'#DDE2E6',.6)
text(3,4,'a  Why generator-level relevance?',8,bold=True,w=55)
text(64,4,'b  Identify from interventions',8,bold=True,w=54)
text(124,4,'c  Freeze and transfer',8,bold=True,w=53)
text(3,12,'One relation; constraints change by stage.',6.7,w=54)
text(64,12,'Vary the relation; observe the response.',6.7,w=54)
text(124,12,'Reuse the law with target geometry.',6.7,w=53)
# Sectional keyed apparatus, two stages. Dimensions are explanatory, not scaled.
def socket(cx,cy,cleared):
 for xx in [cx-9,cx+5]:rect(xx,cy+3,4,15,TEAL,'#537B76',.65)
 for xx in [cx-9,cx+4]:rect(xx,cy,5,3,'#93A1BC','#66768E',.65)
 py=cy-7 if not cleared else cy+7
 rect(cx-2,py-12,4,14,'#E7A486','#B96747',.65)
 rect(cx-3.5,py+2,7,3,'#D97E58','#B96747',.65)
 line([[cx+10,cy-15],[cx+10,cy-9]],BLUE,.8,arrow=True)
 line([[cx-11,cy+3],[cx+11,cy+3]],GRAY,.5,True)
socket(16,40,False);socket(44,40,True)
text(16,20,'Before clearance',6.8,bold=True,align='center',w=26)
text(44,20,'After clearance',6.8,bold=True,align='center',w=26)
text(16,61,'Yaw constrained',6.6,align='center',w=27)
text(44,61,'Yaw admissible',6.6,align='center',w=27)
line([[28,44],[31,44]],GRAY,.8,arrow=True)
text(4,72,'Position alignment remains required.',6.6,w=54)
text(4,79,'Admissible yaw does not force\na zero demonstrated yaw response.',6.5,w=54,h=8)
# Three context observations, primitive target cues.
for x,label in [(69,'Nominal'),(87,'Translate'),(105,'Rotate')]:
 rect(x,24,9,7,'#F1F4F6',GRAY,.7);ellipse(x+3,25.5,3,2,'#FFFFFF',GRAY,.65)
 if label=='Translate':line([[x-1,34],[x+10,34]],BLUE,.8,arrow=True)
 if label=='Rotate':
  ang=np.linspace(.25*np.pi,1.55*np.pi,20);pts=np.c_[x+4.5+6*np.cos(ang),27.5+5*np.sin(ang)];line(pts,ORANGE,.8,arrow=True)
 text(x+4.5,37,label,6.4,align='center',w=18)
line([[90,42],[90,46]],GRAY,.8,arrow=True)
text(90,49,'P(s) = diag[α₁(s), …, αd(s)]',8,align='center',w=52)
# Compact plot of actual archived model outputs, no fabricated curves.
x0,y0,pw,ph=69,78,44,20
line([[x0,56],[x0,y0],[x0+pw,y0]],INK,.65)
for val in [0,1]:
 yy=y0-ph*val/1.15;line([[x0,yy],[x0+pw,yy]],'#CCD3D8',.4,True);text(x0-2,yy,str(val),6,align='right',w=5)
for j in [25,50,75]:line([[x0+pw*j/100,56],[x0+pw*j/100,y0]],'#D9DEE2',.45,True)
s=np.arange(len(d))/(len(d)-1)
for field,color in [('g_translation',BLUE),('g_yaw',ORANGE)]:line(np.c_[x0+pw*s,y0-ph*d[field].to_numpy()/1.15],color,1.2)
text(x0,81,'0',6,align='center',w=4);text(x0+pw,81,'1',6,align='center',w=4)
text(x0+9,85,'Phase-normalized progress s',6.2,w=44)
text(72,56,'Translation',6.1,BLUE,w=24);text(108,74,'Yaw',6.1,ORANGE,align='right',w=16)
text(90,89,'Recorded source fit · selected channels',6,align='center',w=55)
# Target law + one nominal + relation, no target intervention refitting.
rect(128,22,46,10,'#FFFFFF',BLUE,.9)
text(151,27,'Frozen source law  Psrc(s)',7.3,BLUE,True,'center',w=44)
line([[151,32],[151,37]],BLUE,.9,arrow=True)
text(151,41,'One target reference X₀,tgt(s)',7,align='center',w=50)
text(151,48,'+ target relation change ctgt',7,align='center',w=50)
line([[151,52],[151,57]],GRAY,.9,arrow=True)
# nominal and adapted path on a simple task plane: explicitly illustrative.
line([[130,76],[173,76]],'#ADB8C0',.6)
ellipse(154,73,8,4,'#FFFFFF',GRAY,.7);ellipse(165,70,8,4,'#FFFFFF',BLUE,.8)
line([[137,59],[139,65],[148,69],[158,74]],GRAY,1,True)
line([[137,59],[144,60],[159,63],[169,71]],BLUE,1.25)
text(129,80,'Nominal',6.1,GRAY,w=22);text(159,80,'Adapted',6.1,BLUE,w=22)
text(151,86,'Trajectory schematic; not an execution result',5.8,align='center',w=55)
line([[3,91],[177,91]],'#DDE2E6',.6)
text(90,94,'Transfer requires matched generator semantics and specified progress alignment.',6.4,align='center',w=174)
# Export vector composition and operations used to construct editable PPT.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.086614,3.779528));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
for o in ops:
 k=o['kind']
 if k=='text':ax.text(o['x'],o['y'],o['text'],fontsize=o['size'],color=o['color'],fontweight='bold' if o['bold'] else 'normal',ha=o['align'],va='center',linespacing=1.3)
 elif k=='rect':ax.add_patch(Rectangle((o['x'],o['y']),o['w'],o['h'],facecolor=o['fill'],edgecolor=o['edge'],lw=o['lw']))
 elif k=='ellipse':ax.add_patch(Ellipse((o['x']+o['w']/2,o['y']+o['h']/2),o['w'],o['h'],facecolor=o['fill'],edgecolor=o['edge'],lw=o['lw']))
 else:
  pts=np.array(o['points']);ax.plot(pts[:,0],pts[:,1],color=o['color'],lw=o['lw'],ls='--' if o['dash'] else '-')
  if o['arrow']:ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=7,color=o['color'],lw=o['lw']))
fig.savefig(OUT/'fig1_method.png',dpi=600);fig.savefig(OUT/'fig1_method.pdf');fig.savefig(OUT/'fig1_method.svg');fig.savefig(OUT/'fig1_method.tiff',dpi=600);plt.close(fig)
(OUT/'drawing_operations.json').write_text(json.dumps(dict(width_mm=W,height_mm=H,operations=ops),ensure_ascii=False,indent=2))
(OUT/'fig1_caption.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig1_method.pdf}
\caption{Generator-level response identification and matched-relation transfer. (a) Schematic keyed insertion illustrates how admissible motion changes after key clearance while positional alignment remains required. Geometric permission does not uniquely determine the demonstrated response. (b) Controlled relation interventions provide supervision for a generator-wise model. The inset reproduces selected translation and yaw outputs of an archived finite diagonal source fit; it illustrates the representation, not cross-strategy invariance or an uncertainty estimate. (c) The frozen source model is combined with one target nominal trajectory and the target relation change. The paths are schematic, not measured hardware results. Transfer is evaluated under matched generator semantics and specified progress alignment.}
\label{fig:method_overview}
\end{figure*}
''')
(OUT/'provenance.json').write_text(json.dumps(dict(source=str(SRC),sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),filter="model == 'Pdiag finite'",channels=['g_translation','g_yaw'],n=len(d),curve_status='single archived fit; no confidence band inferred',diagram_status='schematic, not measured execution',new_experiments=0),indent=2))
print('rendered',len(ops),'editable primitives')
