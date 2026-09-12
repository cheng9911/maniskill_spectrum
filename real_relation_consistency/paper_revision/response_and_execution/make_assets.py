from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]/'multimodal_task_law'
MODES=['lift_early','lift_late','diagonal_early','diagonal_late']
LABELS=['A: Lift, early','B: Lift, late','C: Diagonal, early','D: Diagonal, late']
COLORS=['#0072B2','#D55E00','#0072B2','#D55E00']
STYLES=['-','-','--','--']
idata=json.loads((BASE/'id_core.json').read_text());protocol=json.loads((BASE/'protocol.json').read_text())
responses=idata['responses'];assert len(responses)==260
blocks=range(5);curves=np.zeros((4,5,150));raw=[]
for m,mode in enumerate(MODES):
 for b in blocks:
  rs=[r for r in responses if r['mode_id']==mode and r['block_id']==b and r['pitch_deg']!=0]
  assert len(rs)==12
  ks=np.array([np.array(r['r_pitch_deg'])/r['pitch_deg'] for r in rs]);assert np.isfinite(ks).all()
  curves[m,b]=ks.mean(0)
  for r,k in zip(rs,ks):
   for i,v in enumerate(k):raw.append(dict(mode=mode,block=b,pitch_deg=r['pitch_deg'],split=r['split'],sample=i,segment=i//50,response_ratio=v))
pd.DataFrame(raw).to_csv(OUT/'response_condition_source.csv',index=False)
rows=[]
for m,mode in enumerate(MODES):
 for b in blocks:
  for i,v in enumerate(curves[m,b]):rows.append(dict(mode=mode,block=b,sample=i,segment=i//50,response_ratio=v))
pd.DataFrame(rows).to_csv(OUT/'response_block_source.csv',index=False)
phase=curves.reshape(4,5,3,50).mean(-1)
effects=np.array([(phase[0]+phase[2]-phase[1]-phase[3])/2,(phase[0]+phase[1]-phase[2]-phase[3])/2])
effrows=[dict(contrast=label,block=b,segment=q,value=effects[k,b,q]) for k,label in enumerate(['Early minus late','Lift minus diagonal']) for b in blocks for q in range(3)]
pd.DataFrame(effrows).to_csv(OUT/'paired_effects.csv',index=False)
pd.DataFrame([dict(mode=mode,segment=['Approach','Entry','Insertion'][q],mean=phase[m,:,q].mean(),block_sd=phase[m,:,q].std(ddof=1)) for m,mode in enumerate(MODES) for q in range(3)]).to_csv(OUT/'phase_summary.csv',index=False)
# -----------------------------------------------------------------------------
# Paper-ready visualization v2
# Key design choice:
#   Panel (a) shows the *timing factor* after averaging over path geometry.
#   Lift and diagonal are intentionally NOT drawn as four overlapping curves;
#   their near-coincidence is quantified explicitly in panel (b).
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8.2,
    'axes.titlesize': 9.0,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.6,
    'ytick.labelsize': 7.6,
    'legend.fontsize': 7.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
})

fig = plt.figure(figsize=(7.086614, 2.62))   # 180 mm, two-column width
gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.72, 1.0],
    left=0.082, right=0.985, bottom=0.205, top=0.910,
    wspace=0.30,
)
ax = fig.add_subplot(gs[0])
bx = fig.add_subplot(gs[1])
x = np.arange(150)

EARLY = '#0072B2'
LATE  = '#D55E00'
PURPLE = '#7C4794'
GREY = '#555555'

# ---- Panel a: timing effect, marginalized over path geometry -----------------
# Pair within each block first, so the shown SD remains a block-level summary.
early_by_block = 0.5 * (curves[0] + curves[2])   # lift_early + diagonal_early
late_by_block  = 0.5 * (curves[1] + curves[3])   # lift_late  + diagonal_late

# Very light stage shading; boundaries are visible without dominating the data.
ax.axvspan(-0.5, 49.5, color='0.968', zorder=-10)
ax.axvspan(99.5, 149.5, color='0.968', zorder=-10)
for v in [49.5, 99.5]:
    ax.axvline(v, color='0.80', lw=0.70, zorder=-2)

for data, color, label in [
    (early_by_block, EARLY, 'Early alignment'),
    (late_by_block,  LATE,  'Late alignment'),
]:
    mu = data.mean(0)
    sd = data.std(0, ddof=1)
    ax.fill_between(x, mu - sd, mu + sd,
                    color=color, alpha=0.11, linewidth=0, zorder=1)
    ax.plot(x, mu, color=color, lw=1.9, label=label, zorder=3)

# Physical reference levels.
ax.axhline(0, color='0.48', lw=0.70, ls=':', zorder=0)
ax.axhline(1, color='0.48', lw=0.70, ls=':', zorder=0)

all_factor_curves = np.concatenate([early_by_block.ravel(), late_by_block.ravel()])
lo = min(-0.10, float(all_factor_curves.min()) - 0.04)
hi = max(1.12, float(all_factor_curves.max()) + 0.04)
ax.set(
    xlim=(-0.5, 149.5), ylim=(lo, hi),
    xticks=[24.5, 74.5, 124.5],
    xticklabels=['Approach', 'Entry', 'Insertion'],
    xlabel='Task progress',
    ylabel=r'Normalized pitch response  $r_\theta/\Delta\theta$',
)
# Compact panel heading: keep panel letter visually attached to the title.
# Using two text objects avoids the large gap produced by placing the letter far
# outside the axes while set_title() starts at x=0.
ax.text(-0.012, 1.025, 'a', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='bottom', ha='right', clip_on=False)
ax.text(0.012, 1.025, 'Response profile', transform=ax.transAxes,
        fontsize=9.0, fontweight='bold', va='bottom', ha='left', clip_on=False)
ax.legend(loc='upper left', frameon=False, handlelength=2.4,
          borderaxespad=0.2)

# Keep annotations away from the curves.
ax.text(148.0, 1.018, 'full propagation', ha='right', va='bottom',
        fontsize=6.6, color='0.38')
ax.text(148.0, 0.018, 'no propagation', ha='right', va='bottom',
        fontsize=6.6, color='0.38')
ax.text(0.985, 0.965, 'averaged over path geometry',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=6.5, color='0.42')

# ---- Panel b: horizontal forest plot of paired factor contrasts -------------
# Horizontal effect axis makes the large Entry timing contrast and near-zero
# path contrast readable at the same time.
contrast_colors = [PURPLE, GREY]
contrast_markers = ['o', 's']
contrast_labels = ['Early − late', 'Lift − diagonal']
phase_labels = ['Approach', 'Entry', 'Insertion']
base_y = np.array([2.0, 1.0, 0.0])
y_offsets = [0.14, -0.14]

for k in range(2):
    for q in range(3):
        vals = effects[k, :, q]
        y0 = base_y[q] + y_offsets[k]
        # Deterministic vertical jitter for the five block-level observations.
        jitter = np.linspace(-0.045, 0.045, len(vals))
        bx.scatter(
            vals, y0 + jitter,
            facecolors='none', edgecolors=contrast_colors[k],
            s=17, linewidths=0.85, alpha=0.62,
            marker=contrast_markers[k], zorder=2,
        )
        bx.errorbar(
            vals.mean(), y0, xerr=vals.std(ddof=1),
            fmt=contrast_markers[k], color=contrast_colors[k],
            markerfacecolor=contrast_colors[k],
            markeredgecolor=contrast_colors[k],
            ms=4.6, capsize=2.4, capthick=0.9, lw=1.0,
            zorder=3,
        )

# Reference line and a compact zero band to aid reading of small path effects.
bx.axvline(0, color='0.48', ls=':', lw=0.70, zorder=0)

# Dynamic limits with a small margin, but never so wide that near-zero effects vanish.
all_eff = effects.ravel()
left = min(-0.085, float(all_eff.min()) - 0.025)
right = max(0.455, float(all_eff.max()) + 0.035)
bx.set_xlim(left, right)
bx.set_ylim(-0.48, 2.48)
bx.set_yticks(base_y)
bx.set_yticklabels(phase_labels)
bx.set_xlabel('Paired contrast in normalized response')
bx.text(-0.018, 1.025, 'b', transform=bx.transAxes,
        fontsize=11, fontweight='bold', va='bottom', ha='right', clip_on=False)
bx.text(0.018, 1.025, 'Factor contrasts', transform=bx.transAxes,
        fontsize=9.0, fontweight='bold', va='bottom', ha='left', clip_on=False)

bx.legend(
    handles=[
        Line2D([], [], color=contrast_colors[k], marker=contrast_markers[k],
               lw=0, markerfacecolor=contrast_colors[k],
               label=contrast_labels[k])
        for k in range(2)
    ],
    loc='upper right', frameon=False, handletextpad=0.45,
    borderaxespad=0.2,
)

# Optional value label for the dominant effect only; avoid clutter near zero.
entry_timing_mean = effects[0, :, 1].mean()
bx.text(entry_timing_mean + 0.012, base_y[1] + y_offsets[0],
        f'{entry_timing_mean:.2f}', va='center', ha='left',
        fontsize=6.8, color=PURPLE)

for a in (ax, bx):
    a.tick_params(direction='out', length=3.2, pad=2.5)

fig.savefig(OUT/'mode_response_paper_v2.png', dpi=600,
            bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT/'mode_response_paper_v2.pdf',
            bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT/'mode_response_paper_v2.svg',
            bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT/'mode_response_paper_v2.tiff', dpi=600,
            bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

ex=json.loads((BASE/'ex_core.json').read_text());df=pd.DataFrame(ex['results']);assert len(df)==240
assert not df.duplicated(['method','target','block','theta_deg']).any()
st=protocol['strict_success']
df['endpoint_pass']=(df.depth<=st['depth_le'])&(df.lateral_mm<=1000*st['lateral_error_m'])&(np.deg2rad(df.axis_err_deg)<=st['axis_angle_rad'])
df.to_csv(OUT/'execution_run_source.csv',index=False)
methods=['no_adapt','rigid','source','target'];labels=['No adaptation','Rigid transform','Frozen source law','Target-trained law']
summ=[]
for meth,label in zip(methods,labels):
 g=df[df.method==meth];assert len(g)==60
 summ.append(dict(method=meth,label=label,n=60,env_success=int(g.success.sum()),endpoint_pass=int(g.endpoint_pass.sum()),endpoint_rate=float(g.endpoint_pass.mean()),mean_axis_deg=float(g.axis_err_deg.mean()),mean_lateral_mm=float(g.lateral_mm.mean()),no_completion='NR',axial_extension='NR',full_target_completion='measured',hold_validation='not available'))
pd.DataFrame(summ).to_csv(OUT/'execution_ablation.csv',index=False)
md=['# 执行消融表（当前可用数据）','','| 方法 | 无末端补偿 | 仅轴向延伸 | 完整目标补偿：终点达标 | env成功 | 轴误差均值（°） | 横向误差均值（mm） |','|---|---|---|---|---|---|---|']
for r in summ:md.append(f"| {r['label']} | 未测 | 未测 | {r['endpoint_pass']}/60 ({100*r['endpoint_rate']:.1f}%) | {r['env_success']}/60 | {r['mean_axis_deg']:.2f} | {r['mean_lateral_mm']:.2f} |")
md+=['','终点达标重算条件：横向误差≤10 mm、轴角误差≤0.05 rad、轴向坐标≤0.072 m。不是已验证持续成功或物理接触真值。protocol另含hold_steps=15，但ex_core.json只存终端量，不能核验保持窗口。所有均值包含失败运行。现有执行代码使用40 waypoint及screw/RRT fallback，末端追加真实孔完整目标位姿；运行级补偿是否完成/实际RRT调用未单独记录。NR不是0%失败。当前表是可追溯的部分消融表，不是完成的归因实验。']
(OUT/'execution_ablation.md').write_text('\n'.join(md)+'\n')
tex=r'''% Requires booktabs. NR = not run; do not replace NR by zero.
\begin{table*}[t]
\centering
\caption{Execution outcomes under the available terminal-completion setting. Missing ablation arms are explicitly marked NR.}
\label{tab:mode_execution_ablation}
\small
\begin{tabular}{lccccc}
\toprule
Method & No completion & Axial extension only & Full-target completion & Axis error ($^\circ$) & Lateral error (mm) \\
\midrule
'''
for r in summ:tex+=f"{r['label']} & NR & NR & {r['endpoint_pass']}/60 ({100*r['endpoint_rate']:.1f}\\%) & {r['mean_axis_deg']:.2f} & {r['mean_lateral_mm']:.2f} \\\\\n"
tex+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\textwidth}\footnotesize
Each method is evaluated on three target modes, four held-out pitch conditions and five blocks (60 runs). Entries report endpoint-criterion attainment, requiring lateral error $\leq10$ mm, axis error $\leq0.05$ rad, and axial coordinate $\leq0.072$ m. The 15-step hold criterion cannot be verified from the stored terminal summaries. Error means include failed runs. The available pipeline plans between generated waypoints and appends the full target pose. NR denotes an unmeasured arm, not failure. Target-trained law uses target training interventions and is a reference comparator.
\end{minipage}
\end{table*}
'''
(OUT/'execution_ablation.tex').write_text(tex)
caption=r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{mode_response_paper_v2.pdf}
\caption{Phase-dependent pitch intervention response and factorial contrasts. (a) Normalized pitch response $r_\theta/\Delta\theta$ over geometrically aligned task progress, after averaging over lift and diagonal path geometries within each block. Curves show five-block means and shaded bands show $\pm1$ block SD; horizontal references denote no response ($0$) and full propagation ($1$). (b) Paired factor contrasts from segment averages. Open markers show block-level contrasts and filled markers with horizontal error bars show mean $\pm1$ block SD. The pronounced Early--Late contrast at Entry, together with the near-zero Lift--Diagonal contrasts, indicates that the response onset is associated with alignment timing rather than nominal path geometry. This is a descriptive mechanism analysis over all available conditions, not a held-out prediction score or an equivalence test.}
\label{fig:mode_response}
\end{figure*}
'''
(OUT/'mode_response.tex').write_text(caption)
(OUT/'provenance.json').write_text(json.dumps({'sources':{str(BASE/n):hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ['id_core.json','ex_core.json','protocol.json']},'response_count':240,'nominal_count':20,'blocks':5,'execution_count':240,'new_simulation_runs':0,'strict_hold_verified':False,'summary':summ},indent=2)+'\n')
print(pd.DataFrame(summ)[['method','env_success','endpoint_pass','mean_axis_deg','mean_lateral_mm']].to_string(index=False))
print('phase means:',phase.mean(1).round(4));print('curve limits',lo,hi)
