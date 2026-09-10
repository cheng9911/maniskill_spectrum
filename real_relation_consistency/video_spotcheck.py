"""Sample metadata-aligned pre-release frames; record video timing mismatches."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/home/rocos/sia/Sim/robomimic/dataset/data/SunJincheng')
OUT=Path(__file__).parent/'fk_results'
FFMPEG='/home/rocos/miniconda3/envs/lerobot/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2'
samples={'insert_jc_fix_3':[0,10,20,30,40,49],'insert_10degree':[0,12,24,31,36,48,60]}
poses=pd.read_csv(OUT/'terminal_pose_per_episode.csv')
(OUT/'video_spotcheck').mkdir(exist_ok=True)
records=[];timings=[]
for name,episodes in samples.items():
    root=BASE/name
    meta=pd.concat([pd.read_parquet(p) for p in sorted((root/'meta/episodes').glob('*/*.parquet'))])
    for _,m in meta.iterrows():
        duration=m['videos/observation.images.head/to_timestamp']-m['videos/observation.images.head/from_timestamp']
        timings.append(dict(dataset=name,episode_index=int(m.episode_index),video_duration_s=duration,
            nominal_data_duration_s=m.length/30,delta_s=duration-m.length/30))
    for ep in episodes:
        m=meta[meta.episode_index==ep].iloc[0];p=poses[(poses.dataset==name)&(poses.episode_index==ep)].iloc[0]
        frame=int((p.window_start+p.window_end_exclusive-1)/2)
        timestamp=m['videos/observation.images.head/from_timestamp']+frame/30
        chunk=int(m['videos/observation.images.head/chunk_index']);file=int(m['videos/observation.images.head/file_index'])
        video=root/f'videos/observation.images.head/chunk-{chunk:03d}/file-{file:03d}.mp4'
        image=OUT/'video_spotcheck'/f'{name}_{ep:03d}.png'
        subprocess.run([FFMPEG,'-hide_banner','-loglevel','error','-c:v','libdav1d','-ss',str(timestamp),'-i',str(video),'-frames:v','1','-y',str(image)],check=True)
        records.append(dict(dataset=name,episode_index=ep,data_frame=frame,video_timestamp_s=timestamp,
                            image=str(image),terminal_tilt_deg=p.terminal_tilt_deg))
pd.DataFrame(records).to_csv(OUT/'video_spotcheck_manifest.csv',index=False)
pd.DataFrame(timings).to_csv(OUT/'video_timing_audit.csv',index=False)
fig,axes=plt.subplots(4,4,figsize=(12,9),layout='constrained')
for ax in axes.flat:ax.axis('off')
for ax,r in zip(axes.flat,records):
    ax.imshow(plt.imread(r['image']));ax.set_title(f"{'Upright' if r['dataset']=='insert_jc_fix_3' else 'Tilted'} ep {r['episode_index']} / {r['terminal_tilt_deg']:.1f}°",fontsize=9)
fig.savefig(OUT/'video_spotcheck_contact_sheet.png',dpi=130);plt.close(fig)
print(json.dumps(dict(sampled_frames=len(records),timing_mismatches=[r for r in timings if abs(r['delta_s'])>1/30]),indent=2))
