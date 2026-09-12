"""Score independent video event intervals; never treats unlabeled frames as truth."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent/'real_phase_identification'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('annotations',type=Path);ap.add_argument('--include_prediction_exposed',action='store_true');ap.add_argument('--out',type=Path,default=BASE/'annotation_evaluation');args=ap.parse_args();manual=json.loads(args.annotations.read_text());pred=pd.read_csv(BASE/'events.csv');rows=[];excluded=[]
 for ann in manual['annotations']:
  exposure=ann.get('predictions_shown',None)
  if exposure is not False and not args.include_prediction_exposed:
   excluded.append(dict(dataset=ann['dataset'],episode=ann['episode'],reason='predictions_exposed_or_unknown'));continue
  for event,label in ann['events'].items():
   if label['status']!='observed':continue
   lo=label.get('lower_clip_s');hi=label.get('upper_clip_s')
   if lo is None or hi is None or not (0<=lo<=hi):raise ValueError(f'Invalid interval for {ann["dataset"]}/{ann["episode"]}/{event}')
   lo+=ann['clip_start_frame']/ann['source_fps'];hi+=ann['clip_start_frame']/ann['source_fps'];g=pred[(pred.dataset==ann['dataset'])&(pred.episode==ann['episode'])&(pred.event==event)]
   detected=len(g)==1 and pd.notna(g.iloc[0].time_s);pt=float(g.iloc[0].time_s) if detected else None
   gap=max(lo-pt,pt-hi,0) if detected else None
   rows.append(dict(dataset=ann['dataset'],episode=ann['episode'],rater=ann.get('rater',''),prediction_exposure='blind' if exposure is False else ('exposed' if exposure is True else 'unknown'),event=event,manual_low_s=lo,manual_high_s=hi,predicted_s=pt,detected=detected,interval_distance_s=gap,within_0p2s=bool(detected and gap<=.2),within_0p5s=bool(detected and gap<=.5),interval_width_s=hi-lo))
 out=args.out;out.mkdir(exist_ok=True,parents=True)
 if not rows:
  for filename in ['per_annotation.csv','summary.csv']:
   (out/filename).unlink(missing_ok=True)
  (out/'status.json').write_text(json.dumps(dict(status='NO_INDEPENDENT_OBSERVED_LABELS',n=0,excluded=excluded),indent=2)+'\n');print('No independent observed labels; accuracy is not estimable.');return
 df=pd.DataFrame(rows);df.to_csv(out/'per_annotation.csv',index=False);summary=df.groupby(['prediction_exposure','event']).agg(n=('detected','size'),detected=('detected','sum'),mean_interval_distance_detected_only_s=('interval_distance_s','mean'),median_interval_distance_detected_only_s=('interval_distance_s','median'),within_0p2s=('within_0p2s','mean'),within_0p5s=('within_0p5s','mean'),median_interval_width_s=('interval_width_s','median'));summary.to_csv(out/'summary.csv');print(summary.to_string())
 (out/'status.json').write_text(json.dumps(dict(status='EVALUATED',prediction_blind_observed_annotations=int((df.prediction_exposure=='blind').sum()),prediction_exposed_observed_annotations=int((df.prediction_exposure=='exposed').sum()),unknown_exposure_observed_annotations=int((df.prediction_exposure=='unknown').sum()),excluded=excluded,note='Event interval agreement, not frame accuracy or contact success. Error summaries conditional on detected predictions; tolerance proportions include misses. Blindness is self-reported. Repeated raters/episodes not independent.'),indent=2)+'\n')
if __name__=='__main__':main()
