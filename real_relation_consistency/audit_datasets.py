"""Read-only LeRobot audit; does not infer interventions from robot motion.

Run with the existing lerobot environment's Python (-s recommended).
Outputs are descriptive, not estimates of the relation operator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, out):
    info = json.loads((root / 'meta/info.json').read_text())
    files = sorted((root / 'data').glob('*/*.parquet'))
    meta_files = sorted((root / 'meta/episodes').glob('*/*.parquet'))
    data = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    meta = pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)
    rows, annotations = [], []
    for episode, group in data.groupby('episode_index', sort=True):
        group = group.sort_values('frame_index')
        state = np.stack(group['observation.state']).astype(float)
        action = np.stack(group['action']).astype(float)
        timestamp = group.timestamp.to_numpy(dtype=float)
        frame = group.frame_index.to_numpy()
        em = meta[meta.episode_index == episode]
        row = dict(dataset=root.name, episode_index=int(episode), frames=len(group),
                   duration_s=float(timestamp[-1] - timestamp[0]),
                   nonfinite_state=int((~np.isfinite(state)).sum()),
                   nonfinite_action=int((~np.isfinite(action)).sum()),
                   duplicate_frame_indices=int(group.frame_index.duplicated().sum()),
                   nonunit_frame_steps=int((np.diff(frame) != 1).sum()),
                   nonincreasing_timestamps=int((np.diff(timestamp) <= 0).sum()),
                   median_dt_s=float(np.median(np.diff(timestamp))) if len(group)>1 else None,
                   max_dt_s=float(np.max(np.diff(timestamp))) if len(group)>1 else None,
                   metadata_rows=len(em),
                   metadata_length_match=bool(len(em)==1 and em.iloc[0]['length']==len(group)))
        # Keep channels generic: metadata does not establish physical units.
        for j in range(state.shape[1]):
            row[f'state_{j}_start'] = float(state[0,j])
            row[f'state_{j}_end'] = float(state[-1,j])
            row[f'state_{j}_min'] = float(state[:,j].min())
            row[f'state_{j}_max'] = float(state[:,j].max())
        rows.append(row)
        annotations.append(dict(dataset=root.name, episode_index=int(episode),
            session_id='', condition_id='', successful='', task_geometry='',
            intervention_frame='', du_m='', dv_m='', dw_m='', roll_rad='',
            pitch_rad='', yaw_rad='', grasp_frame='', alignment_frame='',
            contact_frame='', insertion_end_frame='', source_of_labels='', notes=''))
    per_episode = pd.DataFrame(rows)
    per_episode.to_csv(out / f'{root.name}_episodes.csv', index=False)
    tasks = pd.read_parquet(root / 'meta/tasks.parquet').reset_index().to_dict('records')
    summary = dict(dataset=root.name, episodes=int(data.episode_index.nunique()),
        frames=len(data), fps=info['fps'], declared_episodes=info['total_episodes'],
        declared_frames=info['total_frames'], tasks=tasks,
        data_columns=data.columns.tolist(), episode_metadata_columns=meta.columns.tolist(),
        duration_median_s=float(per_episode.duration_s.median()),
        duration_min_s=float(per_episode.duration_s.min()),
        duration_max_s=float(per_episode.duration_s.max()),
        nonfinite_state=int(per_episode.nonfinite_state.sum()),
        nonfinite_action=int(per_episode.nonfinite_action.sum()),
        duplicate_frame_indices=int(per_episode.duplicate_frame_indices.sum()),
        nonunit_frame_steps=int(per_episode.nonunit_frame_steps.sum()),
        nonincreasing_timestamps=int(per_episode.nonincreasing_timestamps.sum()),
        metadata_length_mismatches=int((~per_episode.metadata_length_match).sum()),
        metadata_duplicate_episodes=int(meta.episode_index.duplicated().sum()),
        metadata_extra_episodes=sorted(set(meta.episode_index)-set(data.episode_index)),
        state_shape=info['features']['observation.state']['shape'],
        action_shape=info['features']['action']['shape'],
        relation_operator_identifiable_from_recorded_tables=False,
        missing=['known task-frame interventions', 'socket/peg pose calibration',
                 'phase correspondence', 'success labels', 'session/block labels'])
    manifest = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha256(p))
                for p in files+meta_files+[root/'meta/info.json',root/'meta/tasks.parquet']]
    return summary, annotations, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sim-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, annotations, manifest, sim = [], [], [], []
    for root in args.datasets:
        s, a, m = audit(root, args.out)
        summaries.append(s); annotations.extend(a); manifest.extend(m)
    # Template is separate from a completed annotation file and never overwritten.
    template = args.out / 'episode_annotations_template.csv'
    if not template.exists():
        pd.DataFrame(annotations).to_csv(template, index=False)
    for path in sorted(args.sim_dir.glob('circular_honest_seed_*.json')):
        doc = json.loads(path.read_text())
        for e in doc['episodes']:
            sim.append(dict(seed=doc['seed'], episode_id=e['episode_id'],
                condition_id=e['condition_id'], generator=e['generator'],
                success=e.get('success'), complete=e.get('complete'),
                **dict(zip(['du','dv','dw','roll','pitch','yaw'],e['causal_delta']))))
        manifest.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path)))
    pd.DataFrame(sim).to_csv(args.out/'simulation_interventions.csv',index=False)
    result = dict(status='AWAITING_EXPERIMENT_METADATA_NOT_A_LAW_TEST',
        datasets=summaries, simulation_attempts=len(sim),
        software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
        exclusions='None; all tabular rows and state channels audited.',
        inference='No response coefficients, significance tests, or equivalence claim computed.')
    (args.out/'audit_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (args.out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({s['dataset']:{k:s[k] for k in ['episodes','frames','duration_median_s',
        'nonfinite_state','nonfinite_action','duplicate_frame_indices','nonunit_frame_steps',
        'nonincreasing_timestamps','metadata_length_mismatches']} for s in summaries},indent=2))


if __name__ == '__main__':
    main()
