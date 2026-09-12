# Simulation scenes figure

- task_scene_overview.pdf:180mm double-column figure; SVG editable labels, PNG/TIFF600dpi.
- task_scene_overview.tex:English caption and figure* insertion.
- make_overview.py:Python composition; crops retain aspect ratio, no generative images or response data invented.
- render_sources.py:replays qpos/peg/socket states in actual environment at1024×1024; no new rollout trial. Same keyed episode7:frames294/357/501/515. Key clearance uses transformed top-corner geometry, not phase labels alone.
- video_manifest.json:timestamps and source clips for four state-level scene probes. Still robots are scene context, not claimed execution.
- render_manifest.json / layout_sources.json / provenance.json:state indices, camera, crop and hashes.

Geometry panels:actual alignment closeups plus schematic sections with key already below gate. They are different states, explicitly noted in caption. Schematics visualize nominal geometry, not measured contacts or response law.

Planar transport now replays the actual heading-constrained PlanarPush-v1 rollout, episode_11 frame310. The collector uses guided grasp-and-slide, not non-prehensile point pushing. See planar_manifest.json and render_planar.py. The other three relation cells are state-level probes. Bowl/stove panel illustrates scene context rather than claiming the shown bowl is already at goal. All source/target object names should remain consistent with paper tables.

QA:source preflight20pass0warn0fail; PDF text minimum6pt. Source screenshots:1024² ManiSkill and512² existing LIBERO videos; final600dpi export does not increase intrinsic video detail. Visually reviewed overall and panels; image crops documented, no aspect stretching. Existing manuscript overview not overwritten. No LaTeX full-document compile performed.
