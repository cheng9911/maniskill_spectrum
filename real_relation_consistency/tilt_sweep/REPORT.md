# 倾斜幅度 sweep → 辨识 α_pitch(s) → 冻结 → ±10° 留出插值校验 + 真机幅度对照

这是「继续」的下一步：从**不含 ±10°** 的 pitch 幅度 sweep 辨识跨幅度响应律 `P_sim(s)=α_pitch(s)`，冻结，再用冻结律对 **±10°** 做有限变换预测，并同时与 (a) **独立采集的仿真 ±10° 轨迹**（留出插值校验）和 (b) **真机终端幅值** 对照。

## 采集

专用 pitch 幅度 sweep，与 sim_10degree 同一 setup（`task_anchor [-0.15,0,0.08]`、`CircularPhaseSwitchSE3-v1`、circular honest、单参考拾取位 `[-0.25,-0.18,0.28]`，即 `PEDESTAL_POS`）：

- 幅度 θ ∈ {0, ±2.5, ±5, ±7.5, ±12.5, ±15}°，**±10° 作 hold-out**（真机幅度，不进入辨识）。
- 11 条件 × 3 seed = 33 条轨迹，全部 `success=True` 且 phase 集 `[-1,0,1,2,3,4,5,6]` 完整。`usage="train"`。

## 辨识与冻结

`P_sim(s)` = Pdiag-finite（SE(3)）生成器相关度剖面 α(s)，有限变换
`C0 · Exp(diag(α(s)) · Log(C0⁻¹C)) · C0⁻¹ · X0(s)`。纯 pitch 干预下只有 pitch 通道（index 4）可辨识，其余五列名义 ~0——真机恰只有倾斜干预，故够用。

三个 seed 各自独立拟合，`α_pitch(s)` 剖面一致：

| seed | align 均值 | enter | unlock | insert | terminal |
|---|---:|---:|---:|---:|---:|
| 20260910 | 0.654 | 1.012 | 0.999 | 0.999 | 1.002 |
| 20270910 | 0.659 | 1.011 | 1.003 | 1.005 | 1.008 |
| 20280910 | 0.648 | 1.009 | 1.000 | 1.002 | 1.005 |

形状：align 阶段内 0→1 上升（peg 由竖直抬升态转到倾斜 socket 位姿，均值 ~0.65），enter/unlock/insert 恒 ≈1（peg 已对齐孔），起点 ≈0（尚未受孔干预）。这是「peg 最终必须对齐倾斜孔」的响应律——**此形状目前只有仿真证据**，尚未与真机瞬态比较。

## 预测 + 留出插值校验 + 真机幅度对照

用冻结律对 `[0,0,0,0,±10°,0]` 做有限变换预测，读 phase-6 终端 peg 轴响应幅值。

| 模型 | 预测终端响应 |
|---|---:|
| **冻结律 α_pitch(s)（+10°）** | **10.018°**（3 seed，sd 0.032°）|
| baseline α≡0（无迁移）| 0° |
| baseline α≡1（瞬时恒等）| 10.000° |
| **真机手爪轴（观测，仅终端幅值）** | **10.300°**（bootstrap 95% CI [9.60, 11.05]°）|

- 冻结律 − 真机 = **−0.28°**；冻结律 − 恒等 = +0.02°。

**留出插值校验**（关键）：±10° 在训练范围 [−15, +15] **之内**，是**留出幅度插值**而非外推。将冻结律预测与**独立采集的 sim_10degree 纯 pitch ±10° 轨迹**（早于本 sweep 的另一 manifest，同 env/几何/拾取位）的实际终端 peg 轴响应对比：

| 留出幅度 | 冻结律预测（peg） | 实际（peg，3 seed 均值） | 预测 − 实际 |
|---|---:|---:|---:|
| pitch +10° | 10.018° | 9.863° | **+0.156°** |
| pitch −10° | 10.018° | 9.998° | **+0.020°** |

逐 seed 预测 − 实际在 [−0.220, +0.265]° 内。跨幅度曲线 `response(θ)≈1.0018·|θ|`（0→0.0、±2.5→2.50、±5→5.01、±7.5→7.51、**±10→10.02**、±12.5→12.52、±15→15.03）。

**注意**：这条跨幅度曲线全部来自模型自预测（纯 pitch 输入 × 固定 α(s) 本来就近似线性）；**曲线的线性本身不是泛化证据**，上表的「预测 − 实际留出误差」才是。

## 判断与边界

**支持**：冻结 sim 律在 held-out ±10° 上预测 ≈10° 终端 peg 轴响应，与独立采集的留出仿真轨迹相差 +0.02°~+0.16°，与真机手爪轴终端幅值 10.30° 相差 −0.28°（落在真机 bootstrap 区间内）。

**不宣称**：
- **这不是真机几何下的轨迹预测**。预测在**仿真名义几何**（冻结的 `nominal_frame_pose`/`nominal_curve`）内进行；真机数据只贡献终端幅值 10.30°，没有重建真机名义轨迹或孔坐标系（真机倾斜轴未知、无独立孔标定）。所以这是「冻结仿真律预测 vs 真机终端幅值对照」，不是「目标几何下的真机轨迹预测」。
- **±10° 是插值**（在 [−15, 15] 内），不是外推。
- 终端幅值**近退化**——物理上 peg 必须对齐孔（α_pitch→1），冻结律终端预测 ≈ 输入幅度，与恒等 baseline 几乎相同。真正的证据是**留出 ±10° 插值误差**与 **α_pitch(s) 瞬态形状**，而非在终端 0.3° 上「击败 identity」。
- 真机倾斜轴未知 → 只比**幅度**，方向无关，不挑选最接近的模拟方向。
- 冻结律预测 **peg** 位姿，真机指标是**手爪**轴。sim_10degree 里手爪—圆柱末段夹角约 0.11–2.15°，但那是**仿真**测量，**不能当作真机手爪—圆柱代理误差的已知范围**（真机该变换未标定）。
- α_pitch(s) 瞬态形状目前**只有仿真证据**，真机瞬态未比较（真机 `grasp_progress_curves.csv` 是抓取相对时间、非 solver_phase，只能定性，且未做）。
- 未设独立标定/装配容差下的等效界限 → 报差值 + 区间，不作 pass/fail。

## 产物

- `contexts.json` / `protocol.json`：冻结 sweep manifest（`usage="train"`，±10° hold-out）+ sha256 预注册。
- `circular_seed_{20260910,20270910,20280910}.h5` / `.json`：33 条正式采集。
- `frozen_law.npz` / `frozen_law.json`：冻结律（每 seed 的 nominal frame、nominal curve、α(s)、参数）+ 来源 sha256。
- `prediction_10degree.csv` / `.json` / `.png` / `.pdf` / `.svg`：预测表、留出校验、跨幅度曲线与 α_pitch(s) 图。
- `heldout_10degree.csv`：逐 seed 留出 ±10° 预测 vs 实际 vs 误差。
- `smoke_seed_20260910.h5`：冒烟产物（非正式）。
- `videos/pitch_{plus,minus}_{0,10,15}deg.mp4` + `*_final.png`：与真机倾斜视频平行的仿真可视化（`render_tilt_videos.py` 回放同一 `solve_se3`，渲染相机 512×512；仅供展示，非冻结协议一部分）。

## 复现

```bash
# 1. 生成冻结 manifest（maniskill_download，-s 用于避免 user-site 冲突）
/home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
  phase_switch_symmetry/generate_tilt_sweep.py

# 2. 三 seed 采集（33 条）
for seed in 20260910 20270910 20280910; do
  /home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
    phase_switch_symmetry/collect_se3_rollouts.py \
    --output real_relation_consistency/tilt_sweep/circular_seed_${seed}.h5 \
    --context-manifest real_relation_consistency/tilt_sweep/contexts.json \
    --experiment real_relation_consistency/tilt_sweep/protocol.json \
    --seed ${seed} --yaw-mode honest --retries-per-condition 1 \
    --robot-init-qpos-noise 0.01
done

# 3. 辨识 + 冻结（去掉 -s：需 user-site 的 pandas/sklearn）
/home/rocos/miniconda3/envs/maniskill_download/bin/python \
  phase_switch_symmetry/identify_pitch_law.py

# 4. 预测 + 留出校验 + 对照（去掉 -s）
/home/rocos/miniconda3/envs/maniskill_download/bin/python \
  real_relation_consistency/predict_10degree.py

# 5. 渲染仿真倾斜视频（可视化，非冻结协议）
/home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
  phase_switch_symmetry/render_tilt_videos.py \
  --amplitudes 0.0 10.0 -10.0 15.0 -15.0 \
  --out-dir real_relation_consistency/tilt_sweep/videos
```

输出 HDF5 为写模式，复现前先复制现有产物或另选目录，避免覆盖。
