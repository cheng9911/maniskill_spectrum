# 实验整理记录（Experiments Record）

> 本文档是对论文《Task Frames Are Too Coarse: Learning Phase-Dependent Generator
> Relevance for Geometric Skill Transfer》**实验部分的完整整理记录**，覆盖论文正文
> （`paper_latex/main.tex`）以及仓库中所有补充实验（SE(3) 六生成元、basis ablation、
> 多生成元 multigen、planar push、LIBERO 10 任务、跨任务迁移 geometry transfer）。
>
> 用途：写作实验章节时逐条对照的"证据→脚本→数字"索引。每个实验都记录：**研究问题、
> 涉及的脚本与命令、输入数据、输出文件、关键数字、映射到的论文表格/图**。
>
> 所有数字均来自冻结的 summary JSON / CSV / VALIDATION 文档，非现场估算。涉及脚本的
> 具体行号可在 `phase_switch_symmetry/` 下按文件名查找。

---

## 目录

- [0. 复现环境与版本](#0-复现环境与版本)
- [1. 实验总览（真实清单，21 项）](#1-实验总览真实清单21-项)
- [2. 核心任务与模型定义（所有实验共享）](#2-核心任务与模型定义)
- [3. 完整流水线顺序](#3-完整流水线顺序)
- [4. 论文正文实验（SE(2) 核心）](#4-论文正文实验se2-核心)
  - [4.1 E1 单种子物理验证](#41-e1-单种子物理验证)
  - [4.2 E2 强基线基准（strong-baseline benchmark）](#42-e2-强基线基准)
  - [4.3 E3 五种子固定上下文复制](#43-e3-五种子固定上下文复制)
  - [4.4 E4 few-shot 与可辨识性](#44-e4-few-shot-与可辨识性)
  - [4.5 E5 非 oracle 进度坐标](#45-e5-非-oracle-进度坐标)
  - [4.6 E6 旋转轴不变性](#46-e6-旋转轴不变性)
  - [4.7 E7 对称干预（symmetry transfer）](#47-e7-对称干预symmetry-transfer)
  - [4.8 全相位重分析（#11）](#48-全相位重分析11)
- [5. 补充实验（supplements）](#5-补充实验supplements)
  - [5.1 S1 SE(3) 六生成元](#51-s1-se3-六生成元)
  - [5.2 S2 Basis / 坐标消融](#52-s2-basis--坐标消融)
  - [5.3 S3 多生成元 multigen（du + yaw）](#53-s3-多生成元-multigendu--yaw)
  - [5.4 S4 Planar Push（无旋转对称任务）](#54-s4-planar-push无旋转对称任务)
  - [5.5 S5 LIBERO 抽屉探针](#55-s5-libero-抽屉探针)
  - [5.6 S6 LIBERO 10 任务关系套件](#56-s6-libero-10-任务关系套件)
  - [5.7 S7 跨任务迁移（geometry transfer）](#57-s7-跨任务迁移geometry-transfer)
- [6. 论文声明 ↔ 代码 ↔ 数字 对照总表](#6-论文声明--代码--数字-对照总表)
- [7. 已知边界与诚实性声明](#7-已知边界与诚实性声明)
- [8. 仓库内其它任务线（非本论文，仅溯源）](#8-仓库内其它任务线非本论文仅溯源)

---

## 0. 复现环境与版本

| 项 | 值 |
|---|---|
| conda 环境 | `maniskill_download` |
| ManiSkill | 3.0.1 |
| SAPIEN | 3.0.3 |
| 物理后端 | PhysX（`physx_cpu`） |
| 机械臂 / 控制 | Panda manipulator，`pd_joint_pos`，动作经 `env.step(action)` 施加 |
| Python 运行方式 | `conda run -n maniskill_download python -u -B ...` |
| 随机种子 | 五种子固定上下文：`20260818, 20270818, 20280818, 20290818, 20300818` |
| 迁移 split seed | `20260831`（geometry transfer） |
| 已知坑 | `ImportError CXXABI_1.3.15`：在 `import torch/sapien` **之前** `import sqlite3`（conda 的 sqlite/libstdc++ 顺序问题）。所有 env 脚本开头均保留该行。 |
| 相机渲染 | 需传 `human_render_camera_configs` 显式设置分辨率，否则默认分辨率出错。 |

**运行约定**（来自 `phase_switch_symmetry/README.md`）：

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/<script>.py [args]
```

所有 benchmark/analyze 脚本都通过 `--output-root` / `--experiment` / `--subsets` 指定产物路径，
默认产物根目录为 `phase_switch_symmetry_multiseed/`。每个结果 JSON 顶部都写入了
`experiment_sha256` / `subset_manifest_sha256` / `source_sha256`，用于冻结追溯。

---

## 1. 实验总览（真实清单，21 项，与仓库产物完全对齐）

> 以下按**实际结果文件**枚举，不看 `main.tex`。A–E 为分组，`#` 为全局序号。
> "论文映射"只对已写入 `main.tex` 的实验标注；其余是补充实验（有冻结 JSON + `VALIDATION_*.md`，
> 尚未进正文）。

### A. 单种子核心（SE(2) keyed 插入，冻结 v2）
| # | 实验 | 研究问题（一句话） | 主要脚本（`phase_switch_symmetry/`） | 真实结果文件 |
|---|---|---|---|---|
| 1 | 单种子物理验证 | yaw 过键前相关、过键后被抑制？ | `collect/analyze_phase_switch_rollouts.py` | `phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5` → `VALIDATION.md`（tab:protocol, tab:physics） |
| 2 | 独立几何反事实探针 | 几何独立于 planner 目标也支持该律？ | `collect/validate_phase_switch_geometry_probes.py` | `keyed_circular_geometry_probes_v2.h5` |
| 3 | 强基线基准（8 模型） | 其它模型能否表达各向异性？ | `benchmark_phase_switch_baselines.py` | `phase_switch_symmetry_baselines/`（tab:baseline, fig:baseline） |

### B. 五种子复制 + follow-up（复用 `rollouts/seed_*.h5`）
| # | 实验 | 研究问题 | 主要脚本 | 真实结果文件 |
|---|---|---|---|---|
| 4 | 五种子固定上下文复制 | 规律跨种子稳定？ | `prepare/collect/analyze/validate_phase_switch_multiseed.py` | `results/multiseed_summary.json`（tab:multiseed） |
| 5 | few-shot 扫描（N=3…30） | 少样本下结构化先验的价值 | `prepare/run/audit/analyze_phase_switch_fewshot.py` | `fewshot_results/fewshot_summary.json`（tab:fewshot_n3, fig:fewshot） |
| 6 | 可辨识性（负结果） | 少样本子集质量能否被条件数/曲率谱预测？（不能） | `collect/analyze_model_aware_identifiability.py` | `identifiability/identifiability_summary.json` |
| 7 | 非 oracle 进度坐标 | 规律依赖语义相位标签？ | `analyze_nonoracle_progress.py` | `nonoracle_progress/nonoracle_progress_summary.json`（tab:nonoracle） |
| 8 | TP-GMM 匹配 few-shot（N=5,10,20,30） | 与旋转感知 TP-GMM SE(2) 的匹配对比 | `run_phase_switch_tpgmm_fewshot.py` | `tpgmm_fewshot/tpgmm_fewshot_summary.json`（tab:fewshot_tpgmm） |
| 9 | TP-GMM few-shot N=3 扩展 | 最缺样本 N=3 下 TP-GMM SE(2) | `run_phase_switch_tpgmm_fewshot_n3.py` | `tpgmm_fewshot_n3/tpgmm_fewshot_n3_summary.json` |
| 10 | 旋转轴不变性 Q1/Q2 | 世界 yaw 启发式 or 任务局部律？ | `collect/analyze_phase_switch_rotated.py` | `phase_switch_symmetry_rollouts_rotated/`（`rotated_profiles.npz`）→ `VALIDATION_rotated_axis.md`（tab:rotated） |
| 11 | 全相位重分析（7 相位） | yaw 真是 0→1→0？ | `analyze_full_phase_profile.py` | `full_phase_reanalysis/full_phase_verdict.json` |

### C. 对称干预（circular honest / placebo）
| # | 实验 | 研究问题 | 主要脚本 | 真实结果文件 |
|---|---|---|---|---|
| 12 | 对称干预 Step0/1/2/3 | 规律 vs 轨迹统计量（关键负对照） | `benchmark_symmetry_transfer.py` | `symmetry_transfer/symmetry_transfer_summary.json`（tab:sym_m1/gap/var, fig:symmetry_transfer, fig:sym_var） |
| 13 | 对称干预 N=3 TP-GMM SE(2) 臂 | N=3 下 TP-GMM SE(2) 的 M1/M2/M2b/M3 | `extend_symmetry_transfer_N3_tpgmm_se2.py` | `symmetry_transfer/symmetry_transfer_N3_tpgmm_se2_summary.json` |

### D. SE(3) 六生成元
| # | 实验 | 研究问题 | 主要脚本 | 真实结果文件 |
|---|---|---|---|---|
| 14 | SE(3) 六生成元 | 6 生成元里锁定唯一选择性生成元 | `collect_se3_rollouts.py`, `benchmark_se3_transfer.py` | `se3_transfer/se3_transfer_summary.json` |
| 15 | basis / 坐标消融 | 对角先验是任务框架的性质？ | `benchmark_se3_basis_ablation.py` | `se3_transfer_basis_ablation/basis_ablation_summary.json` |
| 16 | multigen（du+yaw） | 两个选择性生成元一起恢复 | `collect_se3_multigen_rollouts.py`, `benchmark_se3_multigen.py` | `se3_multigen/multigen_summary.json` |

### E. 其它物理任务 / LIBERO / 跨任务迁移
| # | 实验 | 研究问题 | 主要脚本 | 真实结果文件 |
|---|---|---|---|---|
| 17 | Planar push | 无旋转对称任务反驳"定制模型" | `collect_planar_push_rollouts.py`, `benchmark_planar_push.py` | `planar_push/planar_push_summary.json` |
| 18 | LIBERO 抽屉探针 | 真实平移任务的单轴生成元（+TP-GMM projected 基线） | `collect/benchmark_libero_drawer_probe.py` | `libero_drawer/libero_drawer_summary.json` |
| 19 | LIBERO 10 任务关系套件 | 跨 10 任务关系识别（+TP-GMM projected 基线） | `collect/benchmark_libero_relation_suite.py` | `libero_relation_suite/libero_relation_suite_summary.json` |
| 20 | 跨任务迁移（5 对） | 冻结源 Pdiag 迁移 | `benchmark_geometry_transfer.py` | `geometry_transfer/geometry_transfer_validation.json` |
| 21 | geometry transfer TP-GMM N=8 变体 | N=8 加 TP-GMM SE(3) 对照 | `benchmark_geometry_transfer.py`（tpgmm 变体） | `geometry_transfer_tpgmm_n8/geometry_transfer_validation.json` |

> **诚实性标注**：论文正文（`main.tex` L454–989）只覆盖 #1–#7、#10、#12 的 **SE(2) 部分**；
> #8/#9/#11/#13 以及全部 D/E 组（#14–#21）是补充实验，有冻结结果但尚未写入正文。
> 其中 LIBERO 数据（#18/#19）是**合成探针**（见 §7），geometry transfer（#20/#21）迁移预测是
> **线性近似**（见 §7）。

**全局 # ↔ 章节编号对照**（正文细节节沿用旧 E/S 标签，此处统一）：
§4.1=#1、§4.2=#3、§4.3=#4、§4.4=#5+#6、§4.5=#7、§4.6=#10、§4.7=#12、§4.8=#11；
§5.1=#14、§5.2=#15、§5.3=#16、§5.4=#17、§5.5=#18、§5.6=#19、§5.7=#20。
#2（几何探针）、#8/#9（TP-GMM few-shot）、#13（N=3 对称臂）、#21（N=8 迁移变体）已并入相邻小节。

---

## 2. 核心任务与模型定义

> 这是所有实验共享的"语言"，写实验节时放在 "Keyed-to-Circular Phase-Switch Task"
> 与 "Models and Metrics" 两小节。

### 2.1 任务（KeyedCircularPhaseSwitch-v1）

复合销钉：短矩形 **key** + 圆 **shaft**。键控门（keyed gate）使轴向 yaw 在
对齐/进入阶段相关；key 通过门后，圆 shaft + 圆孔允许轴向旋转（yaw 成为规范对称）。
四个语义相位（论文写作用）：
`align_keyed` → `enter_key` → `unlock_yaw` → `circular_insert`。

- 几何常量：`SHAFT_RADIUS=0.013`、`KEY_HALF_X=0.021`、`KEY_HALF_Y=0.014`、
  `SLOT_HALF_X=0.044`、`SLOT_HALF_Y=0.026`（multigen 用）。
- 干预向量（task-local）：SE(2) `c = [Δx, Δy, Δψ]^T`；SE(3) 扩展为
  `[du, dv, dw, droll, dpitch, dyaw]`。

### 2.2 上下文 c(T) 与生成元基 B

- SE(2)：`_context_twist = se2_log(nominal_frame⁻¹ @ socket_frame(context)) ≈ [du, dv, dψ]`
  （精确）。
- SE(3)：`_context_twist = se3_log(se3_from_pose6(context))`（旋转取 axis-angle log，
  欧拉角仅一阶近似）。
- 生成元基 **B = 单位阵**（context 坐标即生成元坐标）。
- 归一化缩放 `CONTEXT_SCALE = [0.012, 0.012, deg2rad(30)]`（SE(3) 再加角度项），
  用于"可辨识性/qualified"的列缩放条件数判定。

### 2.3 生成元相关性模型 P(s)

- `P(s) = diag(α_1(s), …, α_d(s))`，作用于 `twists * alpha`。
- 每个 α 是**归一化高斯 RBF**（默认 `n_basis=24`、`basis_width=0.065`）+ logistic
  sigmoid：`α = α_max / (1 + exp(-clip(basis @ θ)))`，`α_max=1.25`，加二阶差分平滑。
- LIBERO **10 任务关系套件（#19）**用 `n_basis=8`、`basis_width=0.12`、`smoothness_weight=0.1`、
  `nominal_iterations=1`（更粗，因数据合成；这是运行时 CLI 覆盖，冻结在
  `libero_relation_suite_summary.json` 的 `pdiag_config`，代码默认仍是 24/0.065）。
  LIBERO **抽屉探针（#18）**用默认 `n_basis=24`、`basis_width=0.065`、
  `nominal_iterations=3`（与 SE(2) 主任务一致）。

### 2.4 有限几何实现（finite realization）

```
X̂ = C0 · Exp( P(s) · c ) · C0⁻¹ · X0(s)
```

其中 `C0`/`X0` 是逐进度的名义曲线/名义 socket 框架（`nominal_iterations=3` 迭代精化，
只从 mixed 训练轨迹估计，zero 干预不参与拟合）。

### 2.5 进度网格 progress_grid(bins=25)

`progress = (phase_index + phase_progress) / 4`；数据里 `PHASE_CODES=(3,4,5,6)` 对应
align_keyed / enter_key / unlock_yaw / circular_insert（内部 phase_codes 0–3）。
非 oracle 实验里用 normalized time / SE(3) arc length / 关键点下降 / DTW 对齐替代。

### 2.6 识别机制（模型如何学到"哪个生成元在哪个相位起作用"）

- **反事实敏感性**：`DiagonalOperatorModel.fit` 对每个进度、每个生成元做最小二乘斜率
  `∂X_j / ∂c_j`，得到逐生成元经验响应对角；初始化 RBF-sigmoid 后，用
  `least_squares(method="trf")` 在有限模型预测误差 + 平滑项上精化。
- **oracle 只用于评估，不用于拟合**。oracle 是"按相位手工标注的 α(s) 理想序列"
  （如 yaw = [1,1,0,0]），只在算 `E_α`（生成元律误差）时与拟合结果对比。

### 2.7 激励 / qualified 判定

- 合格子集（qualified）：列缩放上下文的秩 = 3 且 `condition_number < 10.0`。
- 随机子集（random）从不丢弃。

### 2.8 度量（论文 §Method "Symmetry-Aware Evaluation"）

- **Generator-law error** `E_gen`（= E_α）：拟合 `α_j(s)` 与 oracle 的均方误差（不做商投影）。
- **Quotient task-space error** `E_task`：任务误差，仅在物理过键事件之后对轴向 yaw 做商
  （即忽略过键后的 yaw 误差），保留每个相位的平移误差。
- 其它：`g_trans`（最终平移相关性）、`g_ψ^pre`（过键前 yaw 相关性）、
  `g_ψ^final`（最终 yaw 相关性）、switch 检测、law 恢复率、M0–M4（见 E7 / S1）。

---

## 3. 完整流水线顺序

每个实验统一走 **collect → prepare → benchmark → analyze → validate → make figure**
六步。下面以 SE(2) 核心为例给出完整顺序（补充实验同构，脚本见第 5 节）：

```
1. collect     收集 rollout H5（含失败尝试、condition_id、attempt_id、seed、stop_reason、源哈希）
2. prepare     冻结 subset manifest（写 *_subsets.json / *_experiment.json，含 sha256）
3. benchmark   对每个 (seed/subset/模型) 拟合并打分，输出 fits CSV + summary JSON
4. analyze     聚合出论文表格数字（mean/std、配对差、切换率、律恢复率）
5. validate    严格校验（strict 阈值、覆盖率、尝试连续、train/test 不相交、零 post-terminal 动作）
6. make        出图 PNG/PDF（论文 figure）
```

关键产物清单（按实验）见各小节"输出"段。所有 summary JSON 均含 `schema_version`、
`fit_count`、`fit_failure_count` 和各类 sha256 追溯字段。

---

## 4. 论文正文实验（SE(2) 核心）

### 4.1 E1 单种子物理验证

**研究问题**：任务物理本身是否满足"平移始终相关、yaw 仅在过键前相关"的对称切换？
（这是所有模型比较之前的第一步机制验证。）

**脚本 / 命令**（`phase_switch_symmetry/README.md`）：

```bash
# 收集（39 条件：30 mixed + 8 非零 isolated + 1 zero）
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_rollouts.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --mixed-samples 30 --retries-per-condition 3

# 严格校验 + 分析
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_rollouts.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 --strict

# 独立几何反事实（不依赖 planner 目标）
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_geometry_probes.py \
  --output phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_geometry_probes.py \
  phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5 --strict
```

**输入**：`phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5`；
`keyed_circular_geometry_probes_v2.h5`。

**输出**：`phase_switch_symmetry/VALIDATION.md`（冻结解释）；分析脚本 stdout 的严格校验表。

**数据集统计**（tab:protocol 第一行）：
- 干预条件 39；存储 episode 49；成功完整 39；用于拟合的 mixed 30；
- 保留失败尝试 10；非零 peg-socket 接触 3 段（2 成功）；socket 漂移 < 1e-6 m；
- post-terminal / post-truncation 动作 0。

**关键数字（tab:physics + VALIDATION.md）**：

| 校验项 | 判据 | 结果 |
|---|---:|---:|
| 最终平移相关 | ratio ≥ 0.8 | **0.9872** |
| 过键前 yaw 相关 | ratio ≥ 0.8 | **0.9997** |
| unlock 后 yaw 抑制 | ratio ≤ 0.25 | **0.0028** |
| 最终 yaw 抑制 | ratio ≤ 0.25 | **0.0022** |
| scalar → Pdiag 端点误差 | 下降 | **6.554 → 0.639 mm**（90.2% 下降） |
| 配对改进 95% CI | > 0 | **[4.711, 7.202] mm-equiv** |
| 可辨识性（mixed 激励） | rank = 3 | 通过 |

- 学习到的最终响应近似 `diag(1.0024, 1.0000, -0.00031)`；scalar 最优帧权 `w = 0.5681`
  （这是 Eq. best_scalar 所预言的折中：无法同时"平移≈1 且最终 yaw≈0"）。

**几何反事实（独立于 planner 目标）**：
- 匹配 ±30° 进入：过门 +8.0 / +6.4 mm；
- 失配进入：卡在 ≈ -28 mm clearance，产生 48.1 / 49.7 N 接触力；
- 过门后 -30°/+30° 均成功插入，终点 -30.00° / +30.02°；
- 6 项几何检查全部通过，零 post-terminal 动作。

### 4.2 E2 强基线基准

**研究问题**：各向异性行为是否其它模型也能表达？（Pdiag 的定位不是"最低误差轨迹预测器"，
而是"用受限、可解释的有限作用算子恢复任务一致的生成元律"。）

**脚本 / 命令**：

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/benchmark_phase_switch_baselines.py \
  phase_switch_symmetry_rollouts/keyed_circular_phase_switch_physics_v2.h5 \
  --output-root phase_switch_symmetry_baselines

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_baselines.py \
  phase_switch_symmetry_baselines --strict
```

**模型（8 个，MODEL_ORDER）**：Frame-weighted（标量帧权 `w(s)I`）、Phase scalar GP
（oracle 友好的相位标量 GP）、TP-GMM additive、TP-GMM SE(2)（旋转感知）、Generic RBF、
Full operator（稠密）、Pdiag pointwise（逐点对角消融）、Pdiag finite（本方法）。

**输出**：`phase_switch_symmetry_baselines/`（含 `BASELINE_VALIDATION.md`、
`phase_switch_strong_baselines.png/pdf`）。

**关键数字（tab:baseline，mean mm-equiv）**：

| 模型 | Task traj. | Endpoint | E_gen | 最终平移 | 过键前 yaw | 最终 yaw |
|---|---:|---:|---:|---:|---:|---:|
| Frame-weighted | 4.765 | 3.262 | 0.354 | 0.568 | 1.004 | 0.568 |
| Phase scalar GP | 4.764 | 3.261 | 0.354 | 0.568 | 1.004 | 0.568 |
| TP-GMM SE(2) | 4.159 | 0.298 | 0.177 | 0.998 | 0.946 | 0.0018 |
| Generic RBF | **3.316** | **0.251** | 0.196 | 1.001 | 1.000 | -0.0003 |
| Full operator | 3.304 | **0.251** | 0.196 | 1.001 | 1.000 | -0.0003 |
| **Pdiag finite** | 3.448 | 0.427 | 0.192 | 0.985 | 1.000 | **0.0258** |

**解读**：标量帧相关 / 相位标量 GP 只能保留单一帧权，因此无法"平移≈1 且最终 yaw≈0"；
TP-GMM 与稠密/RBF 能恢复切换（说明各向异性不是本方法独有）。Pdiag finite 的定位是
**恢复任务一致的生成元律**，而非最低轨迹误差。

### 4.3 E3 五种子固定上下文复制

**研究问题**：切换规律在独立模拟器/planner 种子上是否可复现？

**脚本 / 命令**：

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/prepare_phase_switch_multiseed.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/collect_phase_switch_multiseed.py --skip-complete
# 每个 seed_<seed>.h5 用 analyze_phase_switch_rollouts.py --strict 校验，
# 再分别传给 benchmark_phase_switch_baselines.py 独立拟合
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_multiseed.py
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_multiseed.py
```

**输入**：`phase_switch_symmetry_multiseed/rollouts/seed_<seed>.h5`（5 个种子）。
**输出**：`phase_switch_symmetry_multiseed/results/multiseed_summary.json` +
`multiseed_seed_summary.csv` + `phase_switch_multiseed.png/pdf`；
`phase_switch_symmetry_multiseed/VALIDATION.md`。

**关键数字（tab:multiseed）**：

| 种子 | 最终平移 | 过键前 yaw | 最终 yaw | E_task |
|---|---:|---:|---:|---:|
| 20260818 | 0.9847 | 0.9983 | 0.0234 | 2.1360 |
| 20270818 | 0.9972 | 0.9966 | 0.0273 | 2.1908 |
| 20280818 | 0.9875 | 0.9985 | 0.0250 | 3.4594 |
| 20290818 | 0.9821 | 0.9959 | 0.0251 | 2.5222 |
| 20300818 | 0.9797 | 0.9999 | 0.0283 | 1.6195 |

- scalar−Pdiag 任务误差改进 **1.7576 ± 0.4653 mm**（种子间），
  种子后条件 bootstrap 区间 **[1.2964, 2.2295]**。
- 汇总 g_translation_final ≈ 0.986、g_yaw_preclear ≈ 0.998、g_yaw_final ≈ 0.0258。
- tab:protocol 第二行：5×39 = 195 条件，全部 strict pass。

### 4.4 E4 few-shot 与可辨识性

**研究问题**：结构化对角有限作用先验在少样本（尤其 N=3）下是否带来样本效率优势？

**脚本 / 命令**：

```bash
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/prepare_phase_switch_fewshot.py

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/run_phase_switch_fewshot.py --seed 20260818   # 对每个种子重复
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/audit_phase_switch_fewshot_pdiag.py --seed 20260818

conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/analyze_phase_switch_fewshot.py
conda run -n maniskill_download python -u -B \
  phase_switch_symmetry/validate_phase_switch_fewshot.py
```

**输入**：`fewshot_subsets.json`（冻结，N∈{3,5,8,10,15,20,30}，每个协议 10 个冻结子集）。
**输出**：`fewshot_results/fewshot_summary.json` + 若干 CSV + `phase_switch_fewshot.png/pdf`；
可辨识性在 `identifiability/identifiability_summary.json`。

**关键数字**：
- 拟合 1815，失败 0（tab:protocol：Few-shot sweep 1815 fits / no drops）。
- **N=3（tab:fewshot_n3，mean over seed means ± between-seed std）**：

| 协议 | 模型 | E_task | E_end | E_gen | g_trans | g_pre | g_final | switch/law |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Random | **Pdiag finite** | **3.27±0.82** | 1.09±0.07 | 0.264±0.128 | 0.997 | 1.000 | 0.015 | 100% / 88% |
| Random | TP-GMM SE(2) | 10.48±0.56 | 1.22±0.37 | 0.363±0.068 | 0.920 | 0.358 | 0.356 | 0% / 0% |
| Random | Full operator | 9.50±1.17 | 6.27±0.16 | 0.670±0.194 | 0.649 | 0.713 | 0.001 | 70% / 0% |
| Random | Generic RBF | 24.12±0.96 | 23.28±0.04 | 0.907±0.091 | 0.817 | 0.380 | 0.000 | 20% / 10% |
| Qualified | **Pdiag finite** | **3.33±0.72** | 1.33±0.09 | 0.264±0.159 | 0.981 | 0.999 | 0.020 | 100% / 92% |
| Qualified | TP-GMM SE(2) | 10.09±0.65 | 0.67±0.20 | 0.354±0.074 | 0.961 | 0.356 | 0.360 | 0% / 0% |
| Qualified | Full operator | 9.14±1.20 | 6.37±0.05 | 0.568±0.155 | 0.638 | 0.726 | 0.000 | 80% / 0% |
| Qualified | Generic RBF | 28.47±1.48 | 28.04±1.01 | 0.963±0.066 | 0.382 | 0.444 | 0.000 | 30% / 0% |

- 律恢复率（law pass rate）：random N=3 **88%**，qualified N=3 **92%**，两者 N≥5 **100%**。
- 配对任务误差差（competitor − Pdiag）：random 下 Full +6.23 mm、RBF +20.86 mm；
  qualified 下 Full +5.81 mm、RBF +25.14 mm；四种比较 Pdiag 胜率均 50/50（跨 cell）。

**#8 matched TP-GMM 少样本（tab:fewshot_tpgmm，任务轨迹误差均值）**：

| N | Pdiag | TP-GMM SE(2) | Full op. | RBF |
|---|---:|---:|---:|---:|
| 5 | **2.71** | 7.58 | 4.11 | 5.91 |
| 10 | **2.67** | 4.43 | 3.94 | 5.27 |
| 20 | **2.43** | 3.78 | 3.28 | 4.02 |
| 30 | **2.39** | 3.46 | 3.07 | 3.63 |

- 脚本 `run_phase_switch_tpgmm_fewshot.py`；产物 `tpgmm_fewshot/tpgmm_fewshot_summary.json`。
- **Pdiag 在每个 matched seed/subset cell 每个 N 都赢**：`pdiag_win_fraction = 1.0`
  （N=5/10/20/30）；配对差（TP-GMM − Pdiag 任务误差）`mean_diff` = 4.87 / 1.76 / 1.35 / 1.07 mm，
  随 N 增大收窄（灵活算子逼近同一响应），与"对角先验主要提升样本效率、而非替代数据"一致。
- 80 fits（tab:protocol）。

**#9 TP-GMM few-shot N=3 扩展**（`tpgmm_fewshot_n3/tpgmm_fewshot_n3_summary.json`）：
- 脚本 `run_phase_switch_tpgmm_fewshot_n3.py`；模型 TP-GMM SE(2)；N=3，20 个子集 × 5 种子
  = 100 fits，0 失败。
- TP-GMM SE(2) 的 N=3 数字（对应 tab:fewshot_n3 的 TP-GMM SE(2) 行）：

| 协议 | E_task | E_end | E_gen | g_trans | g_pre | g_final | switch/law |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 10.48±0.56 | 1.22±0.37 | 0.363±0.068 | 0.920 | 0.358 | 0.356 | 0% / 0% |
| Qualified | 10.09±0.65 | 0.67±0.20 | 0.354±0.074 | 0.961 | 0.356 | 0.360 | 0% / 0% |

- N=3 下 TP-GMM SE(2) 组件候选被 cap 到 (2,)（`expanded-K` capped at N−1），CV splits = 3。

**#6 可辨识性（负结果）**（`identifiability/identifiability_summary.json`）：
- 脚本 `collect_model_aware_identifiability.py` / `analyze_model_aware_identifiability.py`；
  121 个子集 × 5 种子；模型 Pdiag finite。
- 问题：能否**先验地**用（a）上下文条件数 κ(C) 或（b）局部参数空间 Gauss-Newton 谱统计
  （λ_min、λ_ratio、log det、trace-inverse，在拟合 θ 与中性 θ=0 参考处）预测哪个 N=3 子集质量高？
- 答案：**不能**。所有带符号 Spearman ρ 都弱（within-cell |ρ| < 0.4，实际 0.003–0.17）且跨样本量
  符号不一致。均值 ρ：condition_number **0.171**、trace_inv_post 0.136、λ_min_post 0.126、
  λ_ratio_design 0.121、λ_ratio_post 0.106、log_det_design/trace_inv_design 0.105、
  λ_min_design 0.054、log_det_post 0.003。
- 边界：这是关于**这些特定谱统计量**的窄负结果，**不排除**生成元律本身的函数空间可辨识性
  （如 α_yaw^post 的方差或 s_0.5）。

### 4.5 E5 非 oracle 进度坐标

**研究问题**：上面用的相位标签是 solver 的语义标签（privileged）。换用可观测进度坐标，
规律是否还在？

**脚本**：`phase_switch_symmetry/analyze_nonoracle_progress.py`（复用 `rollouts/seed_*.h5`，
无新物理数据）。

**输出**：`nonoracle_progress/nonoracle_progress_summary.json` + `nonoracle_progress_figure.png/pdf`。

**关键数字（tab:nonoracle，五种子平均，dE 相对 oracle 语义相位参考）**：

| 进度 | E_task^phys | dE | E_gen |
|---|---:|---:|---:|
| Oracle phase | 4.60 | 0.00 | 0.218 |
| Time（归一化时间） | 11.11 | +6.50 | 0.745 |
| Arc length（SE(3) 弧长） | 5.50 | +0.90 | 0.328 |
| Keypoint（竖直关键点下降） | 8.89 | +4.29 | 0.349 |
| DTW-position（离线 DTW 对齐） | **3.19** | **-1.42** | **0.130** |

**解读**：归一化时间最差；几何进度显著更好；离线 DTW-position 最强（但它用整条序列对齐，
**不应解释为可部署估计器**，只是上界）。结论：规律不是语义标签的产物。

### 4.6 E6 旋转轴不变性

**研究问题**：模型是否只是学了"世界 yaw 启发式"？（把任务绕全局转 90° 再测。）

**设计**（`VALIDATION_rotated_axis.md`）：
- Q1 = 单位（插入轴 = 世界 z）；Q2 = Ry(90°)（插入轴 = 世界 x）；
  Q3 = Rx(35°)Ry(45°) 因可达性不足在学习前排除。
- 公共锚点 `(-0.35, 0.10, 0.08)`。

**脚本**：`collect_phase_switch_rotated.py` / `analyze_phase_switch_rotated.py` 等 rotated 系列。

**关键数字（tab:rotated，matched Q1/Q2）**：

| 种子 | D_orient | ρ_axial | \|d g_ψ^final\| |
|---|---:|---:|---:|
| 20260818 | 0.112 | 0.985 | 0.015 |
| 20270818 | 0.289 | 0.941 | 0.005 |
| 20280818 | 0.148 | 0.938 | 0.004 |

**解读**：按逆全局任务旋转归一化后，轴向剖面在竖直 Q1 与水平 Q2 间高度一致（ρ_axial
0.94–0.99），支持"任务局部"解释（Q2 的轴向生成元不是世界 yaw）。一个 Q1 种子平移低于
阈值，但所有 matched 种子的轴向律都保持。

### 4.7 E7 对称干预（symmetry transfer）

**研究问题（全文最关键的负对照）**：模型是在发现**任务关系**，还是在记忆**轨迹统计量**？
通过移除 key/keyway 构造 3 个 matched 臂。

**设计**（`symmetry_transfer_experiment.json` + `VALIDATION_symmetry_transfer.md`）：

| 臂 | 环境 | yaw 律 |
|---|---|---|
| keyed | KeyedCircularPhaseSwitch-v1（复用 rotated Q1） | α_ψ^keyed ≈ 1 过键前 → ≈0 过键后 |
| circular_honest | CircularPhaseSwitch-v1（扁平 yaw） | α_ψ ≈ 0（示范者保持 yaw nominal） |
| circular_placebo | CircularPhaseSwitch-v1（固定 15° 对齐扫描） | α_ψ ≈ 0（yaw 运动存在但与 Δψ 无关） |

- 18 个冻结子集：N∈{5,10,20} × {random, qualified} × 3 repeats，跨 3 臂 3 种子共享。
- 执行顺序：Step 0 几何 → Step 1 honest → Step 2 placebo → Step 3 benchmark。

**脚本**：`benchmark_symmetry_transfer.py`（+ Step0 见 `VALIDATION_circular_step0.md`）。

**输出**：`symmetry_transfer/symmetry_transfer_summary.json`（156KB）+ M0/M1/M2/M2b/M3 CSV +
`symmetry_transfer_main.png` + `symmetry_transfer_variance.pdf`。

**关键数字**：
- Step 0：名义圆插入成功；align 相位隔离 yaw 斜率 **0.0000**（Δψ 是规范干预，先于任何拟合）。
- 9 个数据集（3 臂 × 3 种子）全部 39/39 可用条件；benchmark **1296 fits / 0 失败**。

**M1 生成元识别误差（tab:sym_m1，N=20，越低越好）**：

| 模型 | Keyed | Circular honest | Circular placebo |
|---|---:|---:|---:|
| Frame-weighted | 0.338 | 0.262 | 0.284 |
| Phase scalar GP | 0.338 | 0.262 | 0.284 |
| TP-GMM additive | **0.132** | 0.0006 | 0.0005 |
| TP-GMM SE(2) | 0.142 | 0.0006 | **0.0004** |
| Generic RBF | 0.159 | **0.00008** | 0.0006 |
| Full operator | 0.159 | 0.0002 | 0.0006 |
| Pdiag pointwise | 0.158 | 0.0001 | 0.0005 |
| **Pdiag finite** | 0.180 | **0.0016** | **0.0015** |

- 两个 circular 臂上所有关系/结构模型读 α_ψ ≈ 0（Pdiag finite：honest 0.0016、placebo 0.0015）；
  两个边际帧级基线在 circular 臂上"幻觉"出 yaw 相关性（E_α ≈ 0.26–0.28），
  因为标量帧响应无法分离"相关平移"与"无关轴向生成元"。

**M2 判别（keyed vs circular，α_ψ^pre > 0.5 判为 keyed）**：
- Pdiag finite / Full op / pointwise / RBF / TP-GMM SE(2) 在 N≥5 全部 **100%**；
  TP-GMM additive N=5 96.3% → N≥10 100%；
- 帧级基线 N=20 仍 <100%（N=5/10/20 为 70.4% / 92.6% / 96.3%）。

**M0 切换诊断**：Pdiag finite 在 N=5/10/20 全部 100% 检出切换，切换位置（归一化进度）
0.606 / 0.608 / 0.608。

**M2b 安慰剂对称间隙（tab:sym_gap，理想值 1）**：

| 模型 | N=3 | N=5 | N=10 | N=20 |
|---|---:|---:|---:|---:|
| Frame-weighted / Phase scalar GP | — | 0.160 | 0.238 | 0.268 |
| TP-GMM additive | — | 0.627 | 0.782 | 0.790 |
| TP-GMM SE(2) | 0.414 | 0.680 | 0.776 | 0.799 |
| Generic RBF | — | 0.773 | 0.809 | 0.812 |
| Full operator | — | 0.799 | 0.813 | 0.813 |
| Pdiag pointwise | — | **0.814** | **0.815** | **0.813** |
| **Pdiag finite** | — | 0.761 | 0.754 | 0.741 |

- 关系模型间隙大，边际基线≈0.16–0.27；N=3（仅 TP-GMM SE(2)）间隙 0.414 < N=5 的 0.680。
- 间隙 < 1 是因为 keyed 过键前 mean 相关性在整个插入前窗口饱和在 ~0.75–0.81，
  而非恰为 1；Pdiag finite 有意在进度上"铺开"过渡。要点不是最大化这个位置统计量，
  而是**对 honest 与 placebo 圆臂都给出近乎零的 yaw 相关性，尽管 placebo 轨迹含旋转**。

**可靠性分析（tab:sym_var，circular-honest N=5，18 个 seed/subset cell）**：

| 度量 | Full op. | Pdiag finite |
|---|---:|---:|
| Mean E_α | 0.062 | 0.0077 |
| Total std. | 0.253 | 0.011 |
| Seed std. | 0.087 | 0.006 |
| Subset std. | 0.138 | 0.009 |
| Worst case | 1.106 | 0.045 |
| Hallucination | 1/18 | 0/18 |

- 总标准差约低 22 倍、最坏情况约低 25 倍。正确的主张不是"排他准确率"，而是
  **少样本可靠性**：无约束 3×3 算子有肥尾过拟合，对角生成元先验没有。

**M3 跨任务不匹配**：N=20，在 honest 上拟合的 Pdiag finite 评估 placebo 额外误差 +8.89 mm，
反向 +8.46 mm。证明 honest 与 placebo 行为上可区分（两者都 α_ψ=0，但 placebo 含 yaw 运动）。

**#13 对称干预 N=3 TP-GMM SE(2) 臂**（`symmetry_transfer/symmetry_transfer_N3_tpgmm_se2_summary.json`）：
- 脚本 `extend_symmetry_transfer_N3_tpgmm_se2.py`；54 fits，0 失败（18 cell × 3 任务）。
- M1（N=3）：keyed **0.229**、circular_honest 0.0032、circular_placebo 0.0093（对比 N=5 的 keyed 0.142）。
- M2 判别：N=3 掉到 **0.722**（39/54）；对照 Pdiag finite 在 N≥5 仍是 100%。
- M2b 间隙（N=3，即 tab:sym_gap 的 N=3 列）：placebo 臂 G_ψ = **0.414**（honest 臂 0.445）。
- M3 跨任务（N=3 均值）：honest↔placebo ≈ 3.7–4.6 mm、placebo↔keyed ≈ 5.4–6.2 mm。
- 结论：TP-GMM SE(2) 在 N=3 最缺样本时判别从 100% 掉到 72.2%、间隙从 0.680 掉到 0.414，
  与其 few-shot 劣势（tab:fewshot_tpgmm）一致——正是正文"gap 在 N=3 只报 TP-GMM SE(2)"的原因。

---

## 4.8 全相位重分析（#11）

**研究问题**：现有基准只看 4 个插入相位（PHASE_CODES=3,4,5,6）。yaw 到底是不是 0→1→0？
跨**全部 7 个 solver 相位**（reach 0 / grasp 1 / lift 2 / align_keyed 3 / enter_key 4 /
unlock_yaw 5 / circular_insert 6）读经验响应对角。

**脚本**：`analyze_full_phase_profile.py`（复用已拟合模型，无新物理数据）。
**输出**：`full_phase_reanalysis/full_phase_verdict.json` + `full_phase_profile.png/pdf`。

**关键数字（full_phase_verdict.json）**：
- `yaw_is_0_1_0 = true`：yaw 剖面 [≈0, ≈0, ≈0, **1.000**, **1.000**, 0.069, 0.041]（7 相位）——
  在 reach/grasp/lift 为 0，align/enter 为 1，unlock/insert 掉回 ~0.07/0.04。
- `translation_is_0_then_1 = true`、`translation_is_constant_1 = false`：平移剖面
  [≈0, ≈0, 0.004, 0.999, 1.007, 1.008, 1.025]——reach/grasp/lift 是**预接触**（销钉在固定
  底座姿态，与 socket 干预无关），所以平移不是恒 1 而是 0→1。

---

## 5. 补充实验（supplements）

> 以下实验有冻结 summary JSON + `phase_switch_symmetry_multiseed/VALIDATION_*.md`，
> 尚未纳入 `main.tex`。命令与 SE(2) 同构：collect → prepare → benchmark → analyze。

### 5.1 S1 SE(3) 六生成元

**研究问题**：把生成元基从 SE(2) `[du,dv,d_axial]` 推广到 SE(3) `[du,dv,dw,droll,dpitch,dyaw]`，
**6 个生成元中，模型能否锁定 axial-yaw 为唯一选择性生成元**（其它 5 个插入期始终 on）？

**脚本**：`collect_se3_rollouts.py`、`generate_se3_contexts.py`、`prepare_se3_subsets.py`、
`benchmark_se3_transfer.py`。

**输出**：`se3_transfer/se3_transfer_summary.json`；`VALIDATION_se3_six_generator.md`。

**设计要点**：oracle = 6 维逐相位 α(s)；4 个插入相位（align_keyed/enter_key/unlock_yaw/
circular_insert）。拟合 432 / 0 失败；样本 N∈{8,15,30}，每 cell 18（3 种子 × 2 任务 × 3 repeats）。

**关键数字（阈值无关结构指标，论文报告这些而非旧 M4 幅值）**：

| 指标 | 定义 | Pdiag finite（N=8/15/30） |
|---|---|---|
| M1 E_α（keyed） | 生成元识别误差 | 0.105 → 0.102 → **0.095** |
| M1 E_α（circular_honest） | | 0.078 → 0.071 → **0.070** |
| M2 rank | yaw 是 6 个里最大的 clearance drop | **1.000 / 1.000 / 1.000** |
| M3 transition | Δ_yaw > 0.2 | **1.000 / 1.000 / 1.000** |
| purity S_yaw/Σ_j S_j | 选择性质量集中于 yaw | 0.651 / 0.685 / **0.729** |
| M0 switch | 过键切换检出 | 100%（位置 ~0.61） |
| M2b gap G_ψ | keyed − circular | ~0.64–0.67 |

- 结构对比：Pdiag finite `S_yaw ≈ 0.64–0.67` vs `S_others ≤ 0.16`（4–5 倍间隙，随 N 拉大）；
  yaw top-1 100%、Δ_yaw>0.2 100%、purity 0.65→0.73。
- **负对照** Frame-weighted（共享标量 w(s)，无法表达逐生成元选择性）全部失败：
  rank 0.000、transition 0.000、purity 0.167。
- 旧 M4（精确 oracle 幅值、0.5 阈值）= 50/39/11% 是**幅值阈值假象**：真实 yaw 剖面是
  rise-then-fall 凸起 [0.46, 0.95, 0.46, 0.10]，非理想化 [1,1,0,0]（align 相位有真实的
  0→1 上升 transient）。smoothness 0.1 的扫描不改变剖面，拟合正确；错的是 oracle 的理想化。

### 5.2 S2 Basis / 坐标消融

**研究问题**：对角先验是"任务框架（task-local socket frame）的性质"，还是"碰巧手选轴的收益"？

**脚本**：`benchmark_se3_basis_ablation.py`（复用冻结 SE(3) 数据，无新收集）。

**输出**：`se3_transfer_basis_ablation/basis_ablation_summary.json`；`VALIDATION_se3_basis_ablation.md`。

**方法**：把同一批干预用固定任意旋转 R 共轭：`context_rot = pose6(R⁻¹ T(context) R)`，
即对 4×4 干预施加 SE(3) 伴随 Ad(R⁻¹)（含旋转+平移交叉项）。两个旋转
rotated-1 = (30°, -20°, 40°)，rotated-2 = (-35°, 25°, 15°)（刻意非轴置换）。

**关键数字**：
- 任务基：enter 相位经验响应算子**就是对角律**（diag≈1，非对角范数 0.15）；
  旋转基：非对角范数 2.77，对角 = 重标度律 `α_true · diag(R)`，幅值误读最多 28%。
- 预测误差 e_data 升 **2.2–13×**，恢复律误差升 **2.1–4.1×**（每个 N、两个旋转都如此）。
- 不变性对照：Full operator（把旋转吸收进更多非对角质量 0.89→0.93）与 Frame-weighted
  共享标量（1.0–1.15×）是 basis-free。标量族可证明 Ad-不变
  （`C0' Exp(w t') C0'⁻¹ X0'`，`C0'=C0 R`、`t'=Ad(R⁻¹)t` 恰为局部族）——这是整个管线的内部一致性检查。
- 结论：**对角稀疏性只在任务框架成立**——恢复律与拟合数据在任务基是同一个问题，
  这个巧合正是"手选基"买来的东西。

### 5.3 S3 多生成元 multigen（du + yaw）

**研究问题**：模型学的是相关性**向量**（哪些生成元相关），还是只是对"单个胜出生成元"的 argmax？

**任务**：`KeyedCircularPhaseSwitchSE3Multigen-v1`——键控门下是**矩形槽**（宽 du、窄 dv），
key 过门后**同时释放 yaw 和 du**（dv 仍被跟踪）。oracle（4 相位）：
du=[1,1,0,0]（选择性，新）、dv=[1,1,1,1]、dw=[1,1,1,1]、roll=[1,1,1,1]、pitch=[1,1,1,1]、
yaw=[1,1,0,0]（选择性）。

**脚本**：`collect_se3_multigen_rollouts.py`、`benchmark_se3_multigen.py`。
**输出**：`se3_multigen/multigen_summary.json`；`VALIDATION_se3_multigen.md`。拟合 216 / 0 失败。

**关键数字（M_multi：恢复 top-2 clearance-drop 为 {du, yaw}）**：

| N | M_multi accuracy | Δ_du | Δ_yaw |
|---|---:|---:|---:|
| 8 | **0.944** | 0.372 | 0.532 |
| 15 | **1.000** | 0.392 | 0.562 |
| 30 | **1.000** | 0.404 | 0.540 |

- M0 切换：du 切换 100%（位置 ~0.69）、yaw 切换 100%（位置 ~0.58）。
- 负对照（keyed 臂控制）：在**只有 yaw 选择性**的 keyed 任务上 M_multi = 0.0（正确不报告 du）。

### 5.4 S4 Planar Push（无旋转对称任务）

**研究问题**：反驳"你针对 circular symmetry 定制了模型"。用**完全没有旋转对称**的任务：
方块在桌面上平推到 ghost 6-DOF 目标（桌面约束反转生成元角色：离面 dw/roll/pitch 被抑制，
面内 du/dv 及 yaw 取决于臂）。

**脚本**：`collect_planar_push_rollouts.py`、`generate_planar_push_contexts.py`、
`prepare_planar_push_subsets.py`、`benchmark_planar_push.py`。
**输出**：`planar_push/planar_push_summary.json`；`VALIDATION_planar_push.md`。拟合 432 / 0 失败。

**关键数字（两个臂：heading_push 需保持朝向、free_yaw_push 朝向自由）**：

| 指标 | heading_push（N=8/15/30） | free_yaw_push（N=8/15/30） |
|---|---|---|
| M_oop（离面抑制） | 0.889 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| α_active_du | 0.968 / 1.031 / 1.101 | 0.991 / 0.997 / 1.011 |
| α_active_dv | 1.025 / 1.035 / 1.018 | 0.939 / 1.021 / 0.999 |
| α_active_dw/roll/pitch（应≈0） | ≤ 0.23（pitch）/ 其余 ≤ 0.09 | ≤ 0.05 |
| α_active_yaw | 0.659 / 0.677 / 0.696（heading 需跟踪） | 0.108 / 0.048 / 0.050（free 被抑制） |
| M_yaw（heading vs free 区分） | **1.0 / 1.0 / 1.0** | — |

**解读**：桌面约束下离面生成元被抑制（α_dw/roll/pitch ≈ 0），面内 du/dv 保持 ~1；
yaw 在 heading 臂跟踪（~0.66–0.70）而在 free 臂被抑制（~0.05–0.11），M_yaw 100% 区分。

### 5.5 S5 LIBERO 抽屉探针

**研究问题**：真实 LIBERO 平移任务（open drawer）是否被模型读作"单轴平移生成元"？

**脚本**：`collect_libero_drawer_probe.py`、`prepare_libero_drawer_subsets.py`、
`benchmark_libero_drawer_probe.py`。
**输出**：`libero_drawer/libero_drawer_summary.json`。拟合 216 / 0 失败。

**关键数字（M_prismatic：du 是唯一 active 生成元）**：

| N | M_prismatic accuracy | α_du | 其它 5 维 |
|---|---:|---:|---:|
| 8 | **1.0** | 1.003 | ≤ 4.0e-8 |
| 15 | **1.0** | 0.984 | ≤ 2.7e-7 |
| 30 | **1.0** | 1.010 | ≤ 1.3e-7 |

- E_α：2.42e-4 → 1.31e-4 → **1.81e-5**（随 N 迅速收敛，泄漏几乎为零）。
- 抽屉开合方向的单轴平移生成元被**几乎无泄漏**地恢复（其它 5 维 α ≈ 1e-7~1e-8）。

**TP-GMM projected 基线**（`libero_drawer/libero_drawer_summary.json` 的 `tpgmm_projected_baseline`）：
- 仓库 TP-GMM 是 SE(2) 基线，故投影到导轨平面 [du,dv,yaw] 拟合（108 fits，0 失败）。
- `M_prismatic_projected_accuracy` = **1.0**（TP-GMM additive 与 TP-GMM SE(2)，所有 N）。
- `alpha_active_projected`（N=8）：TP-GMM additive du≈0.996 / dv≈0.004 / yaw≈0.003；
  TP-GMM SE(2) du≈0.963 / dv≈0.005 / yaw≈0.006——也能锁定 du，但它**无法表示**离面 dw/roll/pitch。

### 5.6 S6 LIBERO 10 任务关系套件

**研究问题**：跨 10 个真实 LIBERO 任务（含不同几何族），模型能否识别"哪些生成元在
move/hold 活跃相位相关、哪些被抑制"？

**脚本**：`collect_libero_relation_suite.py`（**合成数据**，见 §7 诚实性声明）、
`prepare_libero_relation_suite_subsets.py`、`benchmark_libero_relation_suite.py`。
任务规范在 `libero_relation_suite_specs.py`。

**输出**：`libero_relation_suite/libero_relation_suite_summary.json`（67KB）。拟合 3780 / 0 失败。

**M_relation 定义**：在活跃相位 move+hold 上取 max；每个任务"正确"当且仅当所有
oracle 选中生成元 > 0.5 且所有被抑制生成元 < 0.5（阈值 0.5）。

**10 个任务（family）**：
`drawer_middle_open`（prismatic_sliding）、`stove_knob_turn`（revolute_knob）、
`microwave_door_revolute`（revolute_door）、`plate_front_push`（planar_one_axis_push）、
`bowl_on_stove`（support_placement）、`bowl_on_plate`（stacking_support）、
`cream_cheese_in_bowl`（container_in）、`wine_bottle_on_rack`（rack_slot_placement）、
`wine_bottle_on_cabinet`（upright_support_placement）、`moka_pot_on_stove`（compound_support_subtask）。

**关键数字（M_relation_overall，N∈{8,15,30}，180 cell/组）**：

| 模型 | M_relation | e_alpha（N=8/15/30） |
|---|---:|---:|
| **Pdiag finite** | **1.0 / 1.0 / 1.0** | 5.77e-4 / 5.08e-4 / 4.36e-4 |
| Pdiag pointwise | 1.0 / 1.0 / 1.0 | ~1e-31 |
| Full operator | 1.0 / 1.0 / 1.0 | 1.18e-6 / 1.6e-10 / 2.0e-11 |
| TP-GMM additive | 1.0 / 1.0 / 1.0 | 2.09e-2 / 1.55e-2 / 3.60e-3 |
| TP-GMM SE(3) | 1.0 / 1.0 / 1.0 | 1.82e-2 / 1.63e-2 / 6.14e-3 |
| Frame-weighted | 0.0 / 0.0 / 0.0 | 0.123（不收敛） |
| Phase scalar GP | 0.0 / 0.0 / 0.0 | 0.123（不收敛） |

- by_task：**所有 10 个任务在 Pdiag finite 下 M_relation 都是 1.0**（每个 N）。
- 帧级/标量模型无法表达逐生成元选择性（M_relation = 0，E_α 平在 0.123）。

**TP-GMM projected 基线**（`libero_relation_suite/libero_relation_suite_summary.json` 的 `tpgmm_projected`）：
- 1080 fits，0 失败；TP-GMM 是 SE(2) 基线，只在 [du,dv,yaw] 投影上评估。
- `M_relation_projected_overall` = **1.0**（TP-GMM additive 与 TP-GMM SE(2)，N=8/15/30 均 180/180）。
- 诚实说明：这 10 个任务的 oracle 选择生成元都落在 [du,dv,yaw] 投影内，故 TP-GMM projected
  也能拿 M_relation 1.0；Pdiag finite 的增量价值在于是**全 SE(3)**，同时给出离面维
  （dw/roll/pitch ≈ 1e-8）的近零泄漏，而 TP-GMM 无法表示离面生成元。

### 5.7 S7 跨任务迁移（geometry transfer）

**研究问题**：能否把源任务的 Pdiag 剖面**冻结**迁移到目标任务（只需重估目标名义曲线）？

**脚本**：`benchmark_geometry_transfer.py`。
**输出**：`geometry_transfer/geometry_transfer_validation.json`。拟合 180 / 0 失败；
样本 N∈{3,5,8}、repeats 1、split_seed 20260831。

**5 个迁移对（family）**：
`drawer_middle_open → plate_front_push`（one_axis_sliding）、
`stove_knob_turn → microwave_door_revolute`（revolute_yaw）、
`bowl_on_stove → bowl_on_plate`（support_to_stacking）、
`bowl_on_stove → cream_cheese_in_bowl`（support_to_container）、
`bowl_on_stove → wine_bottle_on_cabinet`（cross_object_support）。

**关键数字（summary，方法 × 样本量）**：

| 方法 | m_transfer_acc | e_alpha_mean | heldout MSE |
|---|---:|---:|---:|
| **Ours transfer**（源 Pdiag N=30 + 目标 nominal N=1） | **1.0** | **2.13e-4** | ~9.3e-9 |
| Pdiag finite target scratch | 1.0 | 3.76e-3 / 9.19e-4 / 5.87e-4 | ~2–3e-7 |
| Frame-weighted target scratch | 0.0 | 0.138 / 0.130 / 0.127 | ~6–8e-6 |
| Phase scalar GP target scratch | 0.0 | 0.138 / 0.130 / 0.127 | ~6–8e-6 |

- Ours transfer 在 N=3/5/8 下 m_transfer_acc 均 **1.0**，e_alpha 恒 2.13e-4（源剖面冻结，
  与目标样本量无关）；目标 scratch 的 Pdiag 也能到 1.0 但 e_alpha 高一个数量级（需更多数据）。
- by_pair：5 个迁移对全部 acc 1.0；e_alpha 分对：drawer→plate 1.35e-5、
  knob→microwave 1.38e-5、bowl_stove→bowl_plate / bowl_plate→cream_cheese /
  bowl_stove→wine_cabinet 均 3.45e-4。

**#21 geometry transfer TP-GMM N=8 变体**（`geometry_transfer_tpgmm_n8/geometry_transfer_validation.json`）：
- 75 fits；N=8 固定；在 #20 基础上新增 **TP-GMM SE(3) target scratch** 对照。
- 结果（N=8，method → m_transfer_acc / e_alpha_mean / heldout MSE）：
  - Ours transfer：**1.0** / 2.13e-4 / 9.28e-9
  - Pdiag finite target scratch：1.0 / 7.42e-4 / 2.16e-7
  - TP-GMM SE(3) target scratch：1.0 / **2.62e-2** / 3.44e-5
  - Frame-weighted / Phase scalar GP scratch：0.0 / 0.128 / 6.26e-6
- 解读：TP-GMM SE(3) 目标 scratch 虽能到 acc 1.0，但 e_alpha 高两个数量级（2.6e-2 vs 2.1e-4）、
  heldout MSE 差约 3 个数量级——再次印证"冻结源剖面 + 重估目标名义"的样本效率优势。

**geometry transfer two-scene 变体**（`geometry_transfer/VALIDATION_geometry_transfer_two_scene.md`
+ `geometry_transfer_two_scene_fewshot.csv`）：从 #20 抽出两个结构不同的目标场景
（`knob_to_microwave_door`：纯 revolute yaw；`bowl_stove_to_cream_cheese_bowl`：
support/container 放置）做聚焦对照。Ours transfer 恒 1.0（e_alpha 1.38e-5 / 3.45e-4），
Frame-weighted / Phase scalar GP scratch 恒 0.0，TP-GMM SE(3) scratch（N=8）e_alpha 3.29e-2。
这是 #20 的聚焦呈现，**不是新实验**。

---

## 6. 论文声明 ↔ 代码 ↔ 数字 对照总表

| 论文声明 | 支持数字 | 代码位置（`phase_switch_symmetry/`） | 产物 JSON |
|---|---|---|---|
| yaw 过键后是规范对称 | final yaw ratio 0.0022、unlock 0.0028；Δψ slope 0.0000 | `analyze_phase_switch_rollouts.py` | `VALIDATION.md`、`VALIDATION_circular_step0.md` |
| yaw 是 0→1→0（全 7 相位） | yaw [0,0,0,1,1,0.07,0.04]；平移 0→1（预接触为 0） | `analyze_full_phase_profile.py` | `full_phase_reanalysis/full_phase_verdict.json` |
| 标量帧权是最优折中 | w=0.5681，端点 6.554→0.639 | `benchmark_phase_switch_baselines.py` | `phase_switch_symmetry_baselines/` |
| 规律跨种子稳定 | 改进 1.7576±0.4653，bootstrap [1.2964,2.2295] | `analyze_phase_switch_multiseed.py` | `results/multiseed_summary.json` |
| 对角先验提升少样本 | N=3 E_task 3.27 vs 9.50/24.12；律恢复 88%/92%→100% | `analyze_phase_switch_fewshot.py` | `fewshot_results/fewshot_summary.json` |
| 规律非语义标签产物 | DTW-pos E_gen 0.130 < oracle 0.218 | `analyze_nonoracle_progress.py` | `nonoracle_progress/nonoracle_progress_summary.json` |
| 规律是世界朝向不变 | ρ_axial 0.938–0.985 | `analyze_phase_switch_rotated.py` | rotated 系列产物 |
| 规律非轨迹统计量 | placebo M1 0.0015；gap 0.741–0.761 | `benchmark_symmetry_transfer.py` | `symmetry_transfer/symmetry_transfer_summary.json` |
| 少样本可靠性 | Pdiag worst 0.045 vs Full 1.106，0/18 vs 1/18 幻觉 | `benchmark_symmetry_transfer.py` | `symmetry_transfer/symmetry_transfer_variance.csv` |
| SE(3) 唯一选择性生成元 | yaw purity 0.651→0.729，rank/transition 100% | `benchmark_se3_transfer.py` | `se3_transfer/se3_transfer_summary.json` |
| 对角先验来自任务框架 | rotated e_data 升 2.2–13×，off-diag 0.15→2.77 | `benchmark_se3_basis_ablation.py` | `se3_transfer_basis_ablation/basis_ablation_summary.json` |
| 同时恢复多个选择性生成元 | M_multi 0.944→1.0 | `benchmark_se3_multigen.py` | `se3_multigen/multigen_summary.json` |
| 非圆对称任务也成立 | M_oop 0.889→1.0，M_yaw 100% | `benchmark_planar_push.py` | `planar_push/planar_push_summary.json` |
| 真实 LIBERO 平移 | M_prismatic 1.0，泄漏 ~1e-7 | `benchmark_libero_drawer_probe.py` | `libero_drawer/libero_drawer_summary.json` |
| 跨 10 任务泛化 | M_relation 1.0（全部任务） | `benchmark_libero_relation_suite.py` | `libero_relation_suite/libero_relation_suite_summary.json` |
| 跨任务迁移 | m_transfer_acc 1.0，e_alpha 2.13e-4 | `benchmark_geometry_transfer.py` | `geometry_transfer/geometry_transfer_validation.json` |

---

## 7. 已知边界与诚实性声明

写作时务必如实标注以下三点（与代码/数据一致，勿过度声称）：

1. **geometry transfer（#20/#21）用线性实现，不是有限 SE(3) 实现**：
   `benchmark_geometry_transfer.py::_transfer_predict`（约 L139–142）用的是
   `nominal_curve + context * profile` 的**线性**形式，而非 §2.4 的
   `C0 Exp(P(s)c) C0⁻¹ X0`。迁移结论对"冻结 Pdiag 剖面"成立，但若论文声称迁移也走
   有限几何实现，需改代码或改措辞。

2. **LIBERO 数据（#18/#19）是合成的，不是真实 rollout，也不是学到的策略**：
   `collect_libero_relation_suite.py`（约 L64–67）用 `object_pose_at = α·(nominal + selector·δ)`
   的**合成状态级 qpos 插值**，通过 `set_free_body` / `set_articulated_joint` 直接写位姿，
   PHASE_CODES=(3,4,5,6)=reach/move/hold/retract。写作时称其为"LIBERO 任务几何/关系套件
   上的合成探针"，**不要**写成"在真实 LIBERO 上训练/评测策略"或"来自 LIBERO 演示"。

3. **论文正文只覆盖 SE(2) 插入核心（#1–#7、#10、#12）**：#8/#9/#11/#13 以及全部 D/E 组
   （#14–#21）是补充实验（rebuttal/附录阶段）。若要把 SE(3)/multigen/planar push/LIBERO
   写进正文，需在 `main.tex` 的 §Experiments/§Results 增加对应小节与表格
   （当前 `main.tex` L454–989 无这些内容）。

4. **旧 v1 数据不得用于报告结果**：`phase_switch_symmetry_rollouts/provisional_v1/`
   仅作调试溯源，`VALIDATION.md` 明确"不得用于报告结果"。

---

## 8. 仓库内其它任务线（非本论文，仅溯源）

> 下列目录/文件**同样存在于仓库**，用**同一套"生成元相关性 / do(δc) 干预"方法学**，
> 但作用于**不同任务**，且**不在本论文（`main.tex`）的实验范围内**。它们是 phase-switch
> 任务定稿前的**早期/替代任务探索**，保留作溯源。写作时**不要**把它们当作本论文的受控实验。

| 任务线 | 位置 | 任务 | 状态 / 产物 | 与本论文关系 |
|---|---|---|---|---|
| vertical peg | `vertical_peg_symmetry/` + `vertical_peg_symmetry_rollouts*/` | 竖直销插入：方形（对齐 yaw 相关）vs 圆形（轴向 gauge） | 只有采集脚本 + 原始 rollout H5/JSON（square/circular、negative-yaw probe、mixed targeted）；**无 benchmark/分析 summary** | 早期替代任务，被 keyed→circular phase-switch 取代 |
| PlugCharger | `plug_charger_causal/` + `physics_causal_rollouts*/` | `PlugChargerCausal-v1`，对插座做 do(dx,dy,dyaw) | reset 数据集已校验；几何 rollout + waypoint 响应已独立核对；**mplib 段错误 → 无完整轨迹级 P(s) 校验** | 早期探索，未完成 |
| StackCube / PushT | `demos/` + `STACK_PUSHT_YAW_RELEVANCE.md` | 官方演示的观测性 yaw 相关性检查 | 下载的官方 demos，**非受控干预** | 背景观测，非受控实验 |

要点：

1. `CLAIM_VALIDATION.md`（仓库根）是这三条早期任务线的一份汇总验证记录；**本文档才是论文
   （phase_switch_symmetry）的完整实验记录**，二者不混用。
2. vertical peg 的原型证据：`square_vertical_peg_physics.json` 中 yaw 干预（±15°/±30°）多为
   `success:false` 且接触力约 77–79 N（方形必须对齐 yaw）；`circular_vertical_peg_physics.json`
   中同量级 yaw 干预 `success:true`（圆形轴向 gauge 忽略 yaw）。这验证了"对齐 yaw 是否任务相关
   取决于几何"，后来被更严格的 keyed→circular **相位切换**任务取代。
3. `plug_charger_causal/VALIDATION.md` 自述"validates the controlled do(δc) data-generation layer
   only"，且 `motionplanning_smoke` H5 为截断/无效（mplib 段错误），故无轨迹级 P(s) 结论。

### 8.1 论文的图/视频素材（属于本论文，非独立实验）

- 图：`docs/assets/img/*.png` — `experiment_overview_scenes`、`identifiability_figure`、
  `nonoracle_progress_figure`、`phase_switch_fewshot`、`se3_generator_relevance`、
  `symmetry_transfer_main`。
- 视频：`docs/assets/videos/*.mp4` 与 `phase_switch_symmetry_videos/*.mp4`（编号 01–29，覆盖
  E1–E7 与 S1–S7）。解说稿：`phase_switch_symmetry_videos/对称迁移三臂实验视频解说.md`、
  `模型预测视频解说.md`；清单：`phase_switch_symmetry_videos/*_video_manifest.json`。

---

## 附：论文表格/图清单（`paper_latex/main.tex`）

- 表：`tab:protocol`(L555)、`tab:physics`(L600)、`tab:baseline`(L634)、
  `tab:multiseed`(L669)、`tab:fewshot_n3`(L705)、`tab:fewshot_tpgmm`(L739)、
  `tab:nonoracle`(L774)、`tab:rotated`(L802)、`tab:sym_m1`(L866)、`tab:sym_gap`(L901)、
  `tab:sym_var`(L964)。
- 图：`fig:overview_scenes`(L579)、`fig:baseline`(L656)、`fig:fewshot`(L759)、
  `fig:symmetry_transfer`(L845)、`fig:sym_var`(L956)。
