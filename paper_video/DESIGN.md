# 论文演示视频设计文档（v2，待审核）

状态：**草稿 v2，按审核意见重排叙事。真机段由用户自备，本阶段先跳过；其余段落实现前请再确认一次。**

---

## 0. 术语规则（硬性约定）

- 所有 ManiSkill/SAPIEN、LIBERO/MuJoCo 的渲染画面一律叫 **simulation rollout / physics-simulated rollout**。
- 只有真实机器人拍摄的数据才能叫 **real-robot demonstration / hardware rollout**。
- 全片（含文档与代码）不得用 "real demonstration" 指代仿真画面。

## 1. 目标与主线（重排后）

一条 ~80s、16:9 的论文演示视频，主线从「资产展示」改为「论文核心论点」：

> **为什么需要 \(P_r(s)\) → 如何辨识 \(P_r(s)\) → \(P_r(s)\) 如何改变轨迹 → 冻结 + 真机迁移（高潮）→ 跨关系广度 → 结论。**

即把可迁移的对象讲成是 **relation law \(P_r(s)\)，而不是 trajectory**。

## 2. 叙事权重（用户给定）

| 段 | 叙事 | 权重 | ~时长 |
|---|---|:--:|:--:|
| 1 | 任务关系变化 → 为什么需要 \(P_r(s)\) | 15% | ~12s |
| 2 | 从 controlled interventions 辨识 \(P_r(s)\)（含矩阵全局结构） | 20% | ~16s |
| 3 | \(P_r(s)\) 如何选择性改变轨迹（动态例子） | 20% | ~16s |
| 4 | 冻结 law → 真机 nominal → 真机倾斜预测/执行 | **30%** | ~24s |
| 5 | 跨关系 breadth evidence（LIBERO） | 10% | ~8s |
| 6 | 结论：law 而非 trajectory 是可迁移对象 | 5% | ~4s |

## 3. 分镜

### 段 1 — 为什么需要 \(P_r(s)\)（~12s）

- 画面：键控轴孔任务仿真 rollout，展示**同一个几何变化在不同阶段作用不同**（keyed 阶段 yaw 决定能否通过矩形 gate，clearance 后进入圆孔则 yaw 无关）。
- 关键动态：`α_ψ(s): 1 → 0`（pre-clearance 相关，post-clearance 被抑制）。
- 文案锚点：
  - 「Not every geometric change should affect the skill equally — or at every stage.」
  - 引入 `P_r(s) = Diag[α_1(s), …, α_d(s)]`。
- 资产：复用 `01_nominal`、`02_yaw_matched`、`04_yaw_mismatched_blocked`、`06_postclear_yaw`（3D 仿真）。

### 段 2 — 辨识 \(P_r(s)\)（~16s）

- 画面：`{c^(n), X^(n)(s)} → P_r(s)`；few-shot `N=3/5/8`。
- 后接**矩阵全局结构图**（原 fig3a，重排为中段结构图，不再叫「6×13 矩阵展示」）：
  - 文案改为：「The learned law exposes which task-local generators matter across different relations.」
  - 动画依次高亮 relation family：`translation → revolute → support → insertion`。
- 资产：复用 few-shot 数据 / `20_fewshot`（2D）；矩阵动画（**新渲染**，复用 `fig3_se3_generators.py` 的 `panel_dot_matrix` 数据）。

### 段 3 — \(P_r(s)\) 如何改变轨迹（~16s）

- 矩阵之后必须接**一个明确的动态例子**（孔倾斜）：
  - `c_tilt = 10°`，`u_r(s) = P_r(s) · c_tilt = α_tilt(s) · 10°`。
  - 并排显示 nominal（0°）轨迹 vs adapted（10°）轨迹，标注 α_tilt(s) 从小到大 = 轨迹**渐进式**响应倾斜。
  - 再叠加 α_tilt(s) 曲线（align 阶段 0→1、insert 恒 ≈1）与移动 marker，让「α(s) 从小到大」有几何含义。
- 资产：复用 5 段倾斜视频（pitch ±0/±10/±15°）+ α_pitch(s) 曲线（`predict_10degree` 产物）；（可选）**新渲染**一个「nominal vs adapted 并排 + α(s) 曲线同屏」的合成画面。

### 段 4 — 冻结 + 真机迁移（~24s，**用户自备**）

- **本阶段跳过，占位。** 由用户提供真机素材与脚本：
  - `P_sim(s) → FREEZE`；真机三个 pickup contexts `E1,E2,E3`；真机竖直 nominal `X_{0,real}^(m)`；真机倾斜孔 `c_{10°}`；生成 `X̂_{10°,real}^(m)`。
  - 叠加：measured real demo / ours / scalar / no-adaptation，重点放大 alignment/insertion。
- 本阶段交付物中预留 `seg4_real_transfer` 插槽与拼接接口。

### 段 5 — 跨关系广度（~8s）

- 快速闪过 LIBERO 的 sliding / revolute / placement 等 relation：
  - 「The same formulation extends beyond insertion.」
- 角色：从「高潮」降为 **breadth evidence**。
- 资产：LIBERO 场景视频（**新渲染**，`lerobot_libero` 环境，state-level、无机械臂——诚实标注为 relation-level 展示）。

### 段 6 — 结论（~4s）

- 「The relation law — not the trajectory — is the transferable object.」
- 资产：PIL 文字卡（新渲染）。

## 4. 资产盘点

### 4.1 复用（仓库已有）

| 资产 | 段 | 形态 |
|---|---|---|
| `01–06` 键控轴孔 yaw/平移干预 | 1 | 3D 仿真 |
| 倾斜 5 段 `pitch_±0/±10/±15°` | 3 | 3D 仿真 |
| α_pitch(s) 曲线（`predict_10degree`） | 3 | 静态图 |
| `20_fewshot`（N=5 vs N=30） | 2 | 2D 示意 |
| `fig3_se3_generators.py` 矩阵数据 | 2 | 静态图数据 |

### 4.2 新渲染（本阶段实现范围，不含真机）

| 资产 | 段 | 环境 |
|---|---|---|
| 矩阵关系族高亮动画 | 2 | `maniskill_download`（matplotlib）|
| 「nominal vs adapted + α(s) 曲线」合成画面 | 3 | `maniskill_download` / PIL |
| LIBERO 十任务 + 迁移对场景视频 | 5 | `lerobot_libero` |
| 标题卡 / 结论卡 | 1/6 | PIL |

### 4.3 用户自备（本阶段跳过）

| 资产 | 段 |
|---|---|
| 真机 nominal / tilted demo、ours/scalar/no-adapt 对比 | 4 |

## 5. 技术方案

| 项 | 方案 |
|---|---|
| 分辨率 / 帧率 | 1920×1080（或 1280×720），30 fps，H.264 `yuv420p` |
| 标题卡 / 字幕 | PIL 叠加（复用 `render_tilt_videos.py` 的 PIL 方案）|
| 拼接 | 系统 ffmpeg `concat`；段 4 留空插槽，用户素材到位后替换 |
| 字幕语言 | 英文主标题（待确认是否加中文字幕轨）|
| 音频 | 无（纯画面 + 文字），投稿时再配 |
| 产物目录 | `paper_video/`（脚本 + `segments/*.mp4` + `paper_video.mp4` + manifest）|

## 6. 待确认项（本轮剩余）

1. **段 4 真机素材**：你自备，何时/以何形式接入？（本阶段留 `seg4_real_transfer` 占位）
2. **段 3 合成画面**：是否需要「并排 + α(s) 曲线同屏」的定制合成，还是直接用现有倾斜视频逐个播放即可？
3. **字幕语言**：英文 / 双语？
4. **分辨率**：1080p / 720p？

## 7. 产出物清单（本阶段）

- `paper_video/make_matrix_animation.py`（段 2）
- `paper_video/make_adapted_trajectory.py`（段 3，可选合成）
- `paper_video/render_libero_scene_videos.py`（段 5）
- `paper_video/make_paper_video.py`（编排 + 拼接 + manifest；段 4 占位）
- `paper_video/segments/*.mp4` + `paper_video/paper_video.mp4`
- `paper_video/manifest.json`

## 8. 复现命令（草稿）

```bash
# 段 2 矩阵动画
/home/rocos/miniconda3/envs/maniskill_download/bin/python paper_video/make_matrix_animation.py

# 段 3 合成（可选）
/home/rocos/miniconda3/envs/maniskill_download/bin/python paper_video/make_adapted_trajectory.py

# 段 5 LIBERO（唯一含 libero）
/home/rocos/miniconda3/envs/lerobot_libero/bin/python paper_video/render_libero_scene_videos.py

# 编排（段 4 占位）
/home/rocos/miniconda3/envs/maniskill_download/bin/python paper_video/make_paper_video.py
```
