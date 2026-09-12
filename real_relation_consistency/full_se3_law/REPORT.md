# 六生成器仿真关系规律：已重新辨识并冻结

本次使用已有圆柱孔 CircularPhaseSwitchSE3-v1 的六维干预 rollout，重新辨识了三个平移及 roll/pitch/yaw 的响应。没有给旧 pitch-only 模型的缺失系数填常数，也没有改动旧冻结文件。新的模型仍是六通道对角 P(s)，不是无约束的 6×6 耦合算子。

## 数据、划分与辨识

源 seed 为 20260818、20270818、20280818。每个 seed 有 75 条件：1 baseline、60 mixed、14 isolated。记录共 228 次尝试，其中 3 次失败；225 个条件各选择最早 usable 的成功完整尝试，有两个条件用到重试。失败完整保存在 attempt_audit.csv；因此辨识与预测评价是成功/重试条件化的数据分析，不是 228 次全部执行的成功率评价。

每个 seed 使用 baseline + 45 mixed，共 46 条件拟合；排序后每第四个 mixed（15 条）及全部 14 isolated 保留作评价，共 29 条件。三 seed 总拟合条件数 138、评价条件数 87。划分由条件 ID 确定，未按误差筛选。所有数据先前已在项目使用过，本次为回顾性新拟合，不称为全新确认性试验。

六维输入与 SE(3) 对数输入经物理量尺度归一化后都为 rank 6，输入条件数约 1.95。满秩支持六个通道获得激励，不等于证明拟合唯一或物理定律唯一。模型使用既有 SE3SmoothFinitePDiagModel：alpha∈(0,1.25)、24 个 RBF、width .065、smoothness .1、3 次 nominal 更新。三次优化均报告成功。

平移训练幅度为 ±12 mm 混合范围，独立平移评价条件为 ±15 mm，属于有限外推。roll/pitch 单轴评价为 ±15°，yaw 为 ±15/30°。完整尝试和训练/评价 ID 均已保存。

## 规律

生成器顺序为 `[du,dv,dw,roll,pitch,yaw]`，位于 nominal socket frame。系数无量纲，缩放的是 SE(3) 对数中的平移/旋转分量，而不是直接逐项缩放有限欧拉角。

\[
P(s)=\operatorname{diag}(\alpha_u,\alpha_v,\alpha_w,\alpha_{roll},\alpha_{pitch},\alpha_{yaw}),
\quad
\widehat X=C_0\operatorname{Exp}\!\left(P(s)\operatorname{Log}(C_0^{-1}C)\right)C_0^{-1}X_0.
\]

三个源拟合和相位内样本的均值：

| 仿真阶段 | du | dv | dw | roll | pitch | yaw |
|---|---:|---:|---:|---:|---:|---:|
| Align | 0.616 | 0.569 | 0.662 | 0.658 | 0.623 | 0.104 |
| Enter | 0.975 | 0.972 | 0.999 | 0.977 | 0.996 | 0.131 |
| Unlock 标签 | 0.967 | 0.982 | 0.995 | 0.975 | 0.995 | 0.091 |
| Insert | 0.966 | 0.983 | 1.001 | 0.981 | 0.999 | 0.089 |

这张表是摘要，实际规律有 100 个相位采样点，每阶段 25 点。前五个通道在 align 中上升，其余装配阶段接近 1；yaw 比它们小，但不是精确零。本次不把小的非零 yaw 人为截断，也不据此宣称真机 yaw 已验证不传播。Unlock 仅为源控制器标签，圆柱真机没有相应解锁标签真值。

## 留出预测检查

下表是对 87 个 seed×条件的“每条相位轨迹 RMSE”再等权平均，不是将全部帧池化的 RMSE，也不是置信区间。

| 方法 | 位置 RMSE (mm) | 完整姿态 RMSE (°) | 轴方向 RMSE (°) |
|---|---:|---:|---:|
| 冻结六通道对角模型 | 3.806 | 1.620 | 0.902 |
| 既有共享标量仿射基线 | 14.996 | 6.670 | 3.639 |
| Identity 完整跟随 | 17.699 | 9.643 | 2.134 |
| 不适应 | 17.119 | 7.065 | 7.029 |

不适应与 identity 使用同一已拟合 nominal 曲线；共享标量基线使用既有仿射实现，不是有限 SE(3) 标量非线性优化器。圆柱孔的 yaw 选择性会影响这些聚合误差，不能把仿真优势直接当成真机执行优势。按条件、生成器、阶段的全部结果在 heldout_metrics.csv。

## 如何用于真实变换矩阵

如果已有基座系左乘变换 Delta 和真实 nominal task frame C0，首先计算：

```
xi = Log(inv(C0) @ Delta @ C0)
Xhat(s) = C0 @ Exp(alpha(s) * xi) @ inv(C0) @ X0(s)
```

实现见 ../full_se3_adapter.py，输入 `X0[N,4,4]`、`Delta[4,4]`、`C0[4,4]` 和 `alpha[N,6]`。适配器已与原模型预测逐项核对，并验证 alpha 全零为 nominal、全一为 Delta@X0。

不能省略 C0 而直接在机器人基座系对六维分量套用源规律。此前从终端 TCP 得到的 Delta 并不能唯一提供孔的完整 nominal frame；圆柱轴自旋自由度、有效中心与孔口位置仍需明确。真机相位识别也没有自动建立与源四个控制阶段的精确对应。本次因此不静默覆盖旧的机器人候选轨迹。

## 交付与复现

- [六通道规律图](pr_all_generators.png)，另有 PDF/SVG/600 dpi TIFF。
- `pr_all_generators.csv`：每点每通道三个 seed、均值、SD。SD 是源拟合间描述性离散，不是置信区间。
- `frozen_full_se3.npz/.json`：六通道系数、参数、nominal 曲线/坐标、训练与评价输入、预测以及哈希。
- `phase_summary.csv`、`heldout_metrics.csv`、`attempt_audit.csv`、`verification.json`。
- source_progress 在相位边界重复，插值应使用相位内进度或唯一 sample_index，不直接把重复坐标当作严格递增轴。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/rocos/miniconda3/envs/maniskill_download/bin/python real_relation_consistency/identify_full_se3.py
OPENBLAS_NUM_THREADS=1 /home/rocos/miniconda3/envs/maniskill_download/bin/python real_relation_consistency/verify_sixlaw_and_phases.py
```

拟合环境依赖 user-site 中的包，此命令不加 `-s`；真机分段使用 lerobot 环境并加 `-s`。原始 rollout 与旧 pitch 冻结模型保持不变。
