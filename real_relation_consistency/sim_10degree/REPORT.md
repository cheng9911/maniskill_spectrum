# 10° 仿真—真机末段响应比较

## 本次完成的实验

新增 CircularPhaseSwitchSE3-v1 仿真：竖直基线、roll −10°/+10°、pitch −10°/+10°，三个 seed，每个条件单次尝试，共 **15 条轨迹**。全部记录 success=True，并包含完整末段；没有剔除失败后重采。求解器的 screw planning 部分尝试失败后使用既有 RRTConnect 回退，这是同一尝试内的预置规划逻辑。

真机仍使用全部 **50 条竖直孔 + 61 条倾斜孔示范**。成功由用户确认；拾取位置变化与孔位置干预已区分。未更新真机轨迹、窗口或轴估计去贴合新仿真结果。

新仿真条件与指标见 `contexts.json`、`protocol.json`，在本次仿真运行前保存并进行 manifest SHA-256 检查。因为真机结果此前已经分析过，本研究属于探索性比较，不称为预注册确认性研究。

## 统一的比较量

主要比较手爪接近轴在竖直基线与倾斜条件之间的三维方向夹角：

\[
\theta_{response}=\arccos(\bar a_0^\top\bar a_{tilt}),\qquad
\bar a=\frac{\operatorname{mean}(a)}{\|\operatorname{mean}(a)\|}.
\]

仿真每个 seed 的倾斜条件与同 seed 的 nominal 配对，再报告三个 seed 的平均夹角。仿真轴从实际执行的 tcp_pose 计算，不从指令目标计算。真机继续以两组示范平均轴计算；因没有场次和配对标签，不能构造与仿真完全一致的配对层级。

真机窗口：松爪前 0.7–0.2 s。仿真窗口：solver_phase=6；同时检验末段后半段和最后一帧。两者均为装配末段代理，但不声称已建立整条轨迹的 phase correspondence。保留仿真 peg_pose 得到的物体轴响应作交叉检查。

真机孔倾斜轴未知，故分别报告四个模拟方向，不挑选最接近的方向，不将正负方向的向量先混合平均，也不将四个方向当作覆盖了所有可能方向。

## 结果

真机平均手爪轴响应 **10.30°**，本次 10,000 次示范 bootstrap 的条件性 95% 区间为 **9.60–11.05°**。

| 仿真干预 | 仿真手爪轴响应，均值 ± SD | 仿真物体轴响应，均值 | 真机 − 仿真手爪轴 | 差值条件性 95% 区间 |
|---|---:|---:|---:|---:|
| roll −10° | 10.50 ± 2.03° | 9.97° | −0.20° | [−2.19, 1.80]° |
| roll +10° | 9.22 ± 1.11° | 9.97° | 1.08° | [−0.06, 2.39]° |
| pitch −10° | 9.99 ± 0.61° | 10.00° | 0.31° | [−0.62, 1.23]° |
| pitch +10° | 9.47 ± 0.78° | 9.86° | 0.83° | [−0.12, 1.89]° |

每个仿真方向 n=3 个 seed；SD 是 seed 间离散度。真实区间对两组 episode 分别重采样；仿真差值区间重采样已计算的 seed 配对响应，并与同一次真实 bootstrap 响应相减。不同方向共享真实数据及仿真基线，因此四个区间相关，均为单独区间而非同时置信带。三个 seed 的经验 bootstrap 只描述现有样本的不确定性，不能视为充分的总体覆盖保证。

本次真实 bootstrap 重采样次数和随机种子与上一轮探索不同，故区间上界由约 11.02° 变为 11.05°；点估计 10.2997276° 完全不变。这是数值 Monte Carlo 差异，未改变真实数据或分段。

## 对“规律一致性”的判断

**结果支持：同样 10° 的孔轴倾斜，在仿真与真实成功示范中均伴随约 10° 的末段轴向调整。** 这是对倾斜响应幅度的结构一致性证据。

**尚不能宣称严格统计等效或完整 P(s) 一致。** 没有基于独立标定和装配容差给出的等效界限，也未量化手爪—圆柱变换误差、场次效应、孔倾斜轴和角度标定误差。区间包含 0 只表示本比较未清楚分辨差异，不能证明等效。没有为了让结果“通过”而事后选 ±1° 或 ±2°。

更重要的是，仿真手爪轴与圆柱轴末段夹角约 **0.11–2.15°**，跨条件/seed 不完全相同。物体轴响应比手爪响应更稳定。因此先前直接以“仿真物体轴”比较“真机手爪轴”会忽略代理误差。本次主要结果已统一到手爪轴；物体轴结果仅作辅助，不能借其数值更接近 10° 来替代主比较。

仿真规划器明确使用已知孔倾斜生成目标姿态，因而本次验证的是它实际执行出的响应与真实示范的相容性，不是“自主发现规律”、冻结学习规律迁移、yaw 不传播、平移选择性或阶段切换。日志的 max_contact 字段为零，本次不据此推断无接触，也不作接触力或摩擦一致性结论。

## 产物

- `comparison_10degree.png/.pdf/.svg`：对照图。左图蓝点为全部 3 个 seed，短横线为方向均值；橙色菱形和误差线为真实夹角与条件性 95% 区间。右图为差值和条件性区间。
- `comparison.csv`：全部方向的主结果与区间。
- `sim_axis_responses.csv`：12 个干预尝试的手爪/圆柱轴响应，保留三种末段窗口。
- `episode_axis_audit.csv`：全部 15 条尝试、成功、末段帧数与轴向量。
- `circular_seed_*.h5/.json`：原始仿真轨迹与采集记录。
- `comparison_provenance.json`：新分析脚本、输入、协议和原始仿真文件的 SHA-256。

## 复现

在仓库根目录，用以下命令采集；将 seed 和文件名分别替换为 20260910、20270910、20280910。输出 HDF5 使用写模式，复现时应先复制现有产物或选择新的输出目录，避免覆盖原始运行。

```bash
/home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
  phase_switch_symmetry/collect_se3_rollouts.py \
  --output real_relation_consistency/sim_10degree/circular_seed_20260910.h5 \
  --context-manifest real_relation_consistency/sim_10degree/contexts.json \
  --experiment real_relation_consistency/sim_10degree/protocol.json \
  --seed 20260910 --yaw-mode honest --retries-per-condition 1 \
  --robot-init-qpos-noise 0.01

/home/rocos/miniconda3/envs/lerobot/bin/python -s \
  real_relation_consistency/compare_10degree.py
```
