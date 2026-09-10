# 拾取上下文 × 孔倾斜干预（E × c）仿真采集

匹配真机的「三个拾取位置 + 固定圆孔 + 竖直/倾斜孔」仿真环境。拾取位置是**执行上下文**（初始条件变化），孔倾斜是**关系干预**，两者在 manifest 里正交。

## 采集范围与边界

- **用途**：`usage="dev"`。本批是环境复刻与采集验证，不是辨识训练集。
- **条件**：3 拾取位置 × 2 倾斜（竖直 0°、pitch +10°）= 6 条件；3 个 seed；每条条件 1 次尝试，共 **18 条轨迹**，全部 `success=True`，phase 集 `[-1,0,1,2,3,4,5,6]` 完整。
- **拾取坐标**（pedestal 顶面 `[x, y, z_top=0.28]`，只开放 x/y）：

| pickup_id | 位置 | 相对孔心 `[-0.15,0]` 的水平间隙 |
|---|---:|---:|
| 0 | `[-0.25, -0.18, 0.28]`（原始） | 0.206 m |
| 1 | `[-0.30, -0.02, 0.28]` | 0.151 m |
| 2 | `[-0.20, -0.30, 0.28]` | 0.304 m |

  间隙阈值 0.10 m = 孔外半径 0.07 + pedestal 半宽 0.03。

- **倾斜**：`+10° pitch` 仅作**幅度对应**，不宣称匹配真机的倾斜轴/方向（真机倾斜轴未知）。`+10°` 后续若留作 hold-out 测试，本批不得混入训练。

## 关键实现（对真机流程的取舍）

- 沿用现有端向抓取 + 抬高 pedestal（保证任何孔朝向可达），未重做径向/桌面抓取；抓取方式与真机的「圆柱轴 ⊥ 夹爪闭合方向」在方法上存在差异，属 nominal motion，不影响插入关系干预。
- pedestal 是 kinematic actor，box 的碰撞/视觉体写在局部 pose `PEDESTAL_POS`，故 actor 世界位姿 = `pickup_position - PEDESTAL_POS`（`z_top` 固定 0.28，z 分量恒为 0）。
- 未改 `_get_obs_extra`，观测 schema 不变；`pickup_id`/`pickup_position` 只记录在 HDF5 episode attrs/dataset 与 JSON manifest。

## 复现

```bash
# 1. 生成 E × c manifest（含间隙检查）
/home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
  phase_switch_symmetry/generate_pickup_contexts.py

# 2. reset/恢复/z_top/间隙校验（快，无运动规划）
/home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
  phase_switch_symmetry/validate_pickup_env.py

# 3. 三 seed 正式采集（18 条，保留全部尝试）
for seed in 20260910 20270910 20280910; do
  /home/rocos/miniconda3/envs/maniskill_download/bin/python -s \
    phase_switch_symmetry/collect_se3_rollouts.py \
    --output real_relation_consistency/pickup_contexts/pickup_seed_${seed}.h5 \
    --context-manifest real_relation_consistency/pickup_contexts/contexts.json \
    --seed ${seed} --yaw-mode honest --retries-per-condition 1 \
    --robot-init-qpos-noise 0.01
done
```

输出 HDF5 为写模式，复现前先复制现有产物或另选目录，避免覆盖。

## 产物

- `contexts.json`：E × c manifest（`schema_version 3`，`usage="dev"`）。
- `pickup_seed_{20260910,20270910,20280910}.h5` / `.json`：18 条正式采集 + manifest。
- `smoke_seed_20260910.h5`、`keyed_regression_smoke.h5`：冒烟产物（非正式）。
- 依赖改动：`phase_switch_symmetry_env.py`（`pickup_position` 维度）、`collect_se3_rollouts.py`（字段传递）、`generate_pickup_contexts.py`、`validate_pickup_env.py`。

## 尚缺（下一阶段）

只有 `0°/+10°`，不足以辨识跨幅度响应律。辨识 `P_sim(s)` 需要倾斜 sweep（含 hold-out 幅度），随后冻结、对真机 `+10°` 预测对照。本批轨迹只支撑「改变拾取上下文后，原求解器仍能完成竖直与倾斜插入」这一检查。
