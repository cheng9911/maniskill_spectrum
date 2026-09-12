# 事实边界
- collect_phase_switch_rotated.py:solve_se3 根据孔姿态设置目标；使用 screw，失败后 RRTConnect；并非纯物理自发响应。
- collect_se3_rollouts.py:collect_se3 圆柱分支明确传 align_yaw=0.0。
- phase_switch_symmetry_env.py:evaluate SE3 success 为位置 <0.012 m 且轴角 <0.15 rad，不限制 yaw。
- full_se3_law/REPORT.md 是既有同一策略六通道留出验证，不是跨策略验证。
- benchmark_se3_transfer.py 开头 oracle 明确来自 solver geometry，不是独立物理真值。
文献事实：本次为本地代码实施方案，无文献结论或新颖性论断；不补充无关引用。
定义：模式是显式不同的轨迹选择规则，seed 不是模式；P 是条件响应模型，不预设策略无关。禁止称仿真模式覆盖人类全部多模态性。
