# 当前真机生成使用的冻结仿真规律

该版本来自 pitch 幅度 sweep，11 条件 × 3 seed，训练角度 0、±2.5、±5、±7.5、±12.5、±15°，±10° 未参与拟合。

模型形式为 P(s)=diag(alpha_du, alpha_dv, alpha_dw, alpha_roll, alpha_pitch, alpha_yaw)。本次只干预 pitch，因此只能解释 alpha_pitch；其他对角系数虽然存在于原始文件中，但未由该设计辨识，有些呈现边界值或 seed 间剧烈变化，不能当成已验证平移/roll/yaw 规律，也不能默认它们是 0 或 1。

在该对角模型假设下，对单一输入 c=theta*e_pitch，有 P(s)c=alpha_pitch(s)*theta*e_pitch。alpha 是无量纲系数，不是角度。有限作用是 C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)，需遵循训练时的任务坐标、生成器顺序和相位定义。

pr_pitch_identified.csv 含全部 100 个样本、三个 seed、均值和跨 seed 标准差。每个相位 25 个点，原 source_progress 在相位边界重复；插值请分相位进行，不要把重复边界坐标当作严格递增网格。CSV 的 sample_index 唯一。标准差描述三个源拟合的离散，不是真机置信区间。

相位名 align/enter/unlock/insert 为原模拟控制器标签；圆柱任务的 unlock 名称不意味着真机存在对应物理解锁事件。近似形状为 align 中 0→1（有约 9% 过冲），其余相位接近 1，并非手工设定分段阶跃。

当前三个源 seed 均值的终端 alpha=1.0052024453，对名义 10° 输入的施加旋转为 10.0520244532°。这不同于对整个 insert 相位平均后得到的终端响应统计，不能混用。

此前新估计的真实变换含独立平移及额外旋转，单 pitch 律不足以确定这些分量各自的响应。将整个 SE(3) 对数都乘 alpha_pitch 是新的标量扩展假设，不能称为本次仿真已辨识出的全六维规律。

原冻结文件未改动；此处只是无损提取与汇总。曲线可见 prediction_10degree.png 的左图。
