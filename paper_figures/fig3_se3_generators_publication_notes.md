# Fig. 3 改版说明与核查

## 面板

- a：沿用 N=30 原汇总，保留全部 78 个均值；仅对 ≥0.08 的值标数字。
- b：沿用四条控制曲线及均值 ±1 SD，每组 18 fits = 3 seeds ×6 subsets。
- c：已从冻结 keyed 仿真轨迹重新估计真实响应矩阵，恢复两张热图。原脚本和 publication 脚本都读取同一份数据。计算细节见 fig3_basis_empirical_methods.md。
- d：保留全部 24 个阶段均值，突出最终阶段 du/yaw。改版 a/d 共享 0–1.05 色标，避免裁切最大值 1.0125。c 使用独立的有符号系数色标。

## c 结果

| 阶段 | 局部基底 r_off | 旋转基底 r_off |
|---|---:|---:|
| align | 0.4060 | 0.4020 |
| enter | 0.0212 | 0.0212 |
| unlock | 0.0391 | 0.1829 |
| insert（图中展示） | 0.0503 | 0.3126 |

r_off 是平均矩阵的非对角 Frobenius 范数 / 平均矩阵的 Frobenius 范数，不是逐拟合比值的平均。所有阶段和逐拟合矩阵均导出。选择 insert 是为了检验 yaw 释放后的各向异性；enter 近似单位阵，正确换基不会使其变稠密。align 也不能声称局部近似对角。结论须限定任务和阶段。

数据来自物理仿真，不是实体机器人。旧图合成矩阵的 0.06/0.75 已移除；记录中的 0.15/2.77 未用作缩放目标。其他历史图中的合成矩阵未在本次范围内修改。

## 复现

```bash
python paper_figures/rebuild_fig3_basis.py
python paper_figures/fig3_se3_generators_publication.py
python paper_figures/fig3_se3_generators.py
```

重算需要 numpy/scipy/h5py，绘图另需 matplotlib。本次使用已有 maniskill_download Conda 环境及 python -s -B，未改系统依赖。publication 图幅 183×108 mm，输出 PDF、SVG、600 dpi PNG 和预览；另有四张面板 CSV。完整 c 数据保存为 fig3_basis_empirical.npz/.csv/.json。

## QA

- 独立矩阵指数检查 SE(3) 对数：零角度、小角度、一般角度均通过。
- 每次拟合验证旋转后重新拟合等于 QᵀAQ，且 Frobenius 范数不变；另验证奇异值不变。
- 全部 18 次拟合满秩，无缺失子集或失败拟合剔除。
- 全部 288 个系数 CSV 与矩阵均值严格相等；原始 HDF5 来源哈希已核查。
- 两份 PDF 实际字形均通过 5 pt 下限检查。
- 整图及 c 面板目视检查，修正顶部说明、基底名称的重叠。
- a/d 为描述性均值图；c 的系数 SD 保存在数据包，图上未展示。不能从热图推断显著性。

正文图注需同步写明 c 使用 keyed/insert 阶段、输入输出同时换基、仿真轨迹拟合均值及 r_off 定义。

本轮排版：按要求移除圆面积说明、18 fits/SD 说明、行列说明及方框说明；相应解释保留在此文档与计算说明中。a/d 紫色色带保留；c 改为青绿—白—紫的发散色带，以青绿表示负值、紫色表示正值。行间留白减少 8 mm、底部减少 3 mm，字号与数据区物理尺寸保持不变。
