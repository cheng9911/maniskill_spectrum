# QA record

- 验证 33 条轨迹（11 条件 × 3 seed）全部 success=True、phase 集完整（含 phase 6），无失败重采。
- 核对每个 HDF5 manifest 哈希与冻结 contexts.json 一致；记录 seed 与干预幅度。幅度集恰为 {0, ±2.5, ±5, ±7.5, ±12.5, ±15}°，±10° 确认不在训练集（作为 hold-out）。
- 辨识三个 seed 独立拟合，`optimization_success=True`，nfev=2/8/2，α_pitch(terminal) ∈ {1.002, 1.008, 1.005}。
- α_pitch(s) 形状核对：align 阶段内 0→1 上升（均值 ~0.65）、enter/unlock/insert 恒 ≈1、起点 ≈0。与物理预期一致；**仅仿真证据**，未与真机瞬态比较。
- 预测用有限 SE(3) 变换（非把 α 对角元当线性导数直接乘 10°），终端轴 = phase-6 各 bin 的 mean z-axis（对齐 compare_10degree 的「phase-6 全帧均值轴」）。
- 留出校验：±10° 在训练范围 [−15, 15] **之内**，是**插值**而非外推。预测与**独立采集**的 sim_10degree 纯 pitch ±10° 轨迹（同 env/几何/拾取位，另一 manifest）的实际 peg 终端轴对比：pitch +10° 误差 +0.156°、pitch −10° 误差 +0.020°；逐 seed 误差在 [−0.220, +0.265]°。sim_10degree 的拾取位为默认 `PEDESTAL_POS = [-0.25,-0.18,0.28]`，与本 sweep 相同。
- 跨幅度曲线全部来自模型自预测，`response(θ)≈1.0018·|θ|`；曲线线性本身不作为泛化证据，留出误差才是。
- **范围更正**：预测在**仿真名义几何**内（冻结 nominal_frame/nominal_curve）进行；真机数据只贡献终端幅值 10.2997°（bootstrap 95% CI [9.60, 11.05]°）。未重建真机名义轨迹或孔坐标系，故**不是真机几何下的轨迹预测**。
- **代理误差更正**：sim_10degree 手爪—圆柱末段夹角 ~0.11–2.15° 是**仿真**测量，不当作真机手爪—圆柱代理误差的已知范围（真机该变换未标定）。
- 未从结果反推孔倾斜轴/角度；真机倾斜轴未知，只比幅度。未作等效 pass/fail，未事后选 ±1°/±2° 界限。
- 图注修正：恒等 baseline 画为 |θ|（指标是非负夹角）；±10° 标记各自独立计算；消除 PDF 中 4.9 pt 下标（ylabel 提升字号，下标 ≥6.3 pt）。
- 原始真机数据与先前仿真产物（sim_10degree、pickup_contexts、SE3 benchmark）未改动；本步产物全部落在 tilt_sweep/ 下（`heldout_10degree.csv` 为新增）。
