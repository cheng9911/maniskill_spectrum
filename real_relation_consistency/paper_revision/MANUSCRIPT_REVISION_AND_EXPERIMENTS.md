# 论文修改与补实验清单

2026-09-12。依据：用户附件96c82ae8…/pasted-text.txt，所提供文本到Finite Geometric Realization中途结束；未提供后续实验、讨论、结论与参考文献。另核对本仓库multimodal_task_law的REPORT、phase_alignment.py、execute_transfers.py、identify_modes.py及环境实现。本次是限定范围审查，不代表重新复现所有数值或文献核验；不改原稿和旧结果。

## 判断

“响应条件化于执行过程、相位对齐不能自动证明唯一共同规律”的前述判断正确，且本文方法段已经部分明确这个边界。无需将全篇改成canonical law理论。保留现有interventional estimand、有限SE3与matched-relation transfer路线，将四模式实验定位为模式依赖和可迁移性边界检验。

原附件建议的严格等式P_z(s)=Pbar(phi_z(s))目前只能作为额外假设。任务约束唯一决定Pbar、所有策略差异都能由进度吸收、97%成功归因于稳定结构，均未由现有实验单独建立。

## 一、逐处修改

### A. 标题和摘要

现标题可保留。没有必要仅因多模态差异删除laws；若想更保守可将Laws改为Models，属于定位选择而非数学必需。

摘要第二段的“preserves its task-local structure across geometric, coordinate, and execution variations”需收紧，不能将坐标等变性与策略不变性并列成同等结论。

建议替换为：

> Experiments evaluate generator selectivity and matched-relation transfer under geometric and coordinate changes. Controlled demonstration modes reveal strategy-dependent transitions, while frozen source models remain useful for cross-mode reproduction under a shared execution pipeline.

最后一句可改为：

> These results support generator relevance as a compact response representation for skill transfer under specified relation semantics, progress alignment, and execution conditions.

90.2%、3.27/10.48与五种matched relations不在本次提供的实验正文内，需逐一对应原表/计算口径；本审查不确认也不否认其数值。

### B. Introduction

保留“response rather than nominal motion”和有限几何分离。增加一句明确边界：

> We identify the response of a specified demonstration-generation process rather than a unique response implied by task success. Transfer is therefore evaluated empirically across matched relation instances and execution modes.

将“which generators should influence”中规范性的should限定为模型选择/任务要求，避免由观测响应直接推断必要性。成功任务集合并不唯一决定一条轨迹响应。

第三条贡献新增mode dependence审计；不要写证明controller independence。

### C. Problem Formulation / eq:interventional_jacobian

原文第256–261行已经写明不预设controller-independent invariant，保留。显式定义执行模式z，并说明为简洁在主推导中省略下标：

J_{c,r,z}(s) = d/dc Log(X_{0,z}(s)^(-1) X_z(s;c))^vee |_(c=0).

这是在固定模式、进度协议及初态约定下的响应。对随机示范须说明是配对初态的条件响应、条件均值响应，还是另一个统计量；当前配对区组数据不能不加定义就称整个人类多模态分布的响应。

同一个s如何对应干预前后状态也属于estimand定义。主分析使用预先规定事件规则；从测试响应本身对齐得到的是回顾性描述，不能当部署时可获得映射。

### D. eq:response_factorization 与唯一性

原文“for responses lying in this supplied generator span”已给出必要限定，不是错误。再加一句防止误读：

> This uniqueness is conditional on a fixed response Jacobian, nominal trajectory, and supplied basis; it does not imply a unique response across execution strategies.

若希望覆盖不满足span的实际响应，用：

J_z = G_z A_z + E_z,
A_z = G_z^dagger J_z,
E_z = (I-G_z G_z^dagger)J_z.

加权度量应明确，位置和角度的数值尺度影响伪逆投影。d_r=6且B可逆时span残差为零，约束转移到A是否近似对角；d_r<6时不可忽略span外分量。对角P是结构模型，不是由激励rank满秩自动推出的物理性质。

### E. 坐标约定 / eq:relation_intervention

本文已经通过C=C0 Exp([Bc]^wedge)定义c。因此之前关于B P c公式的条件性提醒在理论定义层面已满足，有限作用和adjoint表达可以保留，不应误改成不能使用B P c。

但“All experiments ... c generated directly in ... Lie basis”需与实现核对：当前SE3环境接收平移offset和Rx(roll)Ry(pitch)Rz(yaw)构造socket，identify_full_se3调用模型转换到Log坐标。环境参数和理论c并非通常同一个数组。建议改为：

> The simulator uses pose parameters to instantiate relation configurations. Model inputs are obtained from the realized relative transform through the logarithm map and expressed in the supplied basis. Raw simulator offsets and Euler angles are distinguished from these Lie-algebra coordinates.

对每个实验检查实际输入管线后再定稿；纯pitch条件与指数形式可以一致，不能据此覆盖所有mixed条件。

### F. 轨迹对象 / nominal来源

主文X0定义为end-effector，但旧benchmark_se3_transfer.task_curve_se3实际读peg_pose，新multimodal主拟合也读peg。必须明确物体位姿和TCP分别用哪个符号，并报告固定抓持变换H=Tpeg^-1 Ttcp。H稳定时相同左作用可作用于两者；有滑移时不等价。不能把全部peg结果直接标为完整TCP实验。

主文“only P is learned”应说明X0是外部固定输入还是联合估计。旧full_se3拟合nominal_iterations=3，有nominal更新；新pitch模型固定nominal。区分“只有P为可迁移参数”与“优化中只有P被估计”。

### G. 进度与canonical解释

增加短小的Progress alignment and execution dependence段，先不建立新中心理论。

> We use P_{r,z}(s) to make execution dependence explicit and omit z when the source process is fixed. A shared phase-indexed representation is a testable approximation rather than an assumed invariant. Phase mappings used for transfer must be available from the target nominal trajectory or independent measurements and must not be fitted to held-out target responses.

若引入相位符号，用eta或psi_z(s)，不要与现有eq:alpha_rbf中的RBF向量phi(s)混名。可在Discussion提出P_z(s)≈Pbar(psi_z(s))+R_z(s)，保留残差，不宣称已识别出该分解。

### H. 实验新增小节

建议小节名：Mode dependence and cross-mode transfer。

图A四模式实际轨迹；图B相对各自0°参考的响应和差异，分阶段；图C源→目标的全段/过渡/末端误差；表D不同执行后处理下的成功及精度。

将fig5c列为response-aligned descriptive visualization，不能称独立验证canonical law。phase_alignment.py使用自身a>0.05、a>0.95确定边界；a=1-axis_error/abs(theta)是对齐分数，不是独立辨识的alpha_pitch。其a/b面板仅一个block的+10示例，c聚合多block与±10，图注须说明样本范围差异。阈值首次穿越还需检查起始范围、持续时间和无穿越情况。

REPORT的“插入段完全稳定”“差异只来自时机”“正yaw失败就是关节限位根因”“spin≤0.76几乎不响应”应分别改为有区间支持的有限表述。0.76 deg/deg不能仅凭小于1称为可忽略。规划失败不等于已证明几何不可达，需要IK/限位诊断。

58/60与57/60写观察到的成功数接近，不写已等效/无差距；没有统计检验输出也不写无显著差异。全260成功且无筛除时可写“全部收集的示范成功”，不必暗示有未报告的成功筛选。

### I. 现有执行口径的重要修正

execute_transfers.py先把迁移参考下采样为40个peg waypoint，再逐点调用move_to_pose_with_screw，失败fallback RRTConnect；最后base.goal_pose.sp给的是目标完整位置和姿态，经同一move_pose规划执行两次。它不是仅沿轴下降2 mm的普通跟踪修正。

因此当前结果应称“generated waypoints executed with local motion planning and target-aware terminal completion”。不能写无RRT、纯原始轨迹跟踪，或仅凭同样后处理就排除后处理贡献。实际有没有触发RRT需日志证明；不能因代码含fallback就断言某次触发了。

同样检查env termination是否在宽松success时立即抛EpisodeFinished，导致某方法在严格成功前早停。必须统一终止/保持协议后比较严格口径。

## 二、补实验优先级

### P0-1 迁移轨迹与末端控制的贡献分离（最优先新执行）

四方法{no_adapt,rigid,source,target} × 相同目标/条件/区组，分别运行：

A. 仅生成轨迹。禁用goal_pose末端补偿，采用明确的轨迹跟踪；如果仍以40 waypoint规划，明示该执行器并记录实际偏差/RRT触发。
B. 仅沿生成末端轴的有限延伸。保持生成末端姿态，不重新读取真实孔目标姿态；延伸距离和终止条件由独立pilot冻结。此组分离录制参考提前停止造成的深度缺口。
C. 当前target-aware完整目标补偿。保存补偿前后的位姿、成功、角度修正、位移、步数及接触代理。

所有组从配对初态运行，使用相同物理允许的终止/保持规则。若需要先修正参考末端，统一在nominal准备阶段完成并冻结，不能按测试目标表现调补偿。

当前C组240可保留为开发结果；A/B各240是完整追加预算，但若发现终止协议有误需统一重跑C。先在预注册平衡子集做调试，正式参数用新区组确认。至少增加只做共同抓取后直接交给terminal controller的对照（3×4×5=60）可测其独立求解能力；不要只凭成功率差就断言结构是成功的原因。

主要报告三层：计划轨迹、实际补偿前轨迹、最终结果。输出mm/deg错误、深度/横向误差和成功保持时长。严格阈值属于操作性判据，须验证几何依据，不能称接触真值。

### P0-2 对齐的非循环检验（先用已有数据）

a. 保留原物理时间、共同位置事件和轴向位置曲线。
b. 完成原方案未交付的弧长敏感性，原地旋转不能丢掉；所有进度定义预先固定。
c. 相位映射只从训练数据/独立标记/目标0°参考构造，冻结后评估留出±5/±10。没有办法从0°参考确定姿态适应时机时，应承认额外映射输入不可获得，不能偷用目标倾斜轨迹。
d. 比较共享模型+受限映射与各模式模型的留出误差及剩余响应差异。响应阈值对齐仅作明确标记的oracle描述上限。

若只保留“模式依赖且可执行迁移”的主张，不要求此实验一定证明canonical模型存在。若要把canonical作为论文中心，该检验必需。

### P1-1 结构消融（只有要解释成功机制时必需）

使用相同初末增益的平滑ramp、过渡前移/后移、时间反转或阶段错配曲线，与冻结源律比较。改变过渡而保留终端=1，能分离末端对齐和阶段结构的价值；记录可达性/碰撞，不能通过严重不连续制造弱基线。

所有方法使用相同终端处理，最好在A/B组也评估。若source不能优于ramp，应写“紧凑可用响应模型”，不称精确学习过渡必不可少。6维生成器选择性用已有多生成器实验支撑，pitch消融不能替代。

### P1-2 独立复验与统计

先修复协议再增加独立区组。5区组探索结果保留；样本数根据差异/等效容限与区组变异设计，不机械认定520条就足够。报告按条件和区组配对的成功不一致对数、差异区间；多方法/多阶段控制比较范围，帧不能充当独立样本。

### 按论文主张选做

- 只称仿真多模态边界，不必须重采全部1500六维数据。
- 要称完整6维跨策略一致，必须在多模式独立激励这些通道；已有pitch不足。
- 要称sim-to-real执行迁移，必须实际真机执行和基线；离线人类示教相似不能代替。
- 不必新增VLM或在线phase estimator；当前离线任务无需承担额外在线识别主张。

## 三、可直接采用的结论段落

> Generator relevance describes the intervention response of a specified execution process rather than a controller-independent physical invariant. Across controlled demonstration modes, transition timing and location vary with strategy, whereas the tested responses show similar endpoint behavior under shared terminal control. Frozen source models remain useful for cross-mode reproduction within the evaluated execution pipeline. Whether these responses admit a shared phase-indexed representation remains a separate hypothesis requiring independently specified progress mappings and held-out validation.

待P0执行消融完成后，按实际结果替换within the evaluated execution pipeline的具体描述。不要提前补入独立末端控制、统计等效或真机成功等尚无证据的句子。
