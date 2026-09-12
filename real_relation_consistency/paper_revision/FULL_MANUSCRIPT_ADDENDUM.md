# 完整稿复核与前次建议修正

依据完整附件840dd76e-ff2e-4f2c-931b-aac6c1ab9b9c，1177行，包含结论。2026-09-12。补充前次MANUSCRIPT_REVISION_AND_EXPERIMENTS.md；有冲突时本文件优先。未核验参考文献bibliography文件、外部文献、全部图像和所有原始数值；未改论文源文件或运行新实验。

## 1. 撤回或细化之前的缺失判断

- 第471–477行附近已明确nominal由训练干预初始化、交替估计、最终固定。不是遗漏；摘要“only P learned”宜解释为可迁移响应参数，而非声称训练全程无nominal估计。
- 第1026–1036行已有phase/time/SE3 arc-length/keypoint/position-DTW进度对照，且DTW仅offline。无需重复原benchmark；新四模式尚需独立事件与弧长的相应验证，旧同模式对照不能替代。
- 第1156–1164行已明确不主张universal controller independence，允许不同示范者或控制目标产生不同规律。保留并把四模式结果接在这里，不需从头改方法定位。
- 本文s已经是相位归一化进度s=(q+tau)/4，不是单纯时间。把s改名phi不构成理论升级。需要解释的是跨模式阶段语义/边界能否对应，以及剩余差异。
- 正文没有新四模式实验或97%结果，尚无硬件结果；这些是准备新增的证据，不能当作已出现在论文里。

## 2. 最重要的方法修改：A1–A3的逻辑身份

sec:law_transfer目前“Under A1--A3 ... P_b^tr=P_a”可保留为算法定义，不能读成A1–A3推出真实P_b=P_a。

建议在等式前后替换/补入：

> A1–A3 make the transferred operator geometrically and semantically well-defined; they do not guarantee equality of source and target responses. We define the transferred predictor by P_b^tr(s)=P_a(s), and evaluate its predictive and execution performance on the target. Differences in demonstration strategy or control objectives may still lead to P_b(s) differing from P_a(s).

不要加一个“假设源目标规律相同”的A4然后把迁移成功当该假设的证明。响应匹配应成为可检验目标，执行可行性与逐点匹配分开。

在J定义加z或简短说明fixed execution process即可；可在多模式节才显示z，避免全文复杂化。Gdagger J唯一性是给定J的坐标分解，不是策略独立性，增加一句说明即可。

## 3. 摘要与结论

摘要“across geometric, coordinate, and execution variations”需区分两类变化：保留原控制增益/规划器/速度/起始位置的有限稳健性，新增策略模式下过渡不保持。

建议：

> The recovered profiles remain stable under the tested geometric, coordinate, and execution-parameter controls. Deliberately varied demonstration strategies expose mode-dependent transitions, motivating a distinction between response invariance and executable transfer.

第二句只有四模式正式入文后加入。原90.2%与6.554到0.639的降幅在算术上相符，不需要因新实验删除；3.27/10.48应注明N=3 stress test，且四模型完整匹配单元缺失，不能同常规匹配比较混报。具体保留指标以原数据审计为准。

Conclusion将“different demonstrators may induce ...”从仅未来限制改为实际观察：

> Changes to the tested execution parameters preserved the dominant profiles, whereas deliberately altered alignment strategies changed the transition response. The transferable object is therefore a response model conditioned on specified relation semantics and progress conventions, rather than a controller-independent invariant.

再根据补偿消融的实际结果报告跨模式执行，不提前称纯轨迹成功。

## 4. RQ2新增模式依赖小节

现有6类gain/planner/speed控制54/54成功，以及table-to-pedestal控制有价值，不与四模式结果矛盾。前者主要改变执行参数/前段起始条件，后者主动改变适应时机/位置路径。相同任务的两种成功策略响应不同，并不推翻前者的有限稳健性。

保留RQ2，再新增：

> Execution-parameter robustness does not imply invariance to demonstration strategy. We therefore separately vary path geometry and alignment timing while keeping the physical task fixed.

加入4模式矩阵及各自相对0°reference的响应。主文保留物理事件下early/late差异及插入段近似；fig5c仅描述性附图，不用于证明canonical共同函数。a(t)=1-axis_err/abs(theta)另命名对齐分数。

不要写task constraint必然导致yaw不传播。表K2“released yaw is non-propagating”改为“non-propagating response under the tested circular policy”。几何probe证明yaw可被容忍，示范响应是否跟随yaw仍取决于策略。yaw失败需区别planner failure与经过诊断的不可达。

## 5. RQ4明确两个层级

完整稿表格已标五对T1–T5为SL，优点是透明。摘要/RQ4正文也应明确controlled state-level relation probes，避免读者以为drawer→plate等都是端到端机器人装配执行。

建议RQ4拆两个短段：

- RQ4a Matched-relation transfer in controlled state-level probes。保留五对、selector和error。需补充探针生成机制/参考响应来源，是否共享预置响应结构；共享结构时属于受控识别/重用验证，不能作为独立物理发现。
- RQ4b Cross-mode execution in physical simulation。加入四模式源律转移，列计划/实际/末端补偿不同层级结果。

重复支持任务中的相同E_alpha可能来自共享源律或构造，应说明，不以此指控数据错误。现表Pose MSE与全篇common E_task不同，补单位、归一化、聚合与换算；若可以统一则更清楚。

## 6. 额外实现一致性检查

eq:representation_metric使用SO3测地Log误差；当前phase_switch_se3_baselines.py的finite模型residual使用pose6差、欧拉角wrap与se3_to_metric，regularization是否含lambda_theta也要逐项核对。两者不一般相等。应报告实际优化目标并把测地距离作为评估，或修改实现后重跑；不能仅改文字说完全一致。

方法X是TCP，而多个已读pipeline读取peg，必须声明；固定抓持可关联但不能省略证据。C的原始offset/Euler到Lie-log转换需说清。

统计：正文说seed-first；图fewshot又用成功seed–subset cells的SE。其为descriptive已注明，但依赖子集的cell-SE不能当独立seed不确定性，最好改seed-first区间或双层重采样。N=3满秩条件不是所有模型含截距/非线性参数的充分可辨识性保证；已有excitation screen限定应保留。

相关性>=0.99不等于等效性。正文已有相位内相关与近常数段绝对差，保留；表全段0.990与文字变化相位0.982应明确不同范围。图注N=30 fits需澄清是30个训练干预还是30次独立fit。

## 7. 最小新增实验包，不重做全部论文

必做：
1. 现有四模式原始响应复核并补独立位置/弧长敏感性，明确与同模式RQ3进度实验的不同。
2. 迁移执行补偿消融：无goal_pose补偿、保持生成姿态的有限轴向延伸、现有完整目标补偿。记录每段轨迹与RRT实际触发；统一success早停规则。
3. 同初末增益的平滑ramp/过渡移位对照，检验学习到的过渡是否比简单补终端更有用。

条件性必做：
- 要称canonical shared law：独立/训练限定映射冻结后在留出响应上测试共享模型与模式模型残差；不能使用测试a的阈值再证明a相似。
- 要称真机迁移：独立真机执行及相同控制基线。当前全文是仿真论文，硬件不是上述方法定义的逻辑必需条件。
- 要称完整六维跨策略稳定：多模式独立六生成器激励。若只报告pitch边界，无需立即跑1500条。

优先修复归因和协议，再按所需容限/方差增加新区组。旧多模式的97%保留为目标感知末端辅助流程结果，不直接当离线原轨迹成功率。

## 8. 推荐论文结构

I Introduction：保留问题和三贡献，摘要收紧。
II Method：保留J、G、P、有限SE3、辨识；补模式条件化一句与A1–A3非充分保证一句。
III Experiments：RQ1原结构证据；RQ2原有限稳健性+新模式依赖；RQ3原fewshot和进度；RQ4a状态级跨关系探针+RQ4b四模式物理执行与补偿消融。
IV Conclusion/Limitations：写已观察模式差异，canonical为可选未来假设，不把在线相位估计列成必须实现。

不建议仅为了概念统一新增大篇canonical理论。新增四模式一个主图加一张执行消融表即可承担主要证据；详细阈值对齐、yaw挑战和所有失败日志放补充。
