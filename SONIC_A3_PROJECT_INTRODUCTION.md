# SONIC on Zhiyuan A3

## 项目介绍

更新日期：2026-10-14
性质：个人研究与仿真实验

## 项目目标

本项目要把 NVIDIA SONIC 的 whole-body control 思路迁移到智元 A3，建立一条从动作表示到物理控制的 A3 原生链路：

```text
A3 / 人体参考动作
        ↓
Retarget 与动作数据
        ↓
A3 Encoder → 64D motion token
        ↓
Kinematic Decoder / Dynamic Decoder
        ↓
29 维 A3 控制动作
        ↓
Isaac Lab 物理执行
```

最终目标是让同一个 A3 控制系统能够稳定站立、行走和复现全身动作，并在基础能力可靠后接入篮球任务。当前工作仍以仿真验证为主，不把运动学播放等同于物理控制成功。

## 已经完成

- 整理并验证 A3 的 29-DoF 关节名称、顺序、限位、默认姿态和 Isaac Lab/MuJoCo 映射。
- 明确 Isaac Lab 用户配置层的 stiffness/damping 直接使用 SI 单位 `N·m/rad` 与 `N·m·s/rad`；不能再额外乘以约 57.3。
- 建立 A3 motion 数据、retarget 和 GUI 检查流程，筛除了明显不稳定或不适合训练的参考动作。
- 原生训练并独立验证 A3 Encoder 与 Kinematic Decoder；64 维 token 能重建摆臂、腰部重心转移等 A3 动作。
- 私有实验环境曾使用通过合法项目渠道获得的 A3 行走控制器作为 teacher，同时增加 64 维 token 接口并统一到 29 维 A3 action；公开仓库不分发该控制器、部署配置、数值契约、rollout、训练数据或相关模型权重，也不代表厂商背书。teacher-derived 数据或模型的对外发布范围仍待确认。
- 修复 teacher 桥接中的根朝向丢失、转向步频、action scale 和 checkpoint 权重漂移，使监督数据与物理执行保持一致。
- 用 6 个训练命令和 6 个未见命令完成旧协议闭环测试，并在后续严格评估中记录每个环境的首次终止，不再用全局步数上限代替成功判据。
- 在技术层面建立稳定的物理 teacher，并完成 clean、push 和未见命令 rollout 采集；首批数据共 `46,800` 个 transition。相关数据和衍生模型不在公开仓库分发，对外发布须符合适用条款。
- 完成原生 Dynamic Decoder 的随机初始化离线蒸馏、五轮 DAgger 数据聚合和保守在线 PPO 微调。
- 建立统一训练契约：`640D encoder input → 64D FSQ token`，再与 `930D term-major proprio history` 合并输出 `29D action`。
- 建立完整序列 holdout、已知/未见命令双测试集和闭环优先的 checkpoint 选择流程，避免用低离线 loss 代替物理成功。
- 通过参数审计确认训练只改变 A3 Dynamic Decoder；Encoder 与 Kinematic Decoder 始终保持冻结。
- 废弃在错误执行器配置上训练的旧 step700 主线，用正确 stiffness/damping、action scale、root yaw 和历史观测契约重新训练并选定 step8000 backbone。
- 在冻结 step8000 backbone 后加入零输出、15 关节受限 action adapter，并用带 checkpoint/action-contract 哈希的 schema-v2 runtime 数据完成两轮 DAgger。
- Round2 在六个基础动作上 seed0 达到 `6/6`，多种子达到 `23/24`；它是当前受保护的 A3 episodic locomotion 技术基线，但其训练数据来自上述私有 teacher，相关数据或模型的对外发布范围仍待确认。
- 完成 20 动作 Round3 泛化与选择性 DAgger 对照；候选均未超过 Round2，因此按闭环门槛拒绝，没有覆盖成功模型。
- 固化 G1 与 A3 Encoder 的共同输入契约 `10×64=640D` 和 `2×32=64D` FSQ token，完成 joint-order、帧率、配对 motion 与 checkpoint 哈希审计。
- 在 Round2 schema-v2 闭环状态上训练小型 G1→A3 latent adapter；10,110 帧验证集的 token cosine 从 `-0.0566` 提升到 `0.9554`，最终 action cosine 达到 `0.99986`。
- 将冻结 G1 Encoder、latent adapter、A3 Dynamic Decoder 和 Round2 action adapter 接入同一实时推理链；严格复测发现 `forward_040` 在 frame 529 提前终止，因此该分支只保留为离线 walking-domain 对齐实验。
- 修复私有实验环境的 MuJoCo state/history bridge 后，受保护 Round2 在基础动作四秒清单为 `52/54`、未见速度/慢转/曲线四秒清单为 `54/54`，即四秒合计 `106/108`；按三秒可达多初态门为 `108/108`。这些是缺少公开权重、原始日志和专用 evaluator 的自报实验记录。

## 当前结果

当前正式结果是 Round2 A3 episodic locomotion 模型。它不是对 G1 decoder 的简单裁剪，也不是公开版对外部模型的直接调用：A3 Encoder/FSQ、正确配置 step8000 Dynamic Decoder 和 Kinematic Decoder 构成冻结 backbone，最终由一个受限 action adapter 根据实际 runtime 状态与 token 修正 15 个腿部/腰部关节。

现阶段的学生结构是：

```text
冻结的 A3 Encoder / 64D token
            +
930D A3 本体历史状态
            ↓
冻结的正确配置 step8000 Dynamic Decoder
            ↓
15-joint bounded action adapter
            ↓
29D 物理控制动作
```

Round2 对站立、`0.2/0.3/0.4 m/s` 前进和左右 `0.3 rad/s` 转向在 seed0 达到 `6/6`，四个种子共 `23/24`；原未见速度、慢转和曲线为 `15/24`。它也在默认、关闭和双倍观测噪声下分别完成 `20/24`、`21/24` 和 `16/24`。

外部 walking core 不在学生推理链中，但曾作为冻结 teacher 和对照基线。Round2 是当前 reset-per-episode reference tracking 技术基线。MuJoCo 四秒诊断为 `106/108`，三秒可达多初态门为 `108/108`；这些初态来自本地 rollout，其来源权利范围尚待书面核实，也不代表任意随机关节姿态、连续命令导航或 sim-to-real 已完成。

## 证据与授权边界

公开仓库没有发布 Round2 权重、原始日志、rollout manifest 或专用 MuJoCo evaluator，因此上述数字目前不能由第三方只靠本仓库独立复现。涉及外部 teacher 的训练、蒸馏和实验结果均作为暂定历史记录保留，不构成“已获授权”声明。对外介绍不得只写“100%”或“108/108”，必须同时说明四秒 `106/108`、三秒可达状态门槛、公开证据缺失和未完成真机验证。

在 Round2 不变的前提下，当前又建立了 walking-domain 的跨 embodiment token 路径：

```text
冻结的 SONIC G1 Encoder
            ↓
64D G1 latent → 小型 latent adapter → A3 FSQ token
            ↓
冻结的 A3 step8000 Dynamic Decoder
            ↓
Round2 bounded action adapter
            ↓
29D A3 物理控制动作
```

这条链证明了在现有 walking runtime 数据上可以离线逼近 A3 token/action，但不能据此宣称 token 已经“通用”。第一版只用理想 reference 训练时不能覆盖真实闭环姿态误差；改用 schema-v2 runtime 状态后离线结果提高，但严格在线复测仍在 `forward_040` 提前终止。当前结论因此限定为尚未通过在线验收的 walking-domain alignment 实验。

当前 checkpoint：

`GR00T-WholeBodyControl/a3_locomotion/checkpoints/clean_step8000_adapter_dagger_round2_v1/best_closed_loop.pt`

SHA256：`9dd32a55fa29fdfbba41cbbcb92c400a019a4899fb4a9993002c67fe352da4ec`

## 当前采用的方法

- **正确配置重新起步**：错误配置 step700 只保留作对照；正式链路从可审计的正确配置 step8000 开始，不能用旧模型的短期表现替代物理正确性。
- **冻结 backbone、局部修正**：adapter 零初始化并限制残差幅度，只更新 15 个腿/腰关节；加入 adapter 前后行为等价，训练不会直接改写 Encoder、Kinematic Decoder 或完整 Dynamic Decoder。
- **精确 schema-v2 契约**：runtime 数据保存 adapter input、token、teacher/student/executed action、motion/frame/done，以及 source checkpoint、frozen backbone 和 action contract 的 SHA；训练前验证源 adapter 可重放。
- **学生状态上的两轮 DAgger**：Round1 先进入基本闭环分布；Round2 再从 Round1 实际到达的状态查询 teacher，并补充 `forward_020/030/040` 定向数据，解决高速前进的累计偏差。
- **动作均衡而非帧均衡**：即使困难前进轨迹包含更多失败帧，也按 motion id 等概率采样，避免少数动作支配梯度并遗忘站立/转向。
- **闭环选模与拒绝机制**：离线 MAE 只负责监控，checkpoint 必须通过已知/未见多种子闭环。Round1 后续 PPO、Round3 普通 DAgger 和选择性 DAgger 都因闭环退化被拒绝。
- **先审计再对齐**：对齐前固定 joint-name 语义、50 Hz 采样、640D window、Encoder/FSQ 权重和 motion split；不能因为 G1/A3 都是 29 DoF 就假定关节语义相同。
- **分布匹配优先**：对齐器只更新 16,576 个参数，G1/A3 Encoder、A3 decoder 和 Round2 均冻结；训练数据来自真实学生闭环状态，避免用理想 reference 的低 loss 掩盖 runtime distribution gap。
- **逐层验收**：依次检查 latent/token、冻结 A3 decoder action、Round2 最终 action，再运行 650 步在线物理测试；只有 token cosine 提升不能作为完成标准。

## 下一阶段

Round2 继续作为 `A3 episodic locomotion baseline v1` 冻结。基础/未见 exact-state manifest、四秒 `106/108` 诊断结果和三秒 `108/108` 有界门槛结果一起冻结。walking-domain G1→A3 token adapter 作为未通过在线验收的独立实验保存，不允许覆盖 Round2 或被描述为新的 locomotion policy。任何 teacher-derived artifact 的公开使用或发布须先完成权利核实。

`forward_040` 三秒后的累计平衡误差属于独立长时问题，不能用来否定已经通过的三秒门。连续命令切换、任意随机初态、push 和动力学恢复仍应分别建立 curriculum，不能把多个新问题一次叠加后归因于旧模型。

下一步是从许可合规的 BONES-SEED G1 数据中筛选更丰富动作，并为每个源动作生成物理可行的 A3 配对 reference。扩展对齐必须同时保留当前 walking 闭环结果，并增加 gesture、reach、squat、transition 等独立 holdout；只增加 G1 数据而没有 A3 对应动作不能训练真正的跨 embodiment 对齐。篮球任务继续暂缓，等 locomotion、恢复能力和 whole-body physics 分别通过验收后再接入。

具体试验过程、数值结果和 checkpoint 状态见 `SONIC_A3_PROJECT_PROGRESS.md`。
