# SONIC on Zhiyuan A3

## 项目进度

更新日期：2026-10-14
记录原则：按项目发展的因果顺序记录尝试、结果和结论；只有通过独立闭环检查的能力才算完成。

## 当前状态

| 工作项 | 状态 | 当前结论 |
|---|---|---|
| A3 资产、29-DoF schema 与控制接口 | 完成 | 关节顺序、限位、默认姿态、映射、增益单位和 action scale 已核对 |
| A3 motion 与 retarget | 完成首批 | 可支持运动学训练和 GUI 验证；不合格参考动作已排除 |
| A3 Encoder + Kinematic Decoder | 完成 | 64D token 的 A3 动作重建已通过独立测试 |
| 外部 A3 walking policy 接入 SONIC | 技术接口完成、授权待核实 | 历史 12 命令测试已记录；提供方是否允许生成数据、蒸馏、公开结果和衍生模型尚无可核验书面记录 |
| A3 token adapter | 完成一轮对照 | 接口有效，但当前数据上未超过零 token 外部 teacher 基线 |
| 稳定物理 teacher | 技术验证完成、权利待核实 | 外部 core + residual teacher 曾用于 clean、push、未见命令采集和 DAgger 在线标注；相关公开使用边界未确认 |
| 原生 A3 Dynamic Decoder | 完成正确配置重训 | 修正物理配置后重训至 step8000；纯 decoder 为已知 2/6，是当前冻结 backbone 而非最终控制器 |
| Schema-v2 DAgger 与 action adapter | 完成 Round2 | 冻结 step8000 backbone，经两轮在线数据聚合和动作均衡 BC，已知动作多种子达到 23/24 |
| Round2 episodic locomotion | 当前最佳 | 已知 6 动作 seed0 为 6/6，未见动作多种子 15/24；作为受保护基线进入下一阶段 |
| Round3 扩展泛化 | 完成并拒绝 | 普通与选择性 DAgger 均破坏已有动作，未超过 Round2，因此没有替换正式 checkpoint |
| 保守在线 PPO 微调 | 已验证并拒绝 | 在旧原生 decoder 与正确 step8000 adapter 上均未超过对应 BC/DAgger 起点 |
| G1–A3 token alignment | 离线实验完成、在线拒绝 | 同域离线 token/action 接近 A3，但严格复测中 `forward_040` 在 frame 529 提前终止 |
| MuJoCo sim-to-sim | 三秒有界门槛技术通过 | 四秒合计 `106/108`：基础动作 `52/54`、未见动作 `54/54`；三秒可达多初态门 `108/108`；公开仓库缺少独立复现材料 |
| Basketball integration | 暂缓 | 等基础 locomotion 和 whole-body physics 通过验收 |

授权与证据边界：私有实验使用了通过合法项目渠道获得的 A3 行走控制器作为 teacher；公开仓库不分发该控制器、部署配置、数值契约、rollout、训练数据或相关模型权重，也不代表厂商背书。teacher-derived 数据或模型的对外发布范围仍待确认，因此相关结果按有边界的历史实验记录处理。公开仓库未包含权重、原始日志、rollout manifest 和专用 MuJoCo evaluator；不得把三秒 `108/108` 单独概括成“100% 成功率”，必须同时给出四秒 `106/108`、可达 rollout 初态范围和未完成 sim-to-real 的限制。

## 发展过程与结果

### 1. 建立 A3 基础接口

我们先从 A3 配置和机器人资产中整理 29 个受控关节，验证 joint-name 驱动的顺序、关节限位、默认姿态和 Isaac Lab/MuJoCo 双向映射。最终说明确认：Isaac Lab 用户配置层的 stiffness/damping 已按 `N·m/rad` 与 `N·m·s/rad` 解释，不能把现有数值再乘约 57.3。早期“厂家以 N·m/deg 给出、必须先转换”的说法错误，现已撤回。这个修正不影响已经完成的纯运动学 Encoder/Kinematic Decoder。

### 2. 建立 A3 motion representation

我们从已有 A3 配置和仿真动作生成首批 motion，并逐个在 GUI 中检查。摆臂、腰部重心转移等动作能够正确播放；剧烈摇晃、弯腰碎步和不符合 A3 支撑能力的参考动作被排除。

随后原生训练 A3 Encoder 与 Kinematic Decoder。模型可以把 A3 motion 编码为两个 32 维 token，再重建为 A3 动作。Stage-2 独立重载测试记录为 `global_step=1000`、`a3_recon≈0.0005584`，且没有 NaN/Inf。由此确认了动作表示层，但它只证明运动学重建正确，不代表机器人能在重力和接触下执行动作。

### 3. 初始 walking 与 residual teacher 尝试

早期的 standing、reference tracking 和 residual teacher 能让机器人在 GUI 中做出部分目标姿态，但 walking 多次出现脚不离地、抖动或跌倒。排查中先后修正了错误增益、每帧回到默认姿态的 slew-rate、reference 每秒循环、脚部定义和不合格 motion。

这些实验解释了早期失败，却没有产生可靠的 walking rollout，因此旧 residual teacher 和旧 walking checkpoint 只保留诊断价值，不作为当前模型起点。

### 4. 使用通过合法项目渠道获得的 A3 walking teacher

私有实验环境曾使用一个通过合法项目渠道获得的 A3 行走控制器作为物理 teacher，并在 SONIC 侧增加 64 维 token 条件路径，最终统一为 29 维 A3 action。公开仓库不分发该控制器、部署实现、精确数值契约、rollout、训练数据或相关模型权重，也不代表厂商背书；teacher-derived 数据或模型的对外发布范围仍待确认。

第一次接入仍出现前进不足和转向异常。继续检查后发现四个隐蔽接口问题：

- rollout 转换器保存了 root quaternion，但 MotionLib 使用的 `pose_aa[:, 0]` 没有写入根旋转，导致 yaw 被清零；
- 纯转向/横向命令的 gait frequency 必须与外部 teacher 一致；
- action clip 已存在，但各关节的运行时 scale 没有完整复现；
- 所谓 exact checkpoint 曾被一次优化器更新和 weight decay 改动，首帧 action error 达约 `0.0466`，并通过 last-action feedback 逐帧放大。

修复 root rotvec、步频和 action scale 后，我们重新从外部模型恢复并冻结 teacher core，把 token projection 置零，再用逐帧等价性测试确认移植路径没有发生权重漂移。该恢复和契约检查代码仅保留在私有工作区。

### 5. 验收稳定的私有物理 teacher

修复后的 exact-action 模型在 6 个已训练命令上全部运行到 frame 649，只因 episode timeout 结束：站立、`0.2/0.3/0.4 m/s` 前进和左右 `0.3 rad/s` 转向均未提前跌倒。

随后使用 6 个未见命令测试插值和组合泛化：`0.25/0.35 m/s` 前进、左右 `0.15 rad/s` 慢转，以及 `0.3 m/s ± 0.15 rad/s` 曲线行走。6 组也全部运行到 timeout。

这在技术层面说明该 A3 行走控制器在私有实验环境中可作为可靠物理 teacher。它不等于纯原生 SONIC Dynamic Decoder 已完成，也不代表厂商对本项目背书；checkpoint 的正确行为仍依赖匹配的 observation history、robot type 和 action scale 契约。

该 teacher checkpoint 及其运行时契约不在公开仓库分发。

### 6. 验证 64D token 路径

我们冻结外部 teacher core，只训练 token projection 100 iterations。所有候选都保持 6/6 命令稳定，说明 token 可以进入动态策略而不破坏接口；但没有候选在当前 walking 数据上超过 token 为零的 teacher 基线。

因此，`a3_token_yaw_fixed_step050_candidate.pt` 只作为研究候选，主基线仍是 exact-action 模型。当前结论不是 token 无效，而是数据和目标尚不足以让 token 对强外部策略产生可测收益。

### 7. 训练纯原生 Dynamic Decoder

本节记录的是第一代原生 decoder 试验。它正确暴露了状态分布漂移问题，但后续已由精确 runtime teacher 数据、重新随机初始化的 decoder 和五轮 DAgger 继续推进；这里保留原结果作为方法演进依据。

为了避免离线特征与实际运行不一致，我们直接从 SONIC runtime 采集 teacher 特征。每帧输入为 `640` 维 encoder 特征加 `930` 维 term-major proprio history，监督目标为 `29` 维归一化 action。

数据包括 6 个已知命令共 3900 帧，以及 6 个整段未见命令共 3900 帧。训练/验证按完整命令分开，避免相邻帧泄漏。数据保存在：

- `GR00T-WholeBodyControl/sample_data/a3_native_teacher_features_v2`
- `GR00T-WholeBodyControl/sample_data/a3_native_dagger_v2`
- `GR00T-WholeBodyControl/sample_data/a3_native_dagger_v3`
- `GR00T-WholeBodyControl/sample_data/a3_native_dagger_v4`

离线结果持续改善：随机初始化的验证 loss/RMSE 约为 `0.7963/0.2868`；teacher BC 达到约 `0.01665/0.0561`；DAgger 80/20 达到约 `0.0141/0.0533`；50/50 数据达到约 `0.01276/0.0504`。

但纯学生闭环仍失败。首个 BC 模型在 frame 120–181 跌倒；表现最好的 DAgger v3 延长到 frame 124–233；后续更高 student 比例没有继续改善。说明单步 action 拟合已较准确，但小误差进入 last-action 和机器人状态后会离开训练分布，离线 loss 不能代表闭环稳定性。

当前最好的原生研究 checkpoint 是：

`GR00T-WholeBodyControl/a3_locomotion/checkpoints/native_dyn/a3_dyn_dagger_v3_best_research.pt`

### 8. 直接 PPO 试验

我们以原生 Dynamic Decoder 为起点，冻结 Encoder/Kinematic Decoder，运行 2048 环境的短程 PPO pilot。到 step 50 时，闭环失败范围变为 frame 113–209，没有超过 DAgger v3，因此训练已停止，checkpoint 标记为 rejected：

`GR00T-WholeBodyControl/a3_locomotion/checkpoints/native_dyn/a3_dyn_ppo_step050_rejected.pt`

当前没有需要继续等待的 Dynamic Decoder 训练进程。直接对完整 decoder 做 PPO 会同时破坏已学到的动作映射和稳定域，不再作为下一步主线。

### 9. 重建物理 teacher 数据与原生离线初始化

在执行器单位、action scale、root yaw 和本体历史契约全部修正后，我们重新建立 teacher 数据，而不是继续沿用第一代失败 rollout。稳定 teacher 使用冻结的外部 walking core 和已验收 residual checkpoint；在同一物理环境中采集：

- known clean：`3,900` transitions；
- known push：`39,000` transitions；
- unseen clean：`3,900` transitions；
- 合计：`46,800` transitions。

每帧保存 `640D encoder_inputs`、`64D token`、`930D native proprio` 与 `29D normalized action`。其中 proprio 必须保持 Isaac Lab observation manager 的 term-major 顺序：10 帧角速度、10 帧关节位置、10 帧关节速度、10 帧 last action、10 帧重力方向；相同的 `930D` shape 若错误排成 frame-major，离线 loss 仍可能很低，但 runtime 输入实际无效。

新的 Dynamic Decoder 从确定性随机初始化开始训练，A3 Encoder 与 FSQ quantizer 冻结。训练损失为 full-action MSE 加 5 倍腿部与腰部 MSE；验证使用完整未见 motion，不做相邻帧随机泄漏。最佳离线结果约为 validation loss `0.00546`、全关节 RMSE `0.02293`，但闭环在开启观测噪声和关闭噪声时都为 `0/6`。这一步再次确认：单步监督精度已经足够低，主要问题是学生进入自己造成的新状态后没有 teacher 标签。

### 10. 当前采用的 DAgger 方法

DAgger 不再只收集成功 teacher 轨迹。外部 teacher 与学生读取同一个 Isaac Lab 机器人物理状态；每个控制 tick 用学生当前 token 和实际状态查询冻结 teacher，然后记录：

- encoder input、FSQ token 和 native proprio；
- teacher raw/applied action、student action 和实际混合执行 action；
- motion/env/frame id、reward、done 与 timeout；
- teacher contract hash 与必要的运行时元数据，用于检查两套输入契约是否漂移。

执行动作采用 `β·teacher + (1-β)·student` 并通过统一安全裁剪。`β=0.25` 用于进入新分布时保持安全，`β=0.10` 用于让学生承担更多闭环状态。数据按轨迹保存；需要降低困难动作权重时使用确定性 stride 抽样，并强制保留终止帧，避免随机 frame sampling 破坏时间分布。

五轮结果如下：

1. known、`β=0.25`：`39,000` 帧；聚合训练后从 `0/6` 提升到 `4/6`。
2. known、`β=0.10`：`39,000` 帧；第 500 步保持 `4/6` 并降低跟踪误差。
3. 定向采集 `forward_030/040`：完整困难集过度代表失败状态，训练退化；改用 stride-5 的 `6,135` 帧平衡子集后，第 250 步达到已知 `5/6`。
4. 只采集 `forward_040`、`β=0.25`：虽然单动作首次失败延后，却使 `forward_030` 退化，因此拒绝。
5. 六个未见命令、`β=0.25`：`39,000` 帧；某 checkpoint 未见集达到 `4/6`，但已知集退化到 `4/6`，因此仍拒绝。

这里形成了两个关键技术结论：第一，DAgger 的价值来自“在学生实际状态上询问 teacher”，而不是简单增加数据量；第二，失败数据必须与原稳定分布平衡，困难动作占比过高会导致灾难性遗忘。最终选中的 DAgger checkpoint 为 round-3 step 250：已知 `5/6`、未见 `3/6`，唯一已知失败是 `forward_040` 约 frame 341。

数据与选模记录：

- `GR00T-WholeBodyControl/sample_data/a3_dagger_v1`
- `GR00T-WholeBodyControl/a3_locomotion/checkpoints/native_dyn_dagger_v1/best_closed_loop.pt`
- `GR00T-WholeBodyControl/a3_locomotion/checkpoints/native_dyn_dagger_v1/evaluation.json`

### 11. 保守在线物理微调

直接重新运行旧 PPO 配置会重复早期失败，因此在线微调采用了以下保护措施：

1. 冻结 A3 Encoder、FSQ quantizer 和 Kinematic Decoder，只允许 `a3_dyn` 与 critic 更新。
2. 先完全冻结策略，单独预热 critic 20 iterations；tensor 审计确认策略除探索噪声外没有变化。
3. 修复 warm-start 隐蔽问题：checkpoint 的 `std=0.05` 会覆盖配置中的小噪声。新增显式 `reset_loaded_noise_std`，将噪声固定为 `0.002`。
4. Actor 使用固定学习率 `1e-7`、单 PPO epoch、clip `0.05`、max grad norm `0.05`，禁用 entropy 增长和自适应升学习率。
5. 降低纯运动学 action target 的权重，重点使用 anchor XY、anchor linear velocity、joint/feet tracking、upright 与 foot-slip 等物理 reward。
6. 训练拆成小批次，每个 checkpoint 返回确定性闭环；训练 reward、单动作测试和离线指标都不能单独决定模型。

首批受控更新中，batch-1 step 5 保持已知 `5/6`；从它继续的 batch-2 step 10 将 `forward_040` 的 canonical 六动作测试失败点从约 frame 341 延后到 frame 436，同时保持未见 `3/6`。参数审计显示 policy 中只改变 `a3_dyn` 的 14 个张量和预期的 `std`，Encoder/Kinematic Decoder 未变化。该候选保存为：

`GR00T-WholeBodyControl/a3_locomotion/checkpoints/native_dyn_online_tune_v1/candidate_batch2_step10.pt`

随后从该候选运行 50 iterations、`1,228,800` sim steps 的正式续训，每 5 步保存一次。训练自然结束且无报错，但闭环筛选后没有 checkpoint 超过起点：最有希望的 step 40 在完整已知六动作测试中仍为 `5/6`，`forward_040` 约 frame 405，未见仍为 `3/6`。因此正式续训 checkpoint 全部不接受，当前最佳仍是进入续训前的 batch-2 step 10。

这轮结果说明：极小学习率与 critic 预热能避免立即摧毁策略，并能在早期改善高速直行时域；但仅依靠 PPO ratio clip 不能长期保护 teacher 的稳定动作映射，后续更新仍会在不同 motion 之间震荡。

### 12. 为什么废弃旧 step700 基线

后续审计确认，表现较好的旧 `step700` 系列是在错误的执行器物理配置上训练得到的。它包含 conditioned residual adapter、PPO 与 teacher regularization，因此在部分闭环测试中看起来优于新模型，但这种优势不能证明它更接近真实 A3。我们不再把错误配置 checkpoint 当作可延续的物理起点，而是保留它作为能力对照。

使用已经修正的 stiffness/damping、action scale、root yaw 和 observation/action 契约，从干净数据重新训练原生 Dynamic Decoder。动作均衡的 schema-v2 训练自然运行到 10000 steps；闭环筛选选择 `step8000` 作为干净 backbone。它在六个基础动作上的帧数为：

`[233, 649, 649, 634, 352, 400]`

即严格完成 `2/6`。这个结果低于旧 step700，但它的意义是建立了来源正确、可审计、可继续训练的基线。之后所有成功模型都冻结这一 step8000 backbone，不再混用错误配置 checkpoint。

### 13. 从正确 step8000 到 action adapter Round1

我们先尝试直接用 schema-v2 DAgger 更新完整 Dynamic Decoder。它解决了站立漂移并增强 token conditioning，但只达到 `4/6`，`forward_030/040` 仍受长时累计误差影响。为了避免继续改动已经学到的 29 维动作映射，转而在冻结 backbone 后增加一个受限 action adapter：

- 输入使用实际 runtime 的 `actor_obs + tokenizer`，而不是离线重建特征；
- 只修正 15 个腿部/腰部关节，残差最大绝对值限制为 `0.25`；
- adapter 零初始化，因此加入结构前后动作完全等价；
- 每份数据记录 source checkpoint SHA、frozen backbone SHA 和 runtime action-contract SHA；
- 训练前必须验证 checkpoint 能重现数据中记录的 student residual。

Round1 从 step8000 的零输出 adapter 出发，采集 `β=0.25` 和纯学生 `β=0` 的六动作 DAgger 数据。训练集 `37,023` 帧、独立验证集 `18,106` 帧，使用 motion-balanced sampling 训练 5000 steps。离线验证 MAE 从 `0.06008` 降到约 `0.03452`；闭环选择 step5000，结果为：

`[649, 649, 345, 291, 649, 649]`，即 `4/6`。

我们还完整复现了旧 step700 使用的 1000-iteration PPO 配方，但所有候选只有 `3/6`，因此全部拒绝。这证明 Round1 的缺口不是因为遗漏了旧优化器设置，也说明不能用“再跑 PPO”替代新状态数据。

### 14. 成功的 Round2 路径

Round2 的决定性变化不是更大的网络或更长的训练，而是在 **Round1 学生真实到达的状态上再次询问 teacher**。以 Round1 step5000 为唯一源 checkpoint，重新采集：

- 六动作 `β=0.25`：`20,107` 帧，24 条成功、12 条失败 episode；
- `forward_020/030/040` 定向 `β=0.25`：`17,028` 帧，12 条成功、24 条失败 episode；
- 独立六动作验证集：`10,110` 帧。

两类训练数据合计 `37,135` 帧。虽然前进定向数据更多，训练仍按 motion id 等概率采样，防止高速失败帧淹没站立和转向。训练继续冻结 step8000 backbone，只更新同一个 adapter；参数为 5000 steps、batch 1024、`lr=1e-3`、weight decay `1e-5`、gradient clip `1.0`。

离线验证 MAE 从 `0.02918` 降到 `0.01601`。但最终选择仍由闭环决定：step1500 在 seed0 首次达到 `6/6`，最低验证误差的 step4900 同样达到 `6/6`，因此选定 step4900。多种子复测结果为：

- 已知六动作：`23/24` 完成，完成率 `95.8%`；
- 未见六动作：`15/24` 完成，完成率 `62.5%`；
- 默认观测噪声：`20/24`；关闭噪声：`21/24`；双倍噪声：`16/24`；
- mild push：`14/24`；组合动力学随机化：`11/24`。

连续无 reset 的多命令切换和大范围随机初态属于新的组合/恢复课程，不作为当前 episodic reference tracking 的准入门槛；它们不能据此否定 Round2，但也不能把 Round2 描述为连续导航或 sim-to-real 已完成。

受保护的正式 checkpoint：

`GR00T-WholeBodyControl/a3_locomotion/checkpoints/clean_step8000_adapter_dagger_round2_v1/best_closed_loop.pt`

SHA256：`9dd32a55fa29fdfbba41cbbcb92c400a019a4899fb4a9993002c67fe352da4ec`

### 15. Round3 泛化试验与停止条件

为了检查 Round2 能否扩展到更多速度和曲率，我们另外建立 20 动作训练分布、6 动作新 holdout，并保留原有未见六动作不参与训练。期间发现容器内旧版 rollout converter 遗漏 `command`、`raw_policy_action`，且没有把 root orientation 写入 `pose_aa[:, 0]`；错误生成的数据与训练输出全部隔离，随后用正确 converter 重建并逐字段验证。

Round2 在新 20 动作集上为 `15/20`，新 holdout 多种子为 `11/24`。第一种常规 DAgger/BC 虽降低离线 loss，却最多只有 `12/20`、原六动作 `4/6`。随后只对五个失败动作保留 teacher target、对其余 15 个动作使用 Round2 自身 action 作为行为保持目标；选择性 DAgger 最佳仍只有 `12/20`、`4/6`，后期下降到 `10–11/20`、`3/6`。

这说明失败不是单纯缺少训练步数：共享 adapter 的参数会同时影响所有动作，修正某些速度/曲线时会破坏原本稳定的动作。Round3 所有候选因此拒绝，Round2 没有被覆盖。若未来继续提高陌生命令泛化，应使用 command/token-conditioned gating、分区 residual 或重新联合训练 Dynamic Decoder，而不是继续向同一个全局 adapter 叠加 DAgger。

### 16. G1–A3 walking-domain token alignment

Round2 接口冻结后，我们开始验证“同一动作语义能否从 SONIC G1 Encoder 迁移到 A3 控制器”，而不是用 G1 数据检查 A3 的关节结构。首先完成四项契约审计：G1/A3 都接收 `10×64=640D` future motion window、都产生 `2×32=64D` FSQ token、29 个关节按名称映射而不是按数组位置猜测、G1 release checkpoint 与 A3/Round2 checkpoint 均由 SHA-256 固定。修复了 50 Hz 输入被错误丢弃最后一帧的问题后，配对输入可以逐元素复现，但冻结 G1/A3 token 的初始 cosine 只有约 `0.067`，证明相同 shape 不等于 latent 已对齐。

第一轮 pilot 使用理想 reference window，同时送入冻结 G1 与 A3 Encoder，只训练一个 `64→128→64`、共 16,576 参数的 latent adapter。它在 motion-only holdout 上把 token cosine 从 `0.201` 提升到 `0.958`，但放到真实 schema-v2 rollout 后只有约 `0.292`，并且 action 略差于未对齐 G1 token。根因不是网络太小，而是理想 reference 把根朝向相对量规范化掉了，无法覆盖机器人实际跟踪误差。

因此 v2 不再添加网络结构，而是改用 Round2 真实闭环状态。训练数据来自 Round2 schema-v2 的学生状态：训练 `37,135` 帧、9 条轨迹；独立验证 `10,110` 帧、6 条轨迹。A3 数据中记录的 token 能由当前冻结 A3 Encoder 高精度复现，从而确认 checkpoint 与数据契约一致。相同 adapter 架构固定训练 300 epochs，最佳 epoch 为 293；验证结果为：

- token cosine：raw G1 `-0.0566` → adapted `0.9554`；
- latent RMSE：`0.585` → `0.166`；
- 经过冻结 A3 Dynamic Decoder 后，base action RMSE 为 `0.00707`；
- 经过 Round2 action adapter 后，final clipped action RMSE 为 `0.00491`、cosine 为 `0.999857`；
- 相对 native action RMS 的最终误差约 `1.81%`；
- adapted target MAE 为 `0.00924`，native A3 路径为 `0.00898`。

随后增加实时 `G1A3RuntimeAdapter`。旧测试运行到全局 `max_render_steps=650`，当时被误读成六动作均无提前终止；评估循环实际上会让已终止环境 reset 后继续。2026-08-26 使用逐环境首次终止诊断严格复测后，`forward_040` 在 motion frame 529 因 `anchor_pos_xy` 提前终止，XY 误差为 `0.4402 m`；同条件原生 A3 token 没有提前终止。因此旧的“六动作 650 步通过”结论撤回，G1→A3 v2 未通过在线控制验收。

closed-loop v2 在理想化 motion-only OOD 上 token cosine 仍约 `0.066`。现有证据只说明小型 adapter 能在给定的 A3 walking runtime 数据分布上离线逼近 A3 Encoder 输出；它还没有证明跨 embodiment 的通用动作语义。v1/v2 作为互补的分布失配实验保留，不进入正式 locomotion 主线。

### 17. MuJoCo sim-to-sim bridge 审计

第一轮 sim-to-sim 的失败不能评价 Round2，因为旧 runner 没有从 MuJoCo 当前状态重算所有 state-dependent observation，也没有完整恢复中途起点的机器人与历史状态。修复后的私有实验 bridge 从 schema-v2 数据恢复关节、root、状态历史、步态相位和上一实际执行动作，并从目标模拟器当前状态重算学生输入；每个窗口还与外部 teacher 做同初态正对照。该 teacher 的授权范围尚待核实；外部策略契约、专用 bridge、资产、checkpoint 和 rollout 均不在公开仓库分发。

在基础六动作的 54 个 early/middle/late 四秒窗口上，外部正对照为 `54/54`，未修改 Round2 为 `52/54`；两个失败是 `forward_040` 的 middle/late，分别持续 `3.64/3.70 s`。有限 adapter、PPO 和 Dynamic Decoder 微调均未超过 Round2，因而拒绝，没有覆盖受保护 checkpoint。

### 18. 未见动作 MuJoCo 检查与三秒多初态准入

2026-09-08 使用同一受保护 Round2 补做六个未见指令：`0.25/0.35 m/s` 前进、左右 `0.15 rad/s` 慢转，以及 `0.30 m/s ± 0.15 rad/s` 曲线行走。18 个 Isaac 环境重新生成 exact-state schema-v2 rollout；审计覆盖 `10,941` 帧和 `10,923` 个连续转移，previous executed action 误差为 0，历史状态最大误差约 `7.4e-7`。

每个指令使用三个独立环境和 early/middle/late 三个四秒窗口。Round2 与外部正对照从完全相同的状态启动，双方均通过未见清单全部 `54/54`。项目另行记录 **三秒可达多初态恢复门**：基础动作的两个四秒失败都发生在三秒之后，所以基础清单在三秒门下为 `54/54`；加上未见清单为 `108/108`。这些初态来自本地 rollout，其来源权利范围尚待核实，不是任意随机关节姿态；四秒合计 `106/108` 仍是必须同时披露的长时诊断，三秒指标不能单独表述为“100% 成功率”。

## 当前判断

项目已经形成一条技术上可执行的 A3 原生链路：A3 motion 与 64D token 表示、正确的物理/动作契约、外部 walking teacher、正确配置 step8000 Dynamic Decoder，以及两轮基于学生状态的 schema-v2 DAgger action adapter。由于外部 teacher 权利范围和公开复现材料尚不完整，这里不能把整条链描述为公众可独立复现。Round2 是当前最佳且受保护的 episodic locomotion 技术模型，不再使用错误配置 step700 或旧的 `5/6` online-tune candidate 作为主线。

G1→A3 v2 只完成同域离线拟合，严格在线测试未通过，不进入正式控制主线。MuJoCo exact-state bridge 已通过合同审计：原 Round2 在基础与未见动作的三秒可达多初态门合计达到 `108/108`；四秒诊断分别为 `52/54` 和 `54/54`。这项结果独立于 G1 token 支线。

Round2 已解决基础和局部未见参考动作在三秒可达多初态分布中的稳定执行，并完成首个受限 MuJoCo sim-to-sim 验收；它仍不是连续命令导航、任意随机初态恢复或 sim-to-real 策略。当前具体缺口是更长时间的累计平衡余量，而不是桥接或局部速度/曲率泛化尚不可用。

## 下一步计划

1. 冻结 Round2 checkpoint、配置和 SHA，作为 `A3 episodic locomotion baseline v1`；后续实验不得原地覆盖。
2. 冻结基础/未见 exact-state manifest、外部正对照、三秒 `108/108` 有界结果及四秒 `106/108` 诊断；任何候选必须完整保持这些结果，对外表述不得省略四秒结果和证据限制。
3. 将 `forward_040` 三秒后的失衡作为独立长时问题，不再仅用单步 imitation loss 推动共享 adapter。
4. G1→A3 v2 冻结为未通过在线验收的实验记录，不与 Round2 合并；后续对齐需使用真正独立的 G1/A3 同语义动作和动作级 OOD。
5. 扩展任意随机初态、动力学参数和延迟时建立独立 curriculum 与 split，不能把组合难度混入当前固定窗口结论。
6. 只有在物理 counterpart 与许可明确后才扩展 BONES-SEED；连续切换、push/recovery 和篮球任务仍分别推进。
