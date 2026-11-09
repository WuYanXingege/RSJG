# EBJD 终点条件门控联合 DiT：主干与层数升级

状态：`IMPLEMENTED_INTERFACE_VALIDATED_NOT_ACCURACY_VALIDATED`。
已实现8层研究主干及6层归因对照，完成130项CPU测试和真实SDD完整更新/20×20采样检查。
正式六层模型、已有权重及结果保持原身份。实现提交：
`1392916720348ebc0aa1c9f0049f969bd56cd035`。

## 1. 设计判断与层数选择

上一轮加速度桥改的是轨迹表示。本轮直接改GCJD的网络块与条件注入方式，
三臂都保持Brownian桥和原CTF，因此不会把表示变化混入主干比较。

| 设置 | 原正式主干 | 新同深度对照 | 新研究主干 |
|---|---|---|---|
| 配置臂 | legacy6 | gated6 | gated8 |
| 粗去噪＋CTF后精修 | 3＋3 | 3＋3 | 4＋4 |
| 宽度/heads/FFN | 192/4/768 | 192/4/768 | 192/4/768 |
| 块内残差分支 | temporal/social/cross/FFN | 相同4分支 | 相同4分支 |
| 调制条件 | 时间＋历史，共用scale/shift | 每分支独立，时间＋历史＋world终点 | 同左 |
| 残差门 | 无独立门 | tanh有界门，零初始化 | 同左 |
| 注意力Q/K | 原点积 | 按head维RMS归一化 | 同左 |
| 总可训练参数 | 7,587,805 | 9,397,789 | 11,768,797 |

8层由4个coarse块和4个refinement块组成，CTF仍位于二者之间且只计算一次。
这意味着32个块内残差子层，不意味着整个模型只有8个算子；历史GRU、两个历史社交块、
地图U-Net、配对head和预测head都继续存在。

选择8层是相对6层的小幅深度扩展，以便增加CTF前后的处理能力，
不是证明6层不足或8层最优。12帧短时间轴、人数分布和地图计算成本都不支持仅凭图像DiT的
深度经验直接把主干堆到24层。gated6是必要对照：legacy6→gated6检查块设计，
gated6→gated8检查额外深度。

新8层参数比原6层约增加55.1%，新6层约增加23.9%。
该方案不是等参数替换，也没有现成的提速结论。

## 2. 已实现的主方法

针对每个场景b、world p和行人n，构造：

```math
c_{bpn}=e_t+h^{hist}_{bn}+E_g((g_{bpn}-b_{bn,T})/s_g).
```

每层根据c为四个分支分别产生shift、scale和gate：

```math
u_j=(1+scale_j(c))\,LN(h)+shift_j(c),\qquad
h\leftarrow h+\tanh(gate_j(c))F_j(u_j).
```

四个F分别是latent-slot temporal attention、同world的social attention、
每人的history/map cross attention，以及原GELU FFN。归一化后的Q/K仍按head维缩放；
observed bias和原 `alpha²×future_geometry_bias` 继续加到social logits。

原实现只有入口的零初始化proposal embedding；新块额外有独立的终点条件投影，
让G/H对应终点直接调制每一层的四类信息更新。
训练桥仍使用与GT路径一致的GT终点条件，推理条件来自冻结G或已学习配对H；
新块本身不读取未来真值、不做候选筛选或碰撞修复。

核心假设是：在不同噪声水平、历史状态和可靠终点下，各类更新需要不同强度。
这种逐world、逐行人的分支门控比给所有分支共用同一调制更有表达能力。
它是否改善当前任务仍需训练验证，不能从结构表达力直接推导ADE收益。

采用adaLN-Zero式初始化：每个新块初始为identity，首步主要学习门投影，
门打开后注意力/FFN开始获得任务梯度。定向测试确认8次optimizer更新内四类分支与几何bias
均获得非零梯度；完整模型测试也检查新 `endpoint_condition` 投影的非零梯度。
tanh只约束残差系数，不构成整个网络的Lipschitz稳定性或安全性证明。

## 3. 前沿依据与创新边界

这不是把已有方法换名字称作新理论：

- [DiT，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)
  提供条件adaptive normalization和identity初始化的依据。
- [Scaling Rectified Flow Transformers，ICML 2024](https://arxiv.org/abs/2403.03206)
  提供QK RMSNorm用于注意力训练稳定性的相关实践；没有照搬其图像任务结论或优化器参数。
- [Scaling Diffusion Transformers Efficiently via µP，NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/1bf3dbbd6346f50627e2ab1795f90435-Paper-Conference.pdf)
  表明diffusion transformer的结构和缩放仍在发展；它不能证明本项目的8层一定优于6层。

本项目的研究结合点是：可靠终点条件驱动所有因子化分支的门控，同时保持联合配对的
边缘候选集合、world隔离和CTF可微几何。AdaLN、QK归一化本身不属于本项目首创。
没有完成对该具体组合的全面新颖性/专利检索，因此当前只能称方法候选。

方法上保留已验证的“可靠终点→联合配对→条件桥扩散”分解、VP/v-prediction和20步DDIM。
没有同时替换噪声过程、loss、终点提供器或pairing机制。
这种选择让新主干的研究问题可以归因，现有联合与碰撞收益也不会因方法重写而失去对照。
上一轮最小加速度桥保留为独立候选，本轮预置配置不将二者合并。

## 4. 兼容性和验证

- 完整CPU测试130项通过。
- 额外从父提交读取原denoiser代码对照：默认legacy state、forward和梯度逐值一致。
- 新6/8层：完整forward/loss/backward/optimizer正常；attention/FFN和地图路径保留梯度。
- gate打开后的行人/world置换、scene/world隔离、padding均通过。
- activation checkpoint开/关的输出和梯度一致，重算闭包读取正确的终点条件。
- 旧/新架构和6/8层checkpoint不匹配会拒绝strict-load；runner在恢复状态前检查架构签名。
- 语义配置包含 `block_type` 和两阶段层数，它们不是execution-only优化。

真实SDD训练数据检查保留原batch=4场景、累积=4和256×256地图；每个臂2次完整临时update，
加原路径重复对照共8次。覆盖有效人数1–9，所有臂的输入/增强/时间/噪声hash对应。
原路径重复的参数差为0。四类attention/FFN均有非零梯度；各臂都完成了
一个真实2人场景的20-world/20-step G/H采样，输出有限、固定终点和候选集合保持。

CPU检查墙钟167.04秒，包含输入hash、train-only尺度拟合和额外采样，不能用于宣称训练提速。
本轮GPU成本0，正式训练更新0；未启动长期训练。有效计算精度为CPU FP32，
尚无CUDA/BF16的性能或精度验收。

另外，同一个seed不等于不同结构的所有共同参数初始值相同：构造不同深度会消耗不同RNG，
后构造的coupling/head初始化也可能不同。当前检查证明接口和训练路径正常，
未来精度对照应明确匹配共同模块初始化，或准确登记为seed-matched训练。

完整机器收据保存在资产目录，路径与SHA256见本目录 `SUMMARY.json`。

## 5. 可执行配置与入口

新主干：`ebjd/gated_dit.py`；集成：`ebjd/denoiser.py`、`ebjd/model.py`、`ebjd/config.py`。
研究配方：`configs/research/endpoint_gated_dit_v1.yaml`。

已生成三份绑定实现提交的配置，均从现有SDD完整合同派生并使用独立输出目录：
资产目录 `experiments/ebjd_endpoint_gated_dit_v1_20261109/prepared_{legacy6,gated6,gated8}/RESOLVED.json`。
使用现有统一训练runner即可运行；没有替换完成的正式实验配置。
新结构需要主体重新初始化，仅复用核验过的同折地图U-Net，不能将旧6层完整权重当作可续训状态。

重新生成配置（输出目录需为新路径）：

```bash
PY=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
ASSETS=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets
BASE="$ASSETS/experiments/sdd_jmm_unified_coupling_bridge_v1_20261103_155713"
"$PY" scripts/prepare_endpoint_gated_dit.py \
  --base-config "$BASE/00_identity/RESOLVED_SDD_JMM_UNIFIED.json" \
  --arm gated8 --output "$ASSETS/experiments/new_gated8_run"
```

复现真实数据检查：

```bash
"$PY" scripts/check_endpoint_gated_dit.py \
  --config "$BASE/00_identity/RESOLVED_SDD_JMM_UNIFIED.json" \
  --train-bank "$BASE/05_proposal_banks/train/gdts_proposal_train_seed3101.npz" \
  --output "$ASSETS/experiments/new_gated_dit_check" --updates 2 --device cpu
```

## 6. 当前明确结论

已经交付一套能执行、能训练、能20×20采样的现代化主干，以及明确的层数控制配置。
尚未证明它降低minADE/JADE、改善CR或缩短训练时间。
唯一优先下一步是固定数据与训练合同，对legacy6/gated6/gated8做可归因训练比较；
不再同时增加新模块或继续扩宽网络，也不把8层直接改写为已验收的论文主模型。
