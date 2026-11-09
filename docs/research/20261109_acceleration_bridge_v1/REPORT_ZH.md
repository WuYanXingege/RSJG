# EBJD 统一架构审查与最小加速度条件桥实现

本轮实现一项独立研究结构：**初始速度一致的最小加速度条件桥**。
另外将单人场景的配对代理梯度修正为独立开关。状态为
`IMPLEMENTED_AND_INTERFACE_TESTED`；尚无新方案的训练后精度证据，正式默认仍是原桥。
实现提交：`be50a580120a290717410b8bdb4fdd2b36ca030e`。

## 1. 当前统一网络与证据判断

当前实现为冻结原生终点 G → shared MSCE → 可学习一对一联合配对 H →
GT 一致条件联合桥 → 20-world/20-step Raw 推理。
MSCE 包含历史 GRU、历史社交注意力及可训练地图 U-Net；桥为
192/4/768、3 coarse + 3 refinement FJAB。CTF 由粗路径运动特征、可微地图采样
和未来相对几何组成。训练期 rollout/projection OFF。

| 部分 | 实际作用与限制 | 本轮判断 |
|---|---|---|
| 冻结终点提供器 | 给出每人的20个终点；固定端点桥严格保留这些候选 | 保留。固定候选时，桥不能改善 minFDE 覆盖，也不能靠加宽网络解决 ETH 的该项问题 |
| 可学习联合配对 | Hungarian 保持逐人候选集合，Sinkhorn 提供代理梯度；同一排列 gather 桥噪声 | 保留。五折联合指标和 CR 已有正面证据，pair_loss 平台本身不是失效证明 |
| EBD 条件桥 | 线性终点修正 + Brownian 协方差白化 | 选择它作为唯一结构干预：其初始运动连续性先验较弱 |
| MSCE/FJAB/CTF | 联合条件路径建模，CTF 保留可微几何路径 | 本轮保持。白化槽与物理时刻并不等价；以前的 decoder_energy 没有获得可采用收益，不能包装成确定修复 |
| 训练监督 | 桥使用 GT 终点和与之对应的完整 GT 轨迹；proposal 用于配对监督 | 保留一致性。直接换成错误 proposal 终点而不改目标，会重现已发现的条件/目标矛盾 |
| 选模和证据 | 每折当前主要是一个训练 seed；SDD 正常早停11轮、选中7轮 | 不由汇总指标断言过拟合、碰撞因果或最优网络容量 |

已有证据定位了共享路径短板，但没有证明具体原因：

- ETH/UCY 五折：H 相对 G 的 JADE/JFDE 在25个 fold×推理seed配对中均改善；
  CRJADE/CRmean 宏平均分别降低1.4810/1.6496个百分点。
- HOTEL：G 的 minADE 已比 GDTS 高0.017132 m；H反而降低0.000396 m。
- SDD：H相对GDTS的 minADE/JADE分别高0.167859/0.036985 px，
  minFDE/JFDE则分别低1.484289/1.659168 px。H相对G的 minFDE完全相同。

因此主假设针对“终点较好但路径精度不足”，不替换已有有效的配对，也不声称找到了
HOTEL/SDD退化的已证实因果。

证据来源：
[五折正式结果](../../results/ebjd_eth_ucy_unified_fivefold_20261101/REPORT_ZH.md)、
[五折碰撞结果](../../results/ebjd_eth_ucy_jmm_collision_20261102/REPORT_ZH.md)、
[失败的能量汇聚试验](../../validation/ebjd_zara1_ctf_decoder_energy_20261024/REPORT_ZH.md)。
SDD逐seed原始结果及其路径/哈希登记在本目录 `SUMMARY.json`。

## 2. 理论改动：让终点修正具有运动惯性

令 s∈[0,1] 为4.8秒预测区间的归一化时间，b(s)=x₀+4.8s·v₀ 为原恒速参考，
δ=g−b(1)。原桥使用 b(s)+sδ：其连续初始参考速度额外改变 δ/4.8。
这只是参考均值的性质，不能据此断言训练后预测有速度错误。

新桥使用由白加速度驱动的位置过程：

```math
k(s,t)=\frac{\min(s,t)^2[3\max(s,t)-\min(s,t)]}{6},
\quad k(1,1)=\frac13.
```

对终点做高斯条件化，得到同一过程的均值与条件协方差：

```math
h(s)=\frac{k(s,1)}{k(1,1)}=\frac{3s^2-s^3}{2},\qquad
C(s,t)=k(s,t)-\frac{k(s,1)k(t,1)}{k(1,1)}.
```

这里 h(0)=0、h′(0)=0、h(1)=1。h(s)δ 同时是满足
q(0)=q′(0)=0、q(1)=δ，终端速度自由时，使 ∫‖q″(s)‖²ds 最小的参考修正；
自然边界为 q″(1)=0。

在11个内部时间点构造 L=chol(C)，继续使用原11个白化桥变量：

```math
y_{1:11}=b_{1:11}+h\,\delta+s_r Lz,\qquad y_{12}=g.
```

Cholesky以FP64构造，再按原策略使用FP32几何运算；未加jitter、未拟合核超参数。
归一化时间核的整体物理幅度沿用train-fold RMS尺度拟合。
原Brownian分支保留原算术，默认输出与梯度不变。

新均值的第一帧修正系数为0.0101273，原线性值为0.0833333。
这表示参考终点修正更逐渐地进入路径，不表示离散第一步速度被强制等于观测速度。
11维变换仍可逆，因此没有减少表达自由度，也没有对输出做平滑/避碰后处理。

GP运动先验和白加速度建模已有公开工作，例如
[GPMP2原论文](https://homes.cs.washington.edu/~bboots/files/GPMP2.pdf)和
[Differentiable GPMP原论文](https://homes.cs.washington.edu/~bboots/files/DifferentialGPMP.pdf)。
本轮创新候选在于将这一有边界解释的先验接入 EBJD 的可靠终点、联合配对和交互扩散桥，
不能把 integrated-Wiener 核本身宣称为首创，也尚未完成该具体组合的全面新颖性检索。

## 3. 独立修正：单人场景的代理梯度

原实现对N=1的hard assignment强制恒等排列，但ST backward仍使用学习的Sinkhorn矩阵。
单人best-of-K损失对合法排列不变；soft relaxation却可通过混合候选产生hard分支无法实现的方向。

新增 `training.coupling_singleton_backward: identity`，令单人/padding的soft分支也为恒等，
其配对分支对head/context的代理梯度为零；多人场景照常训练。
保留原pair_loss数值和场景权重，未删除单人训练数据。

默认值仍为 `legacy`。研究配置单独提供 `singleton_fix` 臂，
`acceleration` 臂仍用legacy，避免两项改动混合归因。

## 4. 实现入口与恢复边界

- `ebjd/representation.py`：解析条件统计、可选prior、checkpoint buffer family检查。
- `ebjd/config.py`、`ebjd/model.py`：新增语义类型
  `invertible_endpoint_acceleration_bridge_v1`，不归入execution-only exclusions。
- `ebjd/unified.py`：独立单人ST开关。
- 正式统一runner：记录prior与单人策略，恢复时检查语义；身份校验通过后才写RUN_IDENTITY。
- `configs/research/acceleration_bridge_v1.yaml`：三臂分别为control/acceleration/singleton_fix。
- `scripts/prepare_acceleration_bridge_research.py`：从有完整训练合同的fold配置生成独立resolved配置；
  拒绝dirty源码绑定、旧checkpoint输出目录和不完整的历史epoch/早停合同。

两种桥均为7,587,805个可训练参数，state_dict键与shape一致。
**same shape不代表语义兼容**：跨prior完整checkpoint加载会明确失败。
新桥应从主体随机初始化开始，只复用经过哈希验证的同折地图U-Net，并重新拟合train-only尺度。
不能把旧桥的optimizer状态直接接到新桥上称为严格续训。

已核对原SDD epoch7 `best_jade.pt`：
`d09d07a2b4e0f9da92acf2d2699687d5b29d8eb79c558f6aa98d9b04414beef4`。
旧默认仍strict-load成功；新prior明确拒绝加载该完整状态。原权重、bank、指标均未改写。

## 5. 已完成验证与使用方式

完整CPU测试116项通过，包括独立数值积分核验协方差、SPD/边界条件、可逆重建、
梯度、世界/行人置换、padding、旧分支精确一致性、单人代理梯度及配置身份。
一次首次全测因PATH缺少已有zstd命令失败，补上现有环境bin路径后全部通过；未安装新依赖。

SDD、HOTEL真实训练batch的完整临时更新与20/20采样检查见 `SUMMARY.json`。
每组分别运行control、control_repeat、acceleration、singleton_fix，各2次完整update；
逻辑batch=4场景、梯度累积=4、256×256地图，原样保留全部有效行人。

| 真实训练数据 | 窗口有效人数范围 | 临时updates合计 | CPU检查墙钟 | 原路径重复参数差 | 新桥20/20端点检查 |
|---|---:|---:|---:|---:|---|
| SDD | 1–9 | 8 | 142.46 s | 0 | PASS |
| HOTEL留一折训练集 | 1–43 | 8 | 586.99 s | 0 | PASS |

两组每个臂的原始输入、增强、时间和噪声hash均与control对应；主要模块都有非零有限梯度。
新桥的物理终点逐值等于proposal，G/H逐人候选集合相同。
CPU检查墙钟合计729.46 s（约12.16 min），包含额外重复对照、输入hash、尺度拟合与采样，
未计入全套pytest和代码开发时间。
仅这些临时模型被更新，正式训练更新数为0；本轮GPU成本为0。
发现其他项目正在使用GPU后，全部真实数据检查使用2线程CPU。
这些真实更新的有效计算精度是FP32；没有额外验证CUDA/BF16的训练数值或速度。

准备新研究配置示例（只生成配置，不启动训练）：

```bash
PY=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
ROOT=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets
BASE="$ROOT/experiments/sdd_jmm_unified_coupling_bridge_v1_20261103_155713"
"$PY" scripts/prepare_acceleration_bridge_research.py \
  --base-config "$BASE/00_identity/RESOLVED_SDD_JMM_UNIFIED.json" \
  --arm acceleration --output "$ROOT/experiments/acceleration_fresh_run"
```

可复现实数据短测（输出必须是新路径）：

```bash
"$PY" scripts/check_acceleration_bridge.py \
  --config "$BASE/00_identity/RESOLVED_SDD_JMM_UNIFIED.json" \
  --train-bank "$BASE/05_proposal_banks/train/gdts_proposal_train_seed3101.npz" \
  --output "$ROOT/experiments/acceleration_fresh_check" \
  --updates 2 --device cpu --max-seconds 720
```

这是完整update的接口/有限梯度检查，包含额外输入哈希和scale-fit开销，
不将它作为训练提速基准或ADE性能评测。更新边界检查预算，单次正在执行的检查可越过该目标。

## 6. 结论和唯一优先方向

实现与测试支持：新prior数学构造正确、可逆、可训练、保留终点集合；旧路径兼容。
当前不能宣称：新桥降低ADE/JADE、改善碰撞、提升速度，或相对原统一方案已有优势。
观测速度有噪声，且加速度核改变白化损失的几何权重；SDD拟合桥尺度从0.536668变为10.374629，
这不代表精度改善，反而需要注意注释噪声与高频误差的影响。新核条件数约3959，
旧核约57.7，FP64构造与重建检查因此必要。

下一步只优先做**同初始化、同数据/RNG、两臂的新桥与原桥训练对照**，重点看共同路径短板
HOTEL/SDD的ADE/JADE，并同时核验固定端点、JFDE和已有CR。
先单独检验prior，不合并单人ST修正、不扫描核或网络宽度。
本轮已准备可运行配置；没有自动启动长期训练或更改正式默认。
