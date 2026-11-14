# 自有空间终点密度＋联合配对＋条件桥：实现报告

结论：**IMPLEMENTED / PREFLIGHT_PASS / NOT_ACCURACY_ACCEPTED**。
新结构已实现，28项定向CPU测试通过；最终版本通过8次真实ETH累积optimizer更新和两个验证窗口的20-world/20-step推理检查。
这不是训练效果验收，没有证明优于旧GMM、旧统一模型或GDTS。未恢复旧100轮训练，也未启动新长期训练。

## 设计问题与本次范围

旧scratch终点头把64个地图token平均后回归20个等权对角高斯，每个分量抽一个候选。联合配对只能排列候选，eta=0的条件桥不能改变终点，因此弱终点分布无法由后续8层桥修复。此前空间查询残差及覆盖损失短对照均未获得收益；本次不继续这两项补丁。

新链路：

```mermaid
flowchart LR
    X[观测轨迹与语义地图] --> E[共享MSCE编码器]
    E --> D[自有连续空间终点密度]
    D --> G[20个IID终点候选G]
    E --> C[学习联合配对]
    G --> C
    C --> H[保持逐人候选集合的H]
    E --> B[8层未来交互条件桥]
    H --> B
    B --> Y[20个共同world的完整轨迹]
```

保留192/4/768、4个coarse＋4个refinement块、CTF、EBD、原batch/累积/优化器、原混合精度、20/20推理及rollout/projection OFF。
本次没有证明8层优于6层；固定层数用于避免混入另一项结构变量。
所有预测网络参数，包括地图U-Net，从随机初始化训练；只复用数据、语义地图和经manifest核对的训练折尺度。
GDTS指标仍可用于既有selector比较，但不加载其预测权重、终点bank或optimizer。
语义地图沿用既有资产，不宣称地图资产生成也没有外部知识。

## 连续空间密度及越界处理

空间分支直接读取MSCE的32通道空间特征，按`align_corners=True`重采样到64×64；保留xy位置通道，以观测历史/社交向量作FiLM条件。输出4096个空间概率质量，不把空间特征平均后回归局部终点坐标。

定义地图中心c、半宽h（当前16m）、支持域D={g: max(|g−c|)≤h}。连续密度为：

```text
p(g|X) = (1−alpha(X)) q_in(g|X)  ；g∈D
         alpha(X) q_out(g|X)    ；g∉D
```

alpha为可学习的越界概率质量，不是扩散噪声系数。两部分支持域互斥，不能用越界分量解释地图内标签。

地图内，q_in是4096个固定中心、固定带宽的截断高斯核混合。质量通过spatial softmax学习；每个核均除以其在方形域内的准确CDF积分。sigma为网格间距的0.5倍，当前约0.254m。采样使用截断高斯的逆CDF，保留连续坐标，**不是先采样再夹紧边界，也不是只输出网格中心**。

地图外，采用四个不重叠方向区域（+x/−x/+y/−y）。令r=max(|g−c|)>h、u∈[−1,1]为区域内切向坐标；区域概率pi_j由观测条件预测，r−h服从可学习尺度b_j的指数分布，u均匀。坐标变换的Jacobian为r，因此：

```text
q_out(g|X) = pi_j(X) exp(−(r−h)/b_j(X)) / (2 r b_j(X))
```

四个区域覆盖全部外部平面，归一化为1，具有有限矩；不丢弃越界GT、不截断其坐标、不扫描尺度。b_j用训练折goal scale转换为米。域边界及区域对角线按固定规则归属，其概率测度为零。

NLL使用连续密度，按旧归一化坐标习惯加入`2 log(goal_scale)`的Jacobian。GT仅用于监督、诊断及指标，不参与生成或候选筛选。

实现阶段最初考虑了与局部密度重叠的全局Gaussian。短预检发现其初始质量约10%，却解释约85%的地图内标签，因此未保留该设计。最终代码改为上述互斥支持域，不增加新的辅助loss。早期临时收据留在资产目录，不作为最终架构验收。

## 分工、采样及梯度边界

1. **边缘分布**：连续NLL直接训练空间分布、域门控及越界分布。地图内标签有局部空间梯度；不会给越界形状参数梯度。越界标签反之。采样固定为20个IID样本，保留原抽样顺序，没有TTST、GT筛选或增大K。
2. **联合关系**：沿用原learnable coupling及ST Hungarian/Sinkhorn。采样结果不反传到密度头，pair loss训练配对器；共享context仍接收配对与桥梯度，不宣称梯度正交。没有采用失败的soft-min覆盖附加损失。
3. **路径**：沿用GT终点条件训练桥、生成时使用预测终点。该条件分解是合法的训练方式，不把它描述为泄漏或代码错误。CTF地图采样和主网络梯度保留。

这是**同一次训练、同一optimizer下的联合学习**，不是“离散候选采样全链可微”。不需要先训练并冻结Goal provider。

对每个人，H是G的双射排列。因此有限bank中逐人经验边缘集合严格保持，minFDE(H)=minFDE(G)。这不意味着minFDE随训练固定：G来自正在学习的空间密度。不能将这种有限集合性质扩大为已证明连续联合分布、严格copula似然或性能保证。

损失权重沿用NLL=1、pair=1、map=0.1、geometry=0.05、coarse=0.25。已有辅助12帧热图仍保留；终点生成直接使用新的空间密度，不能把辅助BCE下降当作终点覆盖改善。

## 验证结果与证据边界

| 项目 | 实际结果 |
|---|---|
| CPU定向测试 | 28项通过，包含旧模式回归 |
| 概率公式 | 独立Normal/CDF与区域Jacobian参考重算一致 |
| 采样 | 固定噪声复现、区域频率/径向均值检查、连续坐标及边界检查通过 |
| 数据几何 | 非正方形空间特征、平移、行人/world顺序及padding检查通过 |
| 梯度 | 密度头、coupling、U-Net、桥有限且非零；域内外监督分工检查通过 |
| 完整恢复 | 模型strict-load、Adam、RNG及下一次CPU更新精确重放通过 |
| 最终真实预检 | 8个完整累积更新，每update 16场景，单场景最多52人 |
| 真实推理预检 | 1人及2人两个ETH窗口；20world/20step；G/H有限且候选集合保持 |
| 参数量 | 全模型11,836,391；终点头67,594（旧GMM头90,128） |
| 最终预检资源 | GPU任务墙钟45.71s；峰值allocated 2147.79MiB；主进程峰值RSS约4439.84MiB，不含worker总和 |
| 本轮GPU总成本 | 134.34s，含两次初始原型检查；共24次临时更新，正式更新0次 |

最终8次预检中的空间熵仍接近均匀，这是随机初始化短更新的状态，不是已经学到有效终点。新增每update及每epoch诊断：域外概率质量、域外标签比例、域外解释比例和空间熵。epoch值沿用逻辑update/微batch归约，不冒充整个数据集按行人汇总的统计。

未做完整epoch、五推理seed或跨折性能评测，不报告提速倍数、性能提升或完整训练ETA。临时state不能用作正式训练起点。新head参数结构和候选采样分布都改变，不支持把旧epoch44权重伪装成同结构严格续训。

## 创新主张与下一步边界

可检验的研究主张是：**完整支持域的自有空间边缘密度、保持边缘候选集合的联合依赖学习、未来交互条件路径生成，能否分工学习覆盖与协调**。
空间概率图、FiLM、混合密度、截断分布和8层Transformer本身均不是可直接宣称的新算法。本轮未做系统先行工作检索，不声称全球首次或已达到强创新。创新与效果需要相对旧GMM、无配对G、及已有外部先验方案的明确对照支持。

下一步唯一建议：固定本结构，与旧GMM从零进行等预算有界训练对照，先判断完整验证集终点覆盖是否改善，再投入长期训练。新IID采样与旧“一分量一候选”的差异必须披露，不能把效果全部归因于空间编码。不得以几个临时更新的loss、两个窗口指标或更少参数代替验收。

## 文件、命令与运行状态

- 模型：`ebjd/spatial_density_goal.py`
- 研究合同：`configs/research/spatial_density_bridge_v1.json`
- 训练接入：`scripts/train_learned_goal_bridge.py`，保持旧合同默认路径
- 真实预检：`scripts/check_spatial_density_bridge.py`
- 测试：`tests/test_spatial_density_goal.py`
- 资产根：`RSJG_EBJD_assets/experiments/eth_spatial_density_bridge_v1_20261114_180827/`
- 最终验收：`preflight_domain_separated/PREFLIGHT.json`；临时权重只在该资产目录

在论文工作区，使用现有Python运行：

```bash
../.conda/rsjg/bin/python -m pytest -q tests/test_spatial_density_goal.py tests/test_learned_goal.py tests/test_goal_coverage.py tests/test_spatial_goal.py tests/test_unified_coupling.py
```

真实预检入口必须指定独立空输出目录：

```bash
../.conda/rsjg/bin/python scripts/check_spatial_density_bridge.py \
  --config ../RSJG_EBJD_assets/experiments/eth_spatial_density_bridge_v1_20261114_180827/configs/RESOLVED.json \
  --contract configs/research/spatial_density_bridge_v1.json \
  --scale-identity ../RSJG_EBJD_assets/experiments/eth_learned_goal_gated8_scratch100_v1_20261113_223533/training/RUN_IDENTITY.json \
  --output /tmp/ebjd_spatial_density_preflight_unique --updates 8 --device cuda
```

训练入口已可执行，但本轮未调用。新模式必须显式传`--stop-after-epoch`，不会因遗漏预算参数就直接运行100轮；原100轮LR时间轴不变。以下是未来有界试验的单臂入口示例，不是已执行实验，也不能代替配对对照：

```bash
../.conda/rsjg/bin/python scripts/train_learned_goal_bridge.py \
  --config ../RSJG_EBJD_assets/experiments/eth_spatial_density_bridge_v1_20261114_180827/configs/RESOLVED.json \
  --contract configs/research/spatial_density_bridge_v1.json \
  --output ../RSJG_EBJD_assets/experiments/eth_spatial_density_bridge_v1_20261114_180827/future_training_not_started \
  --source-commit "$(git rev-parse HEAD)" --stop-after-epoch 5
```

新输出从零开始；恢复使用同一合同与源版本、完整last状态及`--resume`。达到有界epoch后保存完整状态并停止，不自动追加训练。旧100轮任务保持停止，旧权重、正式结果与两个cleanup目录均保留。
