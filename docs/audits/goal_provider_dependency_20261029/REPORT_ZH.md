# EBJD 可靠终点提供器依赖核对

## 直接结论

1. **当前 G 使用哪些模块？** 必需的学习模块只有 `goal_module` U-Net（encoder、decoder、head）。运行链路还需要既有输入预处理、TTST（1000 点采样后聚为 20 个中心）、`down_factor=8` 和 scene 像素到米制坐标变换。`_legacy_encode` 后续调用的历史编码器不影响 `goal_point`；轨迹 `diffnet`/DDIM 去噪器不参与 G 的生成。
2. **Goal-pretrain 能否直接替代 matched 权重？不能。结论为 `NOT_EQUIVALENT`（ZARA1、ZARA2 均如此）。** 两者虽然具有完全覆盖且可 strict-load 的相同 46 个参数键，但 matched GDTS 基线训练更新了全部 46 个张量；Goal-pretrain-only 回放的热图与世界坐标候选在所有 64 个配对记录中都不一致。
3. **是否需要完整 GDTS 轨迹去噪网络？不需要。结论为 `EXTRACTABLE`。** 独立 C 仅保留 613,772 参数的 Goal U-Net、TTST 和显式坐标转换；不实例化历史编码器或轨迹 denoiser。在两折各 16 个窗口、两个随机种子上，C 与 A 的热图、候选数、候选值和原始顺序全部逐值一致。
4. **250 轮 GDTS 训练属于什么依赖？** 它是当前资产的**权重来源/训练历史依赖**，不是部署时的完整网络结构依赖。本次实际绑定的 selected matched checkpoint 分别来自 ZARA1 epoch 160、ZARA2 epoch 140；本核对不能证明计划的 250 轮在理论上不可缩减，只证明现有 Goal-pretrain 权重不能原样替代 matched 后续训练得到的 Goal U-Net 权重。

## 核对对象与权重

| Fold | Goal-pretrain | Selected matched GDTS | Goal tensor 覆盖 | 完全相等 |
|---|---|---|---:|---:|
| ZARA1 | epoch 57, `c5266fad...a5fd7` | epoch 160, `34af0f43...bf6d7` | 46/46 | 0/46 |
| ZARA2 | epoch 30, `50175952...14db` | epoch 140, `bba3e538...2d80` | 46/46 | 0/46 |

ZARA1 参数张量最大绝对差为 `3.6271190643`，ZARA2 为 `3.1687924862`。Goal U-Net 使用无 BatchNorm 的卷积块，因此未发现 running buffer；影响候选的学习状态就是上述 46 个参数张量。

## 固定回放结果

回放使用 CPU/FP32；每折 16 个固定 validation loader 窗口，覆盖单人、多人和较密集窗口；随机种子为 3101、2035；K=20。每次 A/B/C 从完全相同的 Python、NumPy 和 Torch RNG 状态开始。候选在乘 `down_factor` 并执行同一 scene 世界坐标变换后比较，预设阈值为 `1e-6 m`。

| Fold | 记录数 | A/A 可重复 | A/B 完全一致 | A/C 完全一致 | A/B 有序候选最大距离中位数 | A/B 集合匹配最大距离中位数 |
|---|---:|---:|---:|---:|---:|---:|
| ZARA1 | 32 | 32/32 | 0/32 | 32/32 | 4.117600 m | 3.881792 m |
| ZARA2 | 32 | 32/32 | 0/32 | 32/32 | 3.722973 m | 3.245000 m |

- ZARA1 A/B：热图 logits 最大绝对差范围 `66.2212–287.6252`；有序候选最大距离范围 `2.0094–10.3872 m`。
- ZARA2 A/B：热图 logits 最大绝对差范围 `68.8245–464.9300`；有序候选最大距离范围 `2.0914–6.8794 m`。
- A/C：两折所有记录的 logits 最大绝对差均为 0，候选有序最大距离和集合匹配最大距离均为 0；候选顺序没有发生置换。

因此不能用“小样本 minFDE 接近”替代等价性：B 在候选本身已经明确改变。C 则达到了比 `1e-6 m` 更严格的逐值相等。

## 后续入口建议与边界

后续正式入口可以采用本目录 `minimal_goal_provider.py` 所验证的结构：以 matched checkpoint 中明确提取的 `goal_module.*` 权重 strict-load 到独立 U-Net，然后运行固定 TTST 和坐标转换。正式化时应把数据预处理与 scene transform 接口一并迁入论文工作区，并绑定 checkpoint/配置哈希。

**当前 ZARA2 实验保持原绑定，不中途替换任何资产。** 本轮没有修改其源码 HEAD、配置、proposal bank、预算或进程。

证据边界：本结论只覆盖当前 ZARA1/ZARA2 资产和现有实现；没有外推其他 fold。没有训练、调参、optimizer update 或 GPU 调用；没有重新生成正式 proposal bank。

## 产物

- `DEPENDENCY_GRAPH.md`：真实调用与权重来源链路。
- `PARAMETER_COMPARISON.csv`：92 行逐参数比较（两折各 46 行）。
- `PARAMETER_SUMMARY.json`：checkpoint、epoch、哈希和覆盖摘要。
- `REPLAY_RESULTS.json`：64 条逐窗口、逐 seed 回放结果及身份记录。
- `MINIMAL_PROVIDER_MANIFEST.json`：最小运行模块、参数量、配置与 A/C 复现结论。
- `audit_goal_provider.py`、`minimal_goal_provider.py`：可复核 CPU 脚本与独立原型。
