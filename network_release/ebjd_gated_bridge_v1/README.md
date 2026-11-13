# EBJD 终点条件门控联合桥：干净源码快照

这是修改后网络的独立、可搬运源码目录。默认研究结构为 `gated8`；
`legacy6` 和 `gated6` 是必要的归因对照。**8 层已完成接口检查，尚未证明精度或速度优于原模型。**
现有 ETH/UCY 五折和 SDD 正式结果属于 `legacy6`，不能写到 `gated8` 名下。

论文工作区的 `ebjd/` 仍是唯一开发源。本目录是可校验的交付快照，不是另一个 worktree；
更新应通过 `scripts/export_clean_network.py` 导出到新位置，不能分别维护两套算法。
没有数据、权重、bank、旧实验日志、缓存或旧开发目录链接。

## 入口与文件

| 文件 | 用途 |
|---|---|
| `network.py` | 构造 3 个网络配置、G/H 同噪声配对采样 |
| `ebjd/unified.py` | 可学习双射配对、Sinkhorn 替代梯度、配对损失 |
| `ebjd/model.py`、`encoders.py` | 主模型、MSCE、地图 U-Net 与历史编码 |
| `ebjd/gated_dit.py`、`denoiser.py` | 新门控 block、旧 block、粗预测和 CTF 精修 |
| `ebjd/representation.py`、`sampling.py` | 桥白化、固定终点、VP/DDIM |
| `ebjd/goal_provider.py`、`proposals.py` | 独立原生终点提供器和身份绑定 bank |
| `configs/profiles.yaml` | `legacy6 / gated6 / gated8` 与有效训练合同 |
| `prepare.py`、`train.py` | 绑定已验证数据资产，执行现有统一训练与冻结评测 |
| `release_guard.py` | 源码校验、配置/CLI 一致性、恢复身份检查 |
| `check_network.py` | 无外部数据的 CPU 接口检查；明确使用合成输入 |
| `SOURCE_MANIFEST.json` | 来源提交、逐文件哈希、精确复制/提取清单 |
| `METHOD.md`、`ARCHITECTURE_REVIEW_ZH.md` | 当前方法定义与逻辑/创新评审 |

## 在现有环境中检查

从本目录运行，不设置旧工作区 PYTHONPATH，不安装/升级依赖：

```bash
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python -B \
  check_network.py --output /tmp/ebjd_clean_network_check.json
```

输出路径必须不存在。检查包含完整的临时 loss/backward/optimizer 更新、梯度、
模型与优化器状态恢复，以及 20 worlds / 20 steps 的 G/H 固定终点检查。
它只证明代码接线可用；32px 合成地图不构成真实 256px 数据、GPU 性能或精度验收。

Python API：

```python
from network import build_network, predict_g_h
model = build_network("gated8")
# 正式运行须绑定同折地图 U-Net，并从 train manifest 拟合尺度。
# batch 为 SceneBatch，proposals 为 [B,20,N,2] 米制原生候选。
# initial_noise 为 [B,20,N,12,2]，G/H 共享并按配对同步 gather。
model.eval()
result = predict_g_h(model, batch, proposals, initial_noise)
```

## 后续真实训练入口（本次没有启动）

先使用已经有同折初始化、baseline 和 30 轮统一 early-stop 合同的 resolved config。
下列变量由实际资产路径指定；没有内置猜测的 fold 权重或占位 baseline。

```bash
python prepare.py --base-config "$BASE_CONFIG" --profile gated8 \
  --config-dir "$NEW_CONFIG_DIR" --run-output "$NEW_RUN_DIR"

python train.py --config "$NEW_CONFIG_DIR/RESOLVED.json" \
  --train-bank "$TRAIN_BANK" --validation-bank-pattern "$VAL_BANK_PATTERN" \
  --output "$NEW_RUN_DIR" --source-commit "$UPSTREAM_COMMIT" \
  --fold "$FOLD" --training-seed 3101 --provider-seed 3101 \
  --compact-scenes --device cuda
```

`UPSTREAM_COMMIT` 取 `SOURCE_MANIFEST.json.upstream_commit`；
`VAL_BANK_PATTERN` 含字面量 `{seed}`，对应 2035–2039。
配置应按当前接受的实际 batch、累积、精度和运行时准备。
SDD gated8 可在 `prepare.py` 命令增加
`--denoiser-uncheckpointed-max-agent-slots 16`：单 world 固定终点训练、物理组
不超过 16 个槽位时保留去噪块计算图，较大组回退 checkpoint；地图 U-Net
checkpoint 不变。默认 0 保留历史执行方式。该运行时不改变 endpoint、loss、
参数键或有效 batch；不适用于推理 20 world 或显式指定 checkpoint 的 rollout。
实测范围与数值验收见论文工作区 `docs/performance/20261113_gated_runtime/REPORT_ZH.md`。
`prepare.py` 只规范化现有统一入口实际使用的 30 轮/AdamW/学习率字段，记录改动，
不会默默把旧 250 轮实验变成新试验；它要求输入已经具有 30 轮统一 early-stop 合同。
输出、配置、bank、权重全部保存在本目录外。

恢复同一新实验时给原命令增加 `--resume`，使用同目录完整 `last.pt`；
旧 6 层完整训练状态不能严格续训成新 8 层。只有同架构、同 release、同 resolved config
可以走该恢复入口。旧正式实验仍从原论文工作区入口恢复。

## 交付范围

模型/损失/采样源码按来源逐字复制；loader 与训练折尺度拟合按函数原文提取。
通用训练入口保留原训练、选模、early-stop、评测行为，只替换历史脚本 import 并增加 release 闸门。
完整路径需要既有同折数据/初始化/bank/baseline。在线 provider API 单独提供，
其原生全图预处理不等于 EBJD 的行人局部 crop，见 `DEPENDENCIES.md`。

训练参数量：legacy6=7,587,805；gated6=9,397,789；gated8=11,768,797。
冻结外部 provider 不计入这些数值。本快照不合并加速度桥、decoder-energy CTF 或其他研究候选。
