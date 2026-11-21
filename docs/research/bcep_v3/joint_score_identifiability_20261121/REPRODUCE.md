# 复现命令与依赖

执行位置为 RSJG 根目录下本次独立 worktree，Python 为相邻 `.conda/rsjg/bin/python`。原资产只在训练主机，公开/远端提交不包含权重或 bank；没有这些固定资产时只能运行 CPU 原型测试和已提交 CSV 的汇总。

已完成的本次主机目录：

```text
RSJG_joint_score_audit_20261121/
RSJG_EBJD_assets/experiments/joint_score_identifiability_20261121/
```

只读重做汇总（不前向、不反传）：

```bash
../.conda/rsjg/bin/python scripts/joint_score_audit/summarize.py
```

只运行CPU数学原型与测试（没有optimizer）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.conda/rsjg/bin/python scripts/joint_score_audit/prototype_check.py
```

如需完整复现，应在相同固定资产旁新建 worktree，并选择**尚不存在**的独立审计输出名。入口拒绝覆盖已存在的审计目录，且只允许写入指定 experiments 父目录下 `joint_score_identifiability_` 前缀的新目录。不要删除原结果来重跑。

```bash
export RSJG_JOINT_SCORE_AUDIT_OUT="$PWD/../RSJG_EBJD_assets/experiments/joint_score_identifiability_reproduction_01"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

../.conda/rsjg/bin/python scripts/joint_score_audit/common.py
for model in v2_e5 v2_e26 v3_e12; do
  ../.conda/rsjg/bin/python scripts/joint_score_audit/collect.py --model "$model" || exit
done
../.conda/rsjg/bin/python scripts/joint_score_audit/analyze_scores.py
../.conda/rsjg/bin/python scripts/joint_score_audit/marginals.py
for model in v2_e5 v2_e26 v3_e12; do
  ../.conda/rsjg/bin/python scripts/joint_score_audit/gradients.py --model "$model" || exit
done
../.conda/rsjg/bin/python scripts/joint_score_audit/prototype_check.py
../.conda/rsjg/bin/python scripts/joint_score_audit/summarize.py
../.conda/rsjg/bin/python scripts/joint_score_audit/finalize.py
../.conda/rsjg/bin/python scripts/joint_score_audit/validate_delivery.py
```

`common.py` 首先核验固定checkpoint与上一轮v3评分回放；`collect.py` 在v3只采缺失特征，不重复已有评分。三个受控梯度进程是为获取原生计算图，不是新模型或训练。旧v2使用固定468c5edd工作区；v3使用本worktree的原始正式源码。

还需保留固定提交 `4c9eede551bebdd7634ab81b7b4bd790165c6bb0` 中旧v2的历史排列归档、`2c64c136438cac755b9a3d1723d5225a873dd707` 的既有v3分析CSV，以及训练主机的 `zara2_v3_frozen_feedback_audit_20261121/windows/`。这些都是身份依赖，脚本不会用随机初始化或其他候选替代缺失资产。

当前环境：Python3.10、PyTorch2.8.0+cu128、NumPy1.26.4；正式冻结前向及梯度在同一RTX5070Ti训练主机执行。CPU原型平台/版本与耗时见 `PROTOTYPE_CHECK.json`。改变硬件/库后若近并列评分或排列不一致，必须先查数值差异，不应跳过复现门槛。

所有复现命令都不调用正式训练入口、optimizer.step、旧队列恢复或测试集。汇总/验收会更新新worktree中的报告CSV和收据；不会覆盖历史权重、bank或训练状态。
