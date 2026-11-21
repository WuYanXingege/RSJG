# 复现指令与停止边界

固定起点 `54ef175c3cb2601c60ec178402831cee406eec6e`，独立工作区 `RSJG_SSEC_identity_20261121`；本次结果根目录为相邻 `RSJG_EBJD_assets/experiments/ssec_candidate_identity_20261121`。所有命令只读历史资产，不启动训练。

在本工作区复核已提交收据：

```bash
../.conda/rsjg/bin/python scripts/ssec_identity/check_delivery.py
```

完整复现须在原RSJG主机、另一独立worktree中，设置一个**尚不存在**的输出目录，不能删除或覆盖旧验收资产。脚本只接受相邻experiments下的`ssec_candidate_identity_*`目录。需要原冻结bank/Provider与各历史实验才能核对资产，不能用随机数据替代其身份检查。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export RSJG_SSEC_IDENTITY_OUT="$(pwd)/../RSJG_EBJD_assets/experiments/ssec_candidate_identity_replay_001"

../.conda/rsjg/bin/python scripts/ssec_identity/common.py
../.conda/rsjg/bin/python scripts/ssec_identity/localize.py
../.conda/rsjg/bin/python scripts/ssec_identity/cpu.py
../.conda/rsjg/bin/python scripts/ssec_identity/mechanisms.py
../.conda/rsjg/bin/python scripts/ssec_identity/gpu.py
../.conda/rsjg/bin/python scripts/ssec_identity/interfaces.py
../.conda/rsjg/bin/python scripts/ssec_identity/noise_contract.py
../.conda/rsjg/bin/python scripts/ssec_identity/full_regression_gpu.py
../.conda/rsjg/bin/python scripts/ssec_identity/summarize.py
../.conda/rsjg/bin/python scripts/ssec_identity/check_delivery.py
```

GPU命令顺序执行，避免并发负载污染成本对照。主成本对照为同参数的原SSEC入口与修复入口；`full_regression_gpu.py`额外复用上轮完整GPU脚本，输出独立子目录，覆盖v2/v3/SSEC/可选CSEE、全部人数及最密57人图。

`localize.py`从固定Git对象加载原耦合源码，仅用于同初态重放，不修改旧worktree。`interfaces.py`复用原生分组反向与update=0状态测试，重复执行会保留已有快照、另建attempt目录，不覆盖历史权重。早停/Top5动态训练分支不通过本轮禁止的optimizer.step重测。

本轮使用RTX5070Ti、PyTorch2.8.0+cu128、FP32，确定性CUDA设置；参数/源码哈希见GPU收据及冻结配置。`summarize.py`保留旧门槛名称，失败时写BLOCKED并退出2。不得通过忽略异常、删除失败案例或重标身份继续开训。

没有可执行的正式训练命令；`python -m bcep_scene_energy.interface`仍拒绝。READY仅表示当前冻结协议内的工程检查通过，未授权本轮任何训练或测试集评测。
