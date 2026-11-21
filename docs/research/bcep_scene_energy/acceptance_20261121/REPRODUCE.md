# 复现与冻结边界

独立工作区：`RSJG_SSEC_acceptance_20261121`，相邻Python：`../.conda/rsjg/bin/python`。结果与零更新状态测试快照留在 `RSJG_EBJD_assets/experiments/ssec_network_acceptance_20261121`。没有正式训练命令。

CPU内核验收：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.conda/rsjg/bin/python scripts/ssec_acceptance/run_cpu.py
```

已存在的验收目录中可重做以下只更新本轮诊断结果的检查；它们不会写历史训练目录。GPU性能A/B应单独运行，避免与其他负载并发。

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

../.conda/rsjg/bin/python scripts/ssec_acceptance/gpu.py
../.conda/rsjg/bin/python scripts/ssec_acceptance/mechanisms.py
../.conda/rsjg/bin/python scripts/ssec_acceptance/interfaces.py
../.conda/rsjg/bin/python scripts/ssec_acceptance/ddim_capacity.py
../.conda/rsjg/bin/python scripts/ssec_acceptance/summarize.py
```

最后一条目前**预期退出码2**，表示默认入口候选重编号准入失败、总状态BLOCKED，不表示训练发生过。不要忽略退出码继续开训。`interfaces.py` 的重复调用保留已有零更新快照，另建audit attempt目录，不覆盖历史权重。

首次在另一独立工作区准备新验收根目录时，指定一个尚不存在的输出目录，再运行准备命令。输出只允许位于原 `RSJG_EBJD_assets/experiments` 下，且名称须以 `ssec_network_acceptance_` 开头；准备命令拒绝覆盖已存在目录。以下变量须在后续验收命令中保持一致：

```bash
export RSJG_SSEC_ACCEPTANCE_OUT="$(pwd)/../RSJG_EBJD_assets/experiments/ssec_network_acceptance_replay_001"
../.conda/rsjg/bin/python scripts/ssec_acceptance/common.py
```

随后按顺序运行上面的CPU、GPU、机制、接口、DDIM和汇总命令。工作区须位于原RSJG根目录下；诊断报告写入该独立工作区，不覆盖原工作区。不要删除本次结果来复现。完整资产保护核对依赖训练主机上的原四组实验、前两轮冻结回放，以及固定Provider/bank；缺失时不能用随机数据冒充真实身份验证。

只核对已提交报告/代码清单：

```bash
../.conda/rsjg/bin/python scripts/ssec_acceptance/check_delivery.py
```

该命令验证交付自洽性及明示的BLOCKED状态，不把它转换成工程准入通过。

软件/硬件：PyTorch2.8.0+cu128，NumPy1.26.4，SciPy原`linear_sum_assignment`，RTX5070Ti；精确版本见GPU和CPU收据。网络源码哈希记录在GPU收据及冻结配置中，配置文件SHA和canonical SHA见`CONFIG_SHA256.txt`。

CPU测试、GPU前反向、拒绝代理对照均为合成数据且optimizer.step拦截计数0。官方样本仅selection_dev只读前向；正式输入适配器拒绝heldout/test用途。历史v2 e5/e26、v3 e12仅单独strict兼容加载，不能传给SSEC工厂作初始化。

边界：此发布没有可调用的正式训练入口，`python -m bcep_scene_energy.interface`直接拒绝；不恢复旧五折、candidate_cost或sharedworld。任何后续训练都需要先解决BLOCKED项、重新验收并取得独立授权。
