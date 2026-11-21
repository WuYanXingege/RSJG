# 零更新复现与唯一受控命令

本工作区位于RSJG主机的 `RSJG_SSEC_formal_ready_20261121`；解释器为相邻`.conda/rsjg/bin/python`。新运行目录和验收目录分别为：

- `RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_20261121`
- `RSJG_EBJD_assets/experiments/zara2_ssec_energy_stage2_readiness_20261121`

历史权重、候选库和本轮恢复夹具留在主机，不推送GitHub。

只验证交付一致性：

```bash
../.conda/rsjg/bin/python scripts/ssec_ready/check_delivery.py
```

以下为零更新验收命令，顺序运行；必须保留原冻结资产。每个脚本启动时禁止所有optimizer.step：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

../.conda/rsjg/bin/python scripts/ssec_ready/cpu.py
../.conda/rsjg/bin/python scripts/ssec_ready/real.py
../.conda/rsjg/bin/python scripts/ssec_ready/recovery.py
../.conda/rsjg/bin/python scripts/ssec_ready/prepare.py
../.conda/rsjg/bin/python scripts/ssec_ready/dev.py
../.conda/rsjg/bin/python scripts/ssec_ready/runtime.py
../.conda/rsjg/bin/python scripts/ssec_ready/finalize.py
../.conda/rsjg/bin/python scripts/ssec_ready/report.py
../.conda/rsjg/bin/python scripts/ssec_ready/check_delivery.py
```

`prepare`只能准备本次固定配置，不覆盖不同配置；已有正式`joint/last.pt`时拒绝重新准备。恢复验收重复执行会另建attempt目录，保留旧夹具。运行后将重新写本轮诊断结果，不能用来重写历史实验。源码/数据哈希不符时应停止，不能修改SHA绕过。

**本轮未执行任何train命令。** 唯一未来训练命令在 [CONTROLLED_COMMAND.txt](CONTROLLED_COMMAND.txt)，绑定本工作区和冻结文件SHA；仅在另一次明确授权后执行。技术中断用同一命令恢复本次`last.pt`，不创建第二个独立训练。完成/早停后入口返回，不启动其他折、seed、测试集或旧队列。

`--frozen-sha`使用 [CONFIG_SHA256.txt](CONFIG_SHA256.txt) 中的`file_sha256`，不是同文件另列的canonical hash。运行时激活另写授权记录，冻结配置与本轮零更新收据不变。
