# 正式入口准备与验收命令

在本worktree根目录使用同级 `.conda/rsjg/bin/python`。本目录脚本不启动正式训练。

```bash
OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python -m bcep_v3.formal prepare
OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python -m pytest -q tests/test_bcep_v3_formal.py tests/test_bcep_v3.py tests/test_bcep_v2_formal_controller.py -o addopts=''
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_ready/readiness.py real
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_ready/readiness.py recovery
OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python scripts/v3_ready/readiness.py dev
```

real/dev明确禁止optimizer.step；recovery仅在内置合成数据上执行4+2+2=8次更新，后续真实数据梯度检查不step。已有合成checkpoint时不得把目录清空重试或当新运行使用；真实准备目录和合成验收目录完全分离。正式训练唯一命令由脱敏报告提供，必须等待用户另行授权后执行。
