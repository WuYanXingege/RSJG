# 复现与边界

基点`ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86`；审计代码以本分支最终固定提交为准。Python环境：RSJG根目录`.conda/rsjg/bin/python`。依赖版本见PHASE_RECEIPTS.json。

仅在新的独立worktree执行；生成目录必须不存在，不覆盖旧验收。以下命令执行**本用户已授权范围的11次合成更新**，没有正式ZARA2训练命令：

```bash
export PYTHONPATH="$PWD/scripts:$PWD"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset RUN_ONE_ZARA2_SSEC
../.conda/rsjg/bin/python -m ssec_nonzero.audit prepare
../.conda/rsjg/bin/python -m pytest -q tests/test_ssec_nonzero_audit.py --junitxml=docs/research/bcep_scene_energy/synthetic_nonzero_recovery_20261121/CPU_TESTS.xml
../.conda/rsjg/bin/python -m ssec_nonzero.audit run
../.conda/rsjg/bin/python -m ssec_nonzero.audit compare
../.conda/rsjg/bin/python -m ssec_nonzero.report
```

prepare验证固定正式源码/配置SHA并生成全合成样本；run为每阶段启动独立进程并拒绝已完成阶段的额外更新。若技术失败，保留现场并检查已有OPTIMIZER_EVENTS；不得无审查重复启动产生额外更新。

当前合成checkpoint不能供正式ZARA2训练初始化。正式入口仍要求另行授权与其原冻结SHA，旧零更新收据仍保持原样。本轮不得执行正式CONTROLLED_COMMAND。
