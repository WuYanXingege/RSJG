# 检查、暂停与恢复

本次交付固定源码 `51cdc5e84c3601b75482efeebea374f7ffa260f9`，训练配置中的 `allow_real_training` 为 false。以下命令从 v2 工作区执行。`RUN_PLAN` 指向原五折的只读运行清单，`TEST_OUTPUT` 指向独立私有合成验收目录。

**只读核对。**

```bash
python -m bcep_v2.workflow inspect --plan "$RUN_PLAN" --fold zara1 --device cpu
```

核对 `dry_run=true`、`allow_real_training=false`，以及源码、配置、split、成员和几何身份。旧运行的取消收据与原 RUN_PLAN 分开保存。

**合成测试入口。** 以下是复现命令。先查空闲 GPU，已有合法计算作业时退出。在全新的合成验收目录中，GPU 命令直至退出都在前台运行；同一轮故障复检使用原目录并通过 `--checks` 选择受影响阶段，其账本累计时间和步数。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q -p no:cacheprovider tests/test_bcep_integrated_v2.py tests/test_bcep_v2_readiness.py

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout -k 5s 1170s python -m bcep_v2.check_readiness --plan "$RUN_PLAN" --output "$TEST_OUTPUT"
```

CPU 全套每次含 8 次 Linear 合成更新。一次无故障 GPU 全套需要 24 次合成更新、10 次独立形状前后向。复检要先从累计预算扣除已经发生的调用和耗时，同步缩短外层 `timeout`。主机中断造成 `active=true` 时须先核对未结算的时长与步数。GPU 恢复检查只使用带 TEST_ONLY 身份的合成窗口、候选和临时 checkpoint。数据集成员核对不读取真实未来。验收收据位于 `TEST_ONLY_GPU.json`。

**暂停语义。** 将 SIGINT 或 SIGTERM 发给当前已核实的单阶段 v2 进程。导出完成正在处理的窗口，再退出。训练在已提交逻辑更新边界暂停，正在进行的验证会跑完整个验证并提交 epoch、selector 和 scheduler。退出码 75 表示可恢复暂停，调用方应结束阶段链；其他错误退出同样不能推进。检查独立的 `EXPORT_PAUSED.json` 及相应 bank 的 `COMPLETE.json`。后者才是该 bank 可供下游读取的凭证。

**恢复语义。** 如将来另行授权真实运行，须使用原 v2 来源、配置、split 和 bank 身份。导出再次执行同一阶段和同一 seed/view，按行校验文件哈希和成员，跳过已提交行，补齐其余行。训练须由 `last.pt` 加 `--resume` 和明确的绝对 `--until-update` 续接。阶段身份分别绑定冻结 density、候选缓存、冻结 head 及精修 bank。每个阶段按既定验证指标排名，同分按较早轮次优先；完整 epoch 快照、`TOP5.json`、`best.pt` 和可恢复的 `last.pt` 各有明确含义。

训练状态使用 `bcep-ebjd-integrated-state-v3`，它是断点格式版本。较早格式缺少部分轮次的 raw sums/counts，严格恢复会拒绝它。参数级 strict-load 仅核对张量接口。相同 float32 下游表示中，零初始化候选修正与基础候选逐值一致；原始 float64 KMeans 文件另有自身的字节身份。旧五折暂停断点属于旧代码与旧计划，其恢复需要单独授权。本轮结束状态为旧队列取消、新版训练关闭。
