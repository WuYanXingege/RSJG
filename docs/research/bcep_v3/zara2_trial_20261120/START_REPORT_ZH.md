# BCEP–EBJD v3 唯一 ZARA2 Stage2 启动收据

**已按用户明确授权启动，训练仍在后台运行；这不是最终效果报告。** 快照时间 2026-11-20T11:34:37.770605+08:00。最新记录真实更新 **45**，当前第1轮尚未完整结束；首个可恢复checkpoint（update1）的model、AdamW step计数、scheduler、RNG和运行身份已核验。

运行源码固定 `2e96b6f35554ceceb5a04fc08b6259dac75e486b`，冻结配置SHA `294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43`。新增文件仅为旁路资源记录、结果分析和发布；原源码/配置manifest逐项未变。原Provider和seed2035 train/dev bank、seed3201初态、v3结构、final_coordinate损失、六层桥、K20/20步、优化器/调度及早停/Top5规则均保持。

CSEE、HCRA、GRG-CTF已出现实际参数更新，逐模块计数见[START_RECEIPT.json](START_RECEIPT.json)。已记录277个旧资产SHA，v2五折与candidate_cost仍暂停，sharedworld仍COMPLETE，没有启动它们或测试集。

原生早停/100轮上限完成后，后台收尾将以H-JADE rank1作唯一主结果，核验三项固定门槛，并与旧epoch5作590窗配对比较，发布最终REPORT_ZH.md、逐轮与匿名逐窗CSV、Top5、配置和复现命令。技术中断仅可恢复本次last；没有last时不会另起训练；明确暂停会保留暂停。不会自动扩折、换seed、消融或评测test。

每轮耗时沿用原控制器。显存是每5秒采样的本次PID GPU used_memory（含context/reserved），可漏掉瞬时峰值；未采集PyTorch allocator峰值，不将采样值宣称为精确峰值。

本次实际使用的唯一命令：

```bash
RUN_ONE_ZARA2_V3=true OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python -m bcep_v3.formal train --frozen-sha 294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43
```

[预注册判定](PREREGISTERED_DECISION.json)；最终结果只有原生完成后才生成。

结果分析脚本的5项只读测试通过，覆盖旧归档与当前冻结配置的split/bank/权重身份、同窗口配对、反馈差值分解、资源采样口径及未完成时拒绝生成最终结论。旧归档采用`identity.train_bank/dev_bank`字段，本次采用`bank_sha256`字段，分析脚本显式映射并验证哈希相等。
