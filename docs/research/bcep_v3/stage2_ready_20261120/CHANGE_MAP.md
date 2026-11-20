# 正式入口最小差异

基础提交 `bfc51ba53b9ead6b5b1acd4996672c18e64905ef`。

| 文件 | 作用 |
|---|---|
| bcep_v3/stage2.py | 保留旧synthetic-only接口；增加接收经验证FormalInputs的正式回调，复用原joint_callbacks |
| bcep_v3/formal_inputs.py | 核对固定正式源码、train/dev成员、候选seed/权重/bank哈希；正式数据与bank薄适配器，拒绝test |
| bcep_v3/formal_validation.py | 原20步H推理与H-JADE selector；复用一次配对前向记录G/I/H终点指标和人数分组 |
| bcep_v3/formal_telemetry.py | 原AdamW的step后hook记录实际参数增量；真实验收期间提供禁止step的上下文 |
| bcep_v3/formal.py | 唯一ZARA2 prepare/train/status入口；冻结源/数据/初始化；显式授权、锁、磁盘预检、原run_stage和本次last恢复 |
| scripts/v3_ready/readiness.py | 真实数据零更新、合成恢复、全开发集零更新三项验收 |
| scripts/v3_ready/report.py | 核对回执与完整恢复状态后生成PASS、脱敏报告和唯一命令 |
| tests/test_bcep_v3_formal.py | 未授权拒绝、禁止step、test拒绝、minimum30/patience15、Top5与完整状态 |

CSEE/HCRA/GRG-CTF及其初始化未改；bcep、ebjd、bcep_v2、bcep_joint原实现未改。原Provider、四层Flow、候选修正头、原损失、六层桥、JMM划分、keyed noise、优化器/调度/早停/Top5算法均保留。新增记录仅观察梯度/参数，不把范数小改成失败门槛。
