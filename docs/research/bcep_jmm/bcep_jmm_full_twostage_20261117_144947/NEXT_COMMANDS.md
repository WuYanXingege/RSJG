# 资源释放后的唯一正式入口

当前GPU仍运行旧有界试验；本轮未排队，不自动抢占。待旧任务按合同退出后，在下列隔离工作区执行：

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_BCEP_JMM_20261117_144947
bash scripts/start_bcep_jmm.sh /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/RUN_PLAN.json
```

该命令先检查GPU空闲，执行临时状态GPU验收和完整验证工作量计时；通过后冻结运行时配置，再以nohup+setsid启动唯一有限五折流程。资源忙、预检失败或配置身份不一致均退出，绝不静默启动。现阶段CPU已通过、GPU验收未执行，不能宣称TRAIN_READY。

若正式流程已有EXECUTION_IDENTITY，该入口跳过重新验收/选配置，使用同一完整last与已完成阶段收据恢复；源码commit、配置、split身份必须一致。故障先核查，不改写身份继续。没有无限cron、等待进程或额外训练seed。

顺序：ZARA2→ZARA1→HOTEL→UNIV→ETH；每折Provider→train/val bank→随机六层EBJD→冻结test五seed bank→G/H/H_shuffle评测。日志：/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/logs/pipeline.log。状态：/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_jmm_full_twostage_20261117_144947/PIPELINE_STATE.json。
