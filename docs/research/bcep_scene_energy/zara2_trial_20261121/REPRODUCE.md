# 固定源码与唯一训练命令

正式训练代码固定为`ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86`，报告工作区从未作为训练入口。非零恢复验收固定为`9fad8f6dfbcefaebaa6a4944db3f5d35062ff91b`。

原正式源码中的唯一受控命令：
[CONTROLLED_COMMAND.txt](https://github.com/WuYanXingege/RSJG/blob/ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86/docs/research/bcep_scene_energy/stage2_ready_20261121/CONTROLLED_COMMAND.txt)

冻结配置文件SHA256：`3ff36b653703307be4f38457d9cb9395c264ea7647c0df339913376f38b9d842`。本次已启动，不能再次创建随机初态实验；技术恢复只能由同一命令读取本次`joint/last.pt`。

报告生成仅使用原生开发集CSV、完整checkpoint状态和日志，不调用optimizer、模型推理或测试集。训练结束后可用`python scripts/ssec_trial/report.py final`重建报告；运行中可用`python scripts/ssec_trial/report.py interim`读取快照。
