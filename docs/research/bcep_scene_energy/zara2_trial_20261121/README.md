# SSEC Energy-only ZARA2 Stage2：运行中分析

**尚非最终结果。** 北京时间2026-11-22T12:23:21+08:00快照：已观察到6384次真实更新，完成37轮，当前按H-JADE选择的rank1为epoch 31。

[完整中文阶段分析](status_20261122_122319/README.md) · [三项门槛比较](status_20261122_122319/THRESHOLD_COMPARISON.json) · [匿名逐窗对照](status_20261122_122319/CURRENT_RANK1_VS_OLD5_WINDOWS.csv) · [逐轮CSV](EPOCH_GROUPS.csv)

| 指标 | 当前rank1 m | 旧epoch5门槛 m |
|---|---:|---:|
| 全体H-JADE | 0.409868506 | ≤0.409767019 |
| 全体H-JFDE | 0.813756070 | <0.784179166 |
| N≥9 H-JFDE | 1.340309367 | <1.296843430 |

训练仍在后台按原规则继续；本次报告更新不改变训练、选模或停止规则。完成后监督器将发布最终研究报告并核验远端SHA。正式训练源码固定`ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86`，报告分支不作为训练入口。
