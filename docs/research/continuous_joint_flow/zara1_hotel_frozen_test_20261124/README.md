# ZARA1、HOTEL冻结测试：启动记录

用户于2026-11-24授权立即完成已训练新模型的ZARA1/HOTEL测试。本次提前测试这两折，不等待UNIV/ETH选模结束；原四折控制器的源码、配置、训练和选模规则保持原状态。

固定ZARA1 A epoch15、B epoch6，HOTEL A epoch12、B epoch19。官方heldout_confirmation分别705窗/2356人次和445窗/1197人次，N≥9分别28窗/288人次和0窗/0人次。HOTEL密集组记N/A。

使用原生joint_flow_fourfold.test入口，种子2035–2039，各自K20，M256、12帧、20步DDIM、eta=0。零训练，关闭1 cm校准、整组交换和A轨迹重排。每折五种子全部A/B预测先封存并核验，再读取测试未来，用原生指标评分；报告五种子均值、样本标准差及配对差。

启动：`CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m joint_flow_pair_test.run --push-on-complete`。进程按同一配置恢复，完成后本页自动替换为中文结果报告。原始预测和权重不上传GitHub。
