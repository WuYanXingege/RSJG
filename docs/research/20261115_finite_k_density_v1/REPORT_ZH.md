# ETH有限K终点风险训练与速度验收

本轮完成实现与短测，新方法尚无完整epoch或五种子精度结果，不能宣称minFDE已改善。旧实验已停止；新模型独立随机初始化，不把旧进度记为新方法训练。

## 实际修改

- 保留8层条件桥、原20次IID终点采样、学习配对、原损失项。新增权重1.0的有限20候选覆盖风险，直接训练终点概率分配；公式与近似边界见DESIGN_ZH.md。
- 新配置的小型coupling打分使用FP32，主网络仍使用原BF16/FP32策略。它降低了本次连续重复的配对参数漂移，但没有消除离散分配对微小扰动的敏感性；不能据此认定旧性能差由精度造成。
- 连续场景合并未通过共同父状态门槛，不采用。日志统一回传两次仅约1.2–1.5%，低于预定1.03×启用门槛，不用于正式新训练。

## 正确性与证据边界

离散order-statistic期望与穷举一致；梯度流向更近候选的概率质量，覆盖局部/外部支持、padding及共享特征梯度。正式64×64网格检查通过。4×4粗网格曾出现约0.30m积分偏差，不泛化宣称任意网格精确；旧模型8个真实验证窗口的2×2积分与1000次IID20参考均值最大差约0.019m，只是小范围数值核验。

连续8步baseline对自身也有离散配对分叉，因此另做8个连续参考状态的独立完整单步分叉，恢复相同模型、optimizer及全部RNG，使用相同真实batch和增强。门槛未放宽：loss相对差≤0.1%，梯度范数差≤1%。本轮性能对照不是新旧目标的精度消融。

## 性能

RTX5070 Ti 16GiB、PyTorch2.8.0+cu128；每臂预热3次，计时32次完整optimizer update，配对重复两次并交换执行顺序。包括数据等待、CPU增强、搬运、前后向与参数更新；不包括验证和checkpoint保存，不能称为完整epoch耗时。训练逻辑batch仍是4场景×4累积，保持真实人数分布。

| 窗口 | 16槽位 s/update | 32槽位 s/update | 倍数 |
|---|---:|---:|---:|
| 1 | 4.040085 | 3.905260 | 1.035× |
| 2 | 3.992839 | 3.875194 | 1.030× |

最终选择：`denoiser_graph32`。两臂都含新风险项及FP32配对打分。其余精度、U-Net checkpoint、验证、损失与实际batch不变。大于32槽位仍checkpoint，显式checkpoint请求仍优先。峰值allocated约2.10GiB；资源见TIMING.csv。RSS为主进程累计高水位，不包括所有DataLoader worker的内存总和。

共同父状态验收：`{"max_relative": {"loss": 0.0, "goal_nll": 0.0, "finite_k_risk": 0.0, "pair_loss": 0.0, "diffusion_loss": 0.0, "goal_grad_norm": 0.0, "bridge_grad_norm": 0.0, "map_grad_norm": 2.9998954536434404e-07, "coupling_grad_norm": 1.0331389654559655e-07}, "parameter_l2_relative": {"goal.": 2.3210649797099708e-10, "coupling.": 3.544396803064442e-10, "bridge.": 2.998348066155316e-09}, "input_identities_equal": true}`。

## 训练衔接

旧训练完整epoch43/update11051保留，未保存epoch44的19次更新有停止收据；旧supervisor及自动续跑已停止。新run路径见SUMMARY.json，所有正式权重与日志留在资产目录。

新实验保持seed3101、全部预测参数随机初始化，rollout/projection OFF。100轮学习率时间轴，先完成5轮；沿用户此前明确的四项H验证指标均低于前两轮均值条件，决定是否续至总计100轮。未通过则保留完整第5轮状态并停止。该条件仅用于计算投入，不能代表论文性能通过。

正式启动在本报告及代码提交后，由RUN_MANIFEST绑定该提交与配置哈希。后台采用nohup+setsid、run锁及全局GPU锁，支持完整epoch恢复，不依赖聊天会话。实际PID/进度在PIPELINE_STATE.json；console.log每epoch一行，train.jsonl保留更新记录与新风险项。首个完整epoch耗时由epoch_metrics.csv记录。

## 复现

使用本run的configs/RESOLVED.json及CONTRACT.json；benchmark入口：

```bash
python scripts/check_finite_k_training.py --common-parent-gate --candidate-denoiser-slots 32 --root <独立短测目录> --scale-identity <原run/training/RUN_IDENTITY.json>
```

短测目录需先复制本run的两个小型configs文件，参考运行时阈值设回16；不要向正式训练目录输出临时模型。核验结束启动用资产目录START_OR_RESUME.sh，不重新生成数据、bank或初始化权重。

本轮没有对验证指标调参，没有新增训练seed，也未上传任何权重或bank。总loss加入新项后不能与旧总loss直接比较。下一步只观察新训练是否能将覆盖风险下降转化为验证minFDE收益。

额外GPU任务墙钟合计1631.4秒，临时optimizer更新400次；均未进入正式状态。
