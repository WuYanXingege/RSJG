# UNIV NO_TOKEN 短程迁移：预算边界收据

状态：**PARTIAL_GPU_BUDGET_EXHAUSTED**。

两臂均从冻结的 UNIV epoch73 完整状态续训至固定 epoch81，各完成 8 轮、2,040 次更新；最终 update 为 20,655。两臂输入、增强、基础随机数和学习率逐 update 完全配对，非有限值为 0，rollout 与 projection 均未执行。Control 与 Candidate epoch81 checkpoint 已完整保存并绑定 SHA256。

| epoch81 单次训练验证输出（非五种子正式结果） | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| Control Raw | 0.269695 | 0.507299 | 0.528420 | 1.126444 |
| Candidate Raw | 0.284685 | 0.505523 | 0.618606 | 1.239390 |

上述数值仅是训练流程的单次 validation，不能替代预注册的 2035–2039 Raw/Fixed-anchor 正式表，也不能据此拼接或判定合同。

首次 Control 正式评测在无 native-goal 的 Control 上触发空 endpoint 分位数汇总异常，耗时 371.6 秒且未写正式结果。该只读汇总边界已修复并通过 89 项测试。随后 Candidate Raw 与 Fixed-anchor 同前向五种子评测在剩余预算硬超时 3,400 秒处停止；结果采用原子写入，因此没有产生可认证的部分 seed JSON。

实际 GPU 墙钟口径为 **21,275.53 秒 / 21,600 秒**，剩余 324.47 秒，不足以完成任一完整五种子评测。按照预注册停止条件，不延长预算、不减少 seed/P/DDIM、不使用部分 seed，也不把 Raw 后验替代主候选。

- 旧 UNIV 优势是否由本迁移方案保留：**不可判定**；原 epoch73 成功模型及其结果未被覆盖。
- 升级合同：**NOT_EVALUABLE**。
- 相对 matched GDTS 的原合同：**NOT_EVALUABLE**。

唯一建议：**继续保留原 UNIV epoch73，不升级本迁移分支。若以后明确批准一笔新预算，只补评测两个已冻结的 epoch81 checkpoint；不得重训、调参或延长到 epoch82。**
