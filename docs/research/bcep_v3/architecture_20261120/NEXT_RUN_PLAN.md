# 唯一后续计划——本轮禁止执行

本轮真实训练更新为 0。当前入口只接受内置合成数据类型，即便把 allow_real_training 改为 true 也会拒绝。本文件不是训练授权，不存在自动队列。

审阅通过后，仅考虑一次 ZARA2 Stage2：冻结原自有 Provider、候选修正头和 seed 2035 train/selection_dev refined bank；固定官方 2681/590 成员，记录 127 个 N≥9 开发窗。继续使用语义地图。不读取 test 调参或选模。

从 v2 相同随机初态（下游 seed 3201）构建共享编码/配对/六层桥；新增参数使用独立局部 seed 3201 和零残差，记录旧/新指纹。本次迁移 epoch26 只用于兼容性验证，未来训练不从 epoch26 或其他历史权重微调。保留 formal final_coordinate 及其余有效损失；CSEE+HCRA+GRG-CTF 是结构变量，不再搜索已失败的配对损失。

沿用 FP32、严格确定性、原 AdamW/学习率/数据顺序/增广、5轮 warmup、100轮调度、minimum30/patience15/max100；每轮20步DDIM、K20，仅按完整 dev H-JADE 选择 rank1（同分早轮）。保留所有完成epoch、best、last和五个不同最佳epoch及SHA，不用JFDE另选权重。若后续测试被独立授权，沿用原五seed策略；本计划不启动测试。

记录每轮 G/I/H-JFDE（all与N≥9）、H-JADE、每人边缘minFDE、I−H反馈收益、真实更新数、有效人数峰值、显存和时间。逐窗配对比较旧epoch5（0.4097670185 / 0.7841791655 / 1.2968434302 m）和旧正式epoch26 H-JADE 0.4084707879 m。预先冻结验收规则，不能从Top5中改挑JFDE最低权重。

工程验收不代表精度提升。若一次受控开发集训练不改善或稳健性不足，归档负结果并停止。只有结构有效且另行授权后，才安排有预算的candidate evidence / HCRA / 联合 / bridge gate核心消融；本轮不开展多臂试验、不扩折。
