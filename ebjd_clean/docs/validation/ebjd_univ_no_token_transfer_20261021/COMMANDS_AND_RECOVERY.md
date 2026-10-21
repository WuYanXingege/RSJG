# 固定状态与可选恢复边界

本轮已经按 6 小时硬预算停止，不会自动恢复或继续消耗 GPU。

完整冻结 checkpoint：

- Control epoch81：`e9fd76be585fef0d9cf6ab82422e5a5775392030c59283054a5247a6f47eafdd`
- Candidate epoch81：`8d695b9e494674610cc51a6719ee6227363ddc158846cb3309d1fabe32624a0e`

若未来另行明确批准新预算，只允许从这两个 checkpoint 补做原预注册 2035–2039 评测。不得继续训练、改 seed、降低 P20/20 steps、调 rho 或更换主输出。
