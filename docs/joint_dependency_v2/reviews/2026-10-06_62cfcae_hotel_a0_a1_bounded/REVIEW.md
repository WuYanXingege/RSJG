---
experiment: HOTEL bounded Stage-A A0/A1
date: 2026-10-06
status: HOTEL_A0_A1_RUNNING
parent_sha256: 5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b
---

# HOTEL A0/A1 受控训练审查

本归档执行官方 HOTEL fold 上的 frozen-GDTS Stage-A 对照。A0 使用现有
`mean_energy`；A1 唯一科学差异是 S=4 `expected_conditional_mc`。三组训练种子
为 3101/3102/3103，每对读取相同逐 tensor 初始化；正式顺序固定为
A0-3101、A1-3101、A1-3102、A0-3102、A0-3103、A1-3103。

父模型唯一允许值为官方 HOTEL epoch 110，SHA256
`5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b`。
原 GDTS goal/history/diffusion 全冻结且保持 eval，Stage-A corrector 输出层严格
全零并走 literal bypass。

本轮新增的最小执行适配包括：共享 cache seed、五个显式推理 seed、配对初态
认证、global/loader/CUDA RNG 的完整 epoch-boundary resume、编号 checkpoint 原子
保存、12 GiB reserved-memory 门、6 小时单臂与 48 小时队列 deadline，以及在真实
conditional pair score 入口执行的三项机制干预。

当前已升级为 `READY_FOR_BOUNDED_HOTEL_A0_A1`。父权重、全量 cache、配对初态、
双臂真实 CUDA 更新、100-update 吞吐、完整 445-window validation 和 A1 独立进程
resume 均通过，机器摘要见 `READINESS_RECEIPT.json`。READY 只表示执行合同通过，
不表示 A0/A1 已产生性能收益。

正式运行源码固定为 `664ad0410f5cf3b44d3a6c1892c98e37aa936b9e`；候选 cache
由 `5b6e4e4ee11b5a19884cf2bfb11eae700820dbe5` 生成，并通过显式完整 SHA
override 绑定，未把工具修复伪装成 cache 重建。A0/A1 两步 smoke 的共享初态 hash
为 `c9d2d621…`，冻结家族 hash 为 `df7af51e…`，两臂均有 63 个参数 tensor
实际获得梯度，注册可训练参数为 331,845。

100-update 实测为 A0 0.0679 s/update、A1 0.0467 s/update；完整 validation 为
260.1 s。40 epoch 的训练+逐 epoch validation 投影约为 A0 5.52 h、A1 4.70 h，
仍分别受 6 h hard limit 和总队列 48 h deadline 约束，投影不构成完成保证。

正式队列于 2026-10-06 21:20:22（北京时间）启动。manager PID 1156024，首臂
A0_3101 worker/GPU PID 1156051；PID、PPID、PGID、SID、`/proc` start ticks 和
cmdline 已回读一致。首臂已观察到超过 700 个正式更新并持续推进，详见
`LAUNCH_RECEIPT.json`。当前状态是 RUNNING，不是 COMPLETED，也没有正式性能结论。

启动前的普通 GitHub push 因本机缺少 HTTPS 用户凭据失败；本地归档 commit
`5bf5b60` 完整保留，未强推，训练未受影响。凭据恢复后可普通推送当前分支。

## 证据边界

HOTEL 官方 `val/biwi_hotel.txt` 与 `test/biwi_hotel.txt` 字节相同；父模型已经在
该来源上选模。因此本轮是官方代码协议下的开发对照，不是独立 held-out 证据。
三个 Stage-A head 共用一个 GDTS parent，也不是三个独立 GDTS 训练。论文 HOTEL
0.13/0.18 与本地官方推理 0.13394/0.19140 不能表述为两项均完整复现。

训练 source 是 ETH、UNIV、ZARA1、ZARA2 的六个官方训练文件；HOTEL 文件仅用于
已披露的验证/最终完整评测。Stage-A 使用同步 whole-scene windows，不把 per-agent
minima 称为 JADE/JFDE。
