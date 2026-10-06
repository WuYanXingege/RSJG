---
experiment: HOTEL bounded Stage-A A0/A1
date: 2026-10-06
status: PREPARING
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

当前状态只可在 `PREFLIGHT_RECEIPT.json` 全部通过后升级为
`READY_FOR_BOUNDED_HOTEL_A0_A1`，随后由固定源码 worktree 后台启动。不得把
PREPARING/READY 写成性能收益。

## 证据边界

HOTEL 官方 `val/biwi_hotel.txt` 与 `test/biwi_hotel.txt` 字节相同；父模型已经在
该来源上选模。因此本轮是官方代码协议下的开发对照，不是独立 held-out 证据。
三个 Stage-A head 共用一个 GDTS parent，也不是三个独立 GDTS 训练。论文 HOTEL
0.13/0.18 与本地官方推理 0.13394/0.19140 不能表述为两项均完整复现。

训练 source 是 ETH、UNIV、ZARA1、ZARA2 的六个官方训练文件；HOTEL 文件仅用于
已披露的验证/最终完整评测。Stage-A 使用同步 whole-scene windows，不把 per-agent
minima 称为 JADE/JFDE。
