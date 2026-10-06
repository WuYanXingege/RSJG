# GDTS 官方代码 HOTEL 折复现与启动验收

结论：固定上游 GDTS commit `297d508558c10831983ea4b19c2b3e657459a449` 的 HOTEL 折已完成独立数据预处理、真实 CUDA 更新和推理链路验收，状态为 **OFFICIAL_GDTS_HOTEL_READY**。本实验用于复现作者公开代码路径，不复用 grouped-UNIV 的 goal/joint checkpoint、cache 或选模结果。

## 协议身份

| 项目 | 固定值 |
|---|---|
| 上游源码 | `Winderting/GDTS@297d508558c10831983ea4b19c2b3e657459a449` |
| 测试折 | HOTEL |
| train 原始文件 | ETH 1、UNIV 2、ZARA1 1、ZARA2 2，共6个 |
| valid/test | `biwi_hotel.txt`；两者 SHA256 均为 `9caa771b...bbb7fcf` |
| 序列 | 观察8帧、预测12帧 |
| 采样 | TTST、20条预测 |
| 训练 | seed2025、FP32、batch64、Adam lr `1e-3`、clip1 |
| 调度 | ExponentialLR gamma `0.995`（来自固定官方 trainer） |
| 预算 | joint 250 epochs，每10轮验证与保存 |
| 目标 | `diffusion_loss + 20 × goal_BCE_loss` |

预处理完全使用官方代码和作者数据包，生成 591 个 train batches、19 个 valid batches，占用约19 GiB。原始文件 SHA 与源码 SHA 见 [RESULTS.json](RESULTS.json)。

## 两项运行时兼容补丁

训练算法、成员、概率和损失均不改变，只对现代依赖作两项兼容：

1. 将已移除的 Albumentations 内部命名空间改为等价公开类入口；
2. 对 `64×20=1280` 通道轨迹热图一次采样增强参数，再按256通道分块 replay 同一几何变换，绕开当前 OpenCV 的高通道 flip 限制。

固定补丁 diff SHA256 为 `b8a09ae42f0c1ce3136a4c0a6e40d33f437defff3b87e014fa1c2a8c6b51f5e1`，补丁后 `src/data_loader.py` SHA256 为 `c6c66e761455c126dda173d107033dd30e60d4777acafeb285bd671632376952`。环境为 PyTorch 2.8.0+cu128、NumPy 1.26.4、OpenCV 4.11.0、Albumentations 1.4.24。

## 实际验收

- 首次内嵌 smoke 在 Python 解析阶段失败，更新0。
- 第二次进入 loader 后因旧 Albumentations 私有入口失败，更新0。
- 第三次进入增强后因1280通道 OpenCV flip失败，更新0。
- 最终 smoke PASS：真实1次 optimizer step，goal/registrar/diffusion 三组非零梯度且参数均改变；峰值 reserved 2,405,433,344 bytes。
- 单个 valid batch 的完整 TTST→tree sampling→ADE/FDE 链路 PASS，像素与世界坐标指标均有限。该未训练 smoke 数值不是性能结论。

权威 receipt 见 [OFFICIAL_CUDA_SMOKE.json](OFFICIAL_CUDA_SMOKE.json)。

## 必须保留的论文/代码边界

- 这是“作者公开代码复现”。固定官方 `train.sh` 没有执行 goal pretrain，GDTS 构造函数中的 goal checkpoint 加载代码也被注释，因此正式 joint 从全部模块 fresh 初始化。
- 论文文字描述 goal 先训练150轮、再 joint 250轮。后续“论文意图版”和我们的方法必须作为另一条明确实验臂，不能用本实验冒充两阶段初始化。
- 官方 HOTEL 的 val 与 test 原始文件字节相同，官方 trainer 最终 `test()` 也调用 valid loader。因此公开代码的 best 选择与最终报告不具备独立 held-out test。
- 这条边界不妨碍逐字复现公开代码，但所有论文结论必须同时报告本地重训基线，并另设 leakage-controlled、公平共享的实验协议。
- WandB 仅为日志，不影响模型；正式命令关闭 WandB，其他训练参数与官方 `train.sh` 对齐。

后台启动命令与输出位置见 [COMMANDS_AND_BUDGET.md](COMMANDS_AND_BUDGET.md)。本轮只启动官方代码 HOTEL 基线，不自动启动其他四折、论文意图版或 JDV2 A0/A1。
