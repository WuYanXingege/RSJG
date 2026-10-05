# CUDA/AMP 与真实 Stage-A loss smoke

日期：2026-10-05（Asia/Shanghai）。输入源码：`c9dbfecf64eb95504664c2901d3c2bb26f41aea5`；分支 `research/joint-dependency-v2-clean`。状态：**CUDA_FP32_BF16_LOSS_SMOKE_CERTIFIED_NO_TRAINING**。

这是固定小例和四次真实训练损失图的运行/梯度验收，不是训练、效能、完整推理或 CUDA exact resume 认证。默认仍为 mean_energy；任何后续数据训练须另行授权。

## 结果

| 门槛 | 实际结果 |
|---|---|
| CPU 回归 | 172/172：原 166 项 + 新后端 6 项，整套只跑一次 |
| CUDA 合成 | 16/16 top-level forward + backward；其中两例包含 post/prior 两个 CE primitive |
| 非法输入 | 8/8 按预登记原因拒绝，无 backward；owner 未消费 RNG |
| GPU toy optimizer | 2 次实际 Adam 更新，另 2 次 skip；未使用真实模型 optimizer |
| 真实 Stage-A | R1–R4 共 4 次 get_loss、4 次 encode、4 次 total backward，无失败/重试 |
| 资源 | RTX 5070 Ti；GPU 阶段 3.334183 秒；峰值 allocated 135195136 B，reserved 150994944 B（144 MiB） |
| 禁止项 | 真实参数更新、训练/评估循环、sampler/diffusion/corrector forward、完整轨迹、bank 导出均 0 |

GPU 初始化前无其他 compute owner，可用 15170 MiB。PyTorch 2.8.0+cu128 / CUDA 12.8 / cuDNN 91002；native BF16 API（including_emulation=False）返回 true。TF32、cuDNN、确定性、线程策略前后一致。没有安装、compile、profiler、CUDA Graph、warmup、kill 或 SDD 状态操作。

## 真实身份与数值

严格读取 canonical Stage-A epoch 13 checkpoint 一次，SHA256 `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb`；strict load 无 missing/unexpected keys，stage、strict_no_z、architecture/ablation metadata 全匹配。仅 weights-only load，未恢复旧目标 optimizer/训练状态。

从训练缓存按 source/frame/window 排序固定选取 biwi_hotel：frame 0 / cache 240 / N=3 E=3；frame 620 / cache 175 / N=1 E=0。没有截断 agent/edge，没有看 loss 后换窗，无开发集 fallback、新 final-test 标签解封或缓存重建。原观测/预测 8/12、K=21、M=4、rank=8、Social128、P=20 保留；固定 smoke progress=0.25。具体数据 hash、candidate order、graph 和 prepared shape/dtype/stride 见 [PREFLIGHT](PREFLIGHT.json)。

| 请求 | 目标/精度 | total | post / prior / KL |
|---|---|---:|---|
| R1 | mean_energy FP32 | 3.0453193188 | 3.0438792706 / 3.0439307690 / 0.0141432993 |
| R2 | MC S4 FP32 | 3.0459055901 | 3.0441303253 / 3.0448522568 / 0.0141432993 |
| R3 | MC S4 BF16 autocast，loss FP32 | 3.0459170341 | 3.0441403389 / 3.0448529720 / 0.0142050460 |
| R4 | singleton MC BF16，loss FP32 | 3.0287842751 | 3.0287842751 / 3.0287842751 / 0 |

六个真实 MC CE scalar 对各自 CUDA 捕获的同一 u/q/C/z 的独立 CPU FP64 loop，最大绝对误差 3.68308e-7，均通过预登记 atol=1e-5/rtol=1e-4。E0 退化 unary CE，KL=0，仍 draw 一次。R2/R3/R4 分别消费 12/12/4 个 agent-draw，success=0、skip=1、pending=false。

有边 case 的 social、base relation、unary、dynamic、energy、teacher 均存在 finite 非零梯度；E0 只有 social/unary 是必需连接。没有强求所有参数非零；例如 dynamic relation_embedding 不参与 Stage-A loss。冻结基座/corrector 无梯度。四次前后所有 state_dict 参数/buffer 哈希均为 `025044aa75dbfa5366a34d65b0f7cc54c45c017e18aae9cbc216c9b8cf4f113c`。完整逐参数梯度摘要、key/shape、原生 module pre-hook 次数见 [原始结果](gpu_run/RESULTS.json)。

BF16 相对 FP32：q 和 z exact；u 最大差 0.0255706310，post C 最大差 0.0141534805、prior C 最大差 0.0145859718。它们是不同上游近似产生的不同有效输入；两组均各自通过参考，不能比较 R1/R2 或 R2/R3 来声称收益/退化。

## 算子证据与记录边界

合成覆盖非对称 cost、重复 receiver、多 chunk、singleton/E0、不等 scene、mask、agent/candidate 重排、非连续 stride、M4 teacher/deploy/energy raw 梯度、固定重复、实际 step/skip。固定 inputs/z/CPU generator state 和 hashes 在 GPU 前归档。对 CPU FP32 的 loss/gradient 最大绝对误差 1.19209e-7；double raw-algebra reference 最大 5.31481e-8。

14 个直接 CE case 的有效 u/q/C 是跨设备固定同输入；2 个 M4 raw-chain case 的 energy mixture 在各设备计算，CPU/GPU 的 C 不宣称逐位相同。因此原始 JSON 中 `cpu_fp64_identical_effective_inputs` 对这两例指 CPU 构造的 C，而不是捕获 GPU C；它是补充链式参考，不能误称 GPU 中间 C 已逐位固定。真实 R2/R3 则确实保存并参考了 GPU 实际 C。重复小例在本次运行值/梯度 exact，不推广为确定性保证。

合成 draw 检查 global CPU/CUDA RNG 和独立 evaluation-test generator 不变，z/owner CPU state exact。真实 draw 另用同一实际 q 复核 state/IDs；goal/Gumbel sampler 未执行，不宣称完整 generator 系统续跑。

CPU stdout 超过工具返回预算，中间 per-phase 事件行截断；保留 [原截断输出](CPU_RAW_TRUNCATED.txt) 和完整尾部 counters/pytest 总结 [CPU_RUN](CPU_RUN.json)，没有为重新捕获而重跑。CPU plan 开发有一次转义语法失败及一次 empty-cost max 失败，均在 CUDA 之前；实际 GPU 无失败重跑。运行出现 requires_grad scalar 转换与 Transformer nested-tensor 提示，不是 NaN 或 gate 失败。未偷偷调整策略。

## 修改边界与交付

只新增显式后端门禁、同设备 FP32 loss、CPU RNG label bridge、CUDA 区分 metadata，以及 trainer 更新/CPU generator checkpoint load 必需支持。CPU format1/fingerprint 保持，CUDA format2 明确不认证完整 resume。原 mean 目标、encode、inference、graph、architecture、cache、生产 YAML 未修改；新网络参数/buffer/extra_state 为 0。源码逐定义核验见 [SOURCE_SCOPE](SOURCE_SCOPE.json)，详细矩阵见 [DEVICE_AMP_CONTRACT](DEVICE_AMP_CONTRACT.md)。

入口 [audit_gpu.py](audit_gpu.py) 的 existing-run guard 防止误重跑。当前 GPU 额度已用满（16 合成、8 非法、4 真实），不能删除 guard 后继续。本报告归档并普通推送；固定 commit 远端 bytes/SHA256 回读另生成本地 receipt，避免报告内自引用 commit/hash。

## 仍 OPEN / 下一决策

JADE/JFDE 泛化收益、不可分解 pair energy 因果贡献/新颖性、完整 data/worker/global RNG/CUDA 训练续跑、历史 29 IDs 因果链、GPU observer neutrality、完整 sampler/diffusion 部署以及 clean split/base provenance 均 OPEN。原生 hook 是观测，不是 neutrality 证明。

下一步回到 clean base/split 只读清点与有界对照预算。旧 ETH139 不可作为新目标选择和无偏最终测试；本次通过不授权 204h 计划。SDD 仍未恢复，Skills/Apps 未改，全部历史结果保留。

机器可读汇总：[RESULTS.json](RESULTS.json)；命令与账本：[COMMANDS.md](COMMANDS.md)；原生 small payload：[gpu_run](gpu_run)；校验清单：EVIDENCE_MANIFEST.json。
