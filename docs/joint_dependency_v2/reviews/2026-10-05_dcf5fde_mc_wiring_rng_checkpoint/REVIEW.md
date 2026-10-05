# MC 训练接线、独立 RNG 与 checkpoint：CPU 验收

2026-10-05，Asia/Shanghai。输入源码 dcf5fde1aa77f62a25a9209fd91e4f854a21510a；分支 research/joint-dependency-v2-clean。开始时 tracked 工作区干净；未发现适用 AGENTS.md，原有未跟踪 data/outputs 均保留。

状态：**CPU_MC_WIRING_RNG_CHECKPOINT_CERTIFIED_GPU_PENDING**。最终有效覆盖 **166 项通过：70 项新接线/状态 + 4 项既有参数/恢复 + 92 项原纯损失**。这是 constructor-free CPU 合成合同认证，不是完整模型、CUDA/AMP 或真实训练恢复认证。

## 改动与默认兼容

- 新增显式 `jdv2_goal_objective=expected_conditional_mc`，默认/缺字段为 mean_energy，固定 S=4。只允许 active strict_no_z Stage-A、非 SDD/V4、CPU、无 AMP；parser/trainer/model 构造前都有门禁，不先探测 CUDA 再 fallback。
- MC 独立方法复用原 encode(for_loss=True)、dynamic relation、energy mixing 与 KL 公式。一个请求一次抽样，post/prior 共用同一个 z 对象和 chunk bounds；每块一次 relation/factors/energy 构造、两次不同 mixing。端点 q 固定，teacher 经 Cpost/KL 可导。
- trainer 拥有非 Module 的 ObjectiveRNG；新增参数/buffer 为 0，不进入 architecture_config 或网络 state_dict。推理主体不读该 RNG；MC eval diagnostic loss 拒绝，metrics validation 不再额外调用 MC teacher loss。
- 旧 loss 正文除两行 early dispatch 外逐字不变；实际旧方法与从固定提交编译的旧方法在相同 raw fixture 上值、梯度、keys、coefficients 完全一致。其余 55 个 GDTS 方法（包括 inference/encode/sample）逐字不变，见 [SOURCE_SCOPE](SOURCE_SCOPE.json)。不将静态证据冒充 GPU 输出 parity。
- Stage-A accumulation **仍为 1**；设计中的 8 未实施。独立记录 draw、sampled-agent-draws 和 successful updates；beta 随后者变化。CPU skip 不增 update，错误不重试。
- last 在 scheduler 后，best/numbered 在之前。MC 仅 last 可续跑、仅其启用同目录原子保存；mean 保存策略不改。train_test 的测试加载显式走 weights-only。

接口见 [输入/模式合同](INPUT_AND_MODE_CONTRACT.md)，完整兼容矩阵和失败规则见 [RNG/checkpoint 合同](RNG_CHECKPOINT_CONTRACT.md)。实现位于 [model](../../../../src/models/model.py)、[trainer](../../../../src/trainer.py)、[helper](../../../../src/jdv2_objective_state.py)、[parser](../../../../src/parser.py)。

## 数值与恢复证据

FP32 wiring 最大记录绝对误差 8.009374141693115e−8，低于预登记 atol=1e−6/rtol=1e−5。raw unary/social、deploy relation、teacher、energy 两因子梯度均与独立候选/边循环参考一致。CE 按 occupied scene 等权，KL 按全 batch 边平均。E0、N1、mixed、all-active、不连续 scene 标签、非均匀 scene size、mask 均覆盖；无效标签/支持/scene/dtype/AMP 在抽样前拒绝。

原 92 项纯损失 CPU FP32/FP64 测试原样通过；本轮 production-like wiring 保留既有 FP32 标签/cost 边界。**不声称整个网络有 FP64 合同**；FP64 认证限定原纯损失及独立 RNG 的合法 FP64 q。

3 epoch × 3 updates：连续 9 步 vs 3 步保存 last + 独立新进程恢复 6 步，逐步比较 z、post/prior/KL/total、coefficients、gradients、parameters、optimizer、scheduler、progress、RNG bytes SHA256 与 counters，全部 exact。最终 trace 摘要 88d99c99a1ead16e6d2405672cfa239289564af4606e8beb9a1afaea08bab433。子进程调用实际 _load_or_restart → _load_checkpoint → _load_state_file 和实际 MC optimizer/progress 方法，不是同名 mock。

fixture 用 raw Parameters 与代数 callable 替换真实 encode/relation/energy；未运行真实模型构造、真实 forward、loader 或训练循环。还单独覆盖生产“先加载权重、后创建 optimizer”的延迟恢复顺序。

原子故障测试：序列化、flush、文件 fsync、replace 失败保留旧目标字节并清理临时文件；rename 后目录 fsync 失败抛出带 committed=True 的错误，明确新文件已提交、持久性未知。

## 测试历史与账本诚实性

[RESULTS](RESULTS.json) 汇总逐项记录；[命令](COMMANDS.md) 给出顺序。

首轮 59 项中 33 通过、26 失败：25 次验收脚本误取 Path.name，错误拒绝 /tmp toy checkpoint；另 1 项 parser fixture 漏填 test_set=eth。没有真实 artifact 读取。修正后仅重跑这 26 项，全部通过。随后补充边界/加强校验，重跑受影响新套件 68 项通过；最后新增 2 项 NaN/optimizer-error 测试通过。纯 loss 92 项没有重复跑。合计 247 test-call，221 pass、26 已解释失败；不是 247 个独立案例。

首轮 stdout 太长，被工具截断；**原截断文本保留**，没有伪造完整 JSON。首轮丢失的计数块在 [CPU_RUN_1_SUMMARY](CPU_RUN_1_SUMMARY.json) 按固定测试控制流、未改 RNG 案例和后续完整记录显式重建，并标为 derived，而非原始 telemetry。当前最终 RUN3/4 的逐项结果、计数、误差和恢复摘要均完整。

全轮包含重跑与两个 resume 子进程：合成 successful optimizer updates **279**；目标 RNG draws **307**、sampled-agent-draws **5732**；实际 no-z 方法 fixture dispatch **286**（含预期拒绝）；原始张量图 backward/gradient API 请求 **282**。NN Module forward 为 0，不意味着 raw autograd/optimizer 为 0。两次子进程是验收重跑，不当独立统计重复。测试 runner 主体合计 22.475 秒；进程启动/源码检查/文档和 Git 操作另计，远低于 CPU 1 小时验收预算。

真实 full-A/encoder/GRU/message/relation-energy/sampler/diffusion/corrector、GPU、数据集训练/评估、真实 checkpoint/bank 读写均 0。SDD 未恢复/干预，Skills/Apps 未改。

## 限定与下一门槛

双 post/prior chunk 列表保留全部上游 autograd 图直到 backward；不承诺峰值只与 chunk 大小有关。

恢复只覆盖**相同固定输入序列、相同计划总步数、已完成 epoch scheduler 的 optimizer boundary**。DataLoader/worker、augmentation、全局 RNG、输入顺序及 CUDA 非确定性均 OPEN，不称真实训练逐位续跑。

JADE/JFDE、learned pair energy 因果、新颖性、历史 29 IDs 链和 GPU observer 中立性均维持旧状态。历史 C 仍是 V2-A checkpoint 但 correction arithmetic 关闭，不是 D。

停在本轮 CPU 单元。下一步须另行授权同-device CUDA/AMP 纯算子验收与最小真实生产 smoke；再之后才讨论有 clean split/base provenance、预注册与预算的性能实验。未启动 204h、clean-base、geometry6、V4 或任何训练。
