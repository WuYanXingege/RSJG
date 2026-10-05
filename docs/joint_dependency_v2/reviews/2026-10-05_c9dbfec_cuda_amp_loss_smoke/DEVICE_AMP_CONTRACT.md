# 设备、AMP、RNG 与 checkpoint 合同

本轮实现相对 c9dbfec；全局默认 mean_energy，不自动切换生产配置。

| 模式 | 计算输入 | autocast | checkpoint 身份 |
|---|---|---|---|
| mean_energy | 原行为 | 原行为 | 原行为不改 |
| MC 缺 backend / cpu_v1 | CPU FP32/FP64 | CPU AMP 拒绝 | format1、旧 fingerprint/schema |
| MC cuda_fp32_bf16_v1 | 同一 CUDA 的 FP32 u/q/C | off 或外层 BF16 | format2、backend/effective dtype/AMP policy |
| MC CUDA FP64/FP16、BF16 C、GradScaler | 拒绝 | FP16 拒绝 | 不开放 |

parser/helper/model/request/loss/trainer step 对齐：必须 active JDV2 strict_no_z Stage-A、S4、非 SDD/非 V4。CUDA 后端须显式 CUDA device；parser 若发现不可用不能静默 CPU fallback。模型与请求拒绝错误 autocast；trainer 仅 scaler disabled、CUDA FP32 参数同设备，finite check → clip → actual step → owner.finish_update。

u/q/C 必须 dense strided，同设备、相同 FP32 dtype；允许非连续 stride。scene/edge/z 为同设备 int64、mask 为 bool；q 固定且规范化，非法原请求在 draw 前拒绝。cost 在 loss chunk 消费处校验；若模型在 draw 后才发现下游错误，state 已消费、pending 保留，不回滚。纯 loss accumulation、log_softmax、CE、scene/draw 归约在 autocast disabled 内，输出 scalar FP32，梯度仍在原 CUDA 图。原 social/base relation FP32 islands 和 energy mixing、其他 AMP MLP 未修改。

Backward 放在 forward autocast 区域之外；本轮不使用 GradScaler。[PyTorch 2.8 AMP 官方示例](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html) 将 autocast 与 scaler 作为可分开的部件，并说明反向应在 autocast 区域外运行。

## RNG 与恢复矩阵

独立 CPU torch.Generator namespace/seed 派生/MT19937/multinomial replacement-Nx4-transpose 不改。先在原设备验证 u/q/scene/edge/mask；只 q.detach() → CPU，抽样 IDs → 原 CUDA。trainable u/C/factors 不为计算 loss 搬 CPU。post/prior 共享一次 draw 的同一个 z；E0 仍 draw 一次。

snapshot 的 uint8 generator state 是 CPU clone。cpu_v1 format1 不加新字段；CUDA format2 加 loss_compute_backend、loss_effective_dtype=float32、amp_policy=off/bf16，并将有限 resume_scope 后缀标为 CUDA exact resume NOT certified。读取 MC checkpoint 的 map_location=cpu，避免 generator bytes 被自动搬 CUDA；模型/optimizer 按原 loader 装载。

| 来源 → 目标（same-stage training resume） | 门禁 |
|---|---|
| CPU v1 → CPU v1 | 原合同保留，合成独立进程续跑回归通过 |
| CPU v1 → CUDA v2 / CUDA v2 → CPU v1 | 拒绝 format/fingerprint 不匹配 |
| CUDA v2 off ↔ BF16 | 拒绝 AMP policy/fingerprint 不匹配 |
| CUDA v2 同 policy | metadata 门禁通过；不等于真实 CUDA 训练续跑认证 |
| 历史 mean checkpoint → 本轮模型 | 仅 weights-only strict smoke，不能伪装 MC training state |

GPU toy 更新 2 次、skip 2 次验证实际生产 _mc_optimizer_step 的 pending/count/progress。真实四次不创建 optimizer，三次 MC owner 均 finish_update(False)，成功更新数始终 0。

## 实际平台与数值范围

PyTorch 2.8.0+cu128 / CUDA 12.8 / cuDNN 91002；RTX 5070 Ti capability 12.0，driver 570.133.07。实际调用 is_bf16_supported(including_emulation=False)=true；安装版本源码与 [官方 v2.8 CUDA API 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/__init__.py) 核对，不依赖可能已升级的 stable 文档。

策略不改：threads/inter-op=10/10、deterministic=false、matmul TF32=false、cuDNN TF32=true、benchmark=false、cuDNN deterministic=false、matmul precision=highest。未设置环境中的 CUBLAS 策略。

atol=1e-5、rtol=1e-4 只用于预登记小例和同实际输入 loss smoke；RNG bytes/IDs/counters exact。浮点测试通过不意味着 CPU/GPU 逐位等价，参见 [PyTorch 2.8 numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)。BF16 上游实际 u/C 可不同，必须分别与各自输入的参考比较。M4 合成 raw-chain 与隔离 CE 参考的区别见 REVIEW，不能把 CPU 构造 C 冒称捕获 GPU C。

新增参数、buffer、extra_state=0。未认证 FP16、完整训练续跑、GPU observer neutrality、采样/diffusion pipeline 或性能增益。
