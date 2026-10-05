# 输入、模式与梯度合同

版本：jdv2.expected-conditional.cpu.v1；默认 mean_energy，缺失新 args/config 字段也视为 mean_energy。默认分支不创建/抽取新 RNG。

显式参数：

```text
--jdv2_goal_objective expected_conditional_mc
--jdv2_neighbor_draws 4
--jdv2_neighbor_seed <optional integer in [0,2**63)>
--device cpu
--amp_enabled False --amp_dtype fp32
```

必须 active JDV2、strict_no_z、joint_goal、use_scene_latent=False；拒绝 SDD、V4、independent、其他 stage、未知目标、S 非 4、CUDA/reduced precision/AMP。没有忽略门禁或 CPU fallback 选项。旧模式不新增设备/精度限制。生产 YAML/启动脚本未修改。

MC 方法只在 model.training=True 下允许；eval-mode diagnostic loss 在 encode/draw 前拒绝。metrics 的额外 teacher diagnostic 对 MC 跳过，原模式维持。weights-only evaluation 不恢复目标 RNG；可用默认 mean 模式加载 MC 模型权重。

## 数据与计算

encode(for_loss=True) 仅一次。候选/GT 建立原 endpoint soft target q；q.requires_grad 必须 False，不以 detach 隐藏非法可训练标签。teacher 的 log probability 是另一个张量，经 Cpost/KL 保持梯度。

u/q：[N,K]，dense CPU，同 FP32 或 FP64；有限、N/K>0；q 非负、行和在原纯函数容差内为 1，masked q 精确零。mask bool、每人至少一个有效候选。scene int64[N] 可不连续/为负。edge int64[2,E] 要求唯一 canonical src<dst、合法索引、同 scene。非法标签/支持/scene/device/dtype 在 draw 前拒绝；正常 E0 仍抽一次。

既有 build_soft_goal_target 与 cost 网络有 FP32 边界，本轮不重写。接线数值/梯度 fixture 使用 FP32；纯函数原 FP64 验收和 RNG FP64 输入验收不外推到完整网络。新接线只将固定标签 dtype 对齐 u，不搬 trainable cost 到 CPU，也不转换候选精度。

z = multinomial(q,4,replacement=True,独立generator).T.contiguous()，shape[4,N]。post/prior 用同一 z 对象。各 chunk 保留原：
Cprior = −logsumexp(log p − E)，Cpost = −logsumexp(log q_relation − E)。
dynamic/energy ablation 分支沿用；teacher 不 detach。CE 完成全部 chunk 后才归一化，scene 等权再 draw mean；KL 边 mean。loss keys 和 .5/.5/beta 不改。beta = .1 min(progress/.2,1)，progress 按 successful updates / 固定总计划。

没有 sampler/ST/额外诊断抽样。双 cost 列表和共享能量图在 backward 前保持存活；内存不严格 chunk-bounded。新增 Parameter/buffer/extra_state 为 0，架构/维度/候选/图/推断/噪声不变。

## 验收

[测试](../../../../tests/test_jdv2_mc_wiring.py) / [fixture](../../../../tests/mc_wiring_fixture.py) / [预登记](AUDIT_PLAN.json)。FP32 atol1e−6/rtol1e−5；FP64 pure loss atol1e−10/rtol1e−8；RNG、IDs、counters 和固定 CPU resume exact。只用 raw Parameters/CPU toy optimizer，未实例化真实网络或读取真实数据。
