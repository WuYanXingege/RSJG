# BCEP–EBJD v3 架构升级与工程验收

**已实现 CSEE、HCRA、GRG-CTF；工程验收 PASS。新架构真实训练更新=0，尚无精度提升证据。** 累计合成 GPU optimizer updates **31/32**，没有真实数据前向/训练，没有恢复旧队列，没有创建后续队列。本轮所有验收进程已退出；正式五折保持 PAUSED_RESUMABLE，candidate_cost 保持 SESSION_BUDGET_PAUSED，sharedworld 已 COMPLETE。

正式起点 `468c5eddde48bdeec47c4bb3c39418352be16bb9`；sharedworld 负结果归档 `76d80b508ad79f52965c15109b58d2d835473fd7`。独立分支 `research/bcep-ebjd-v3-architecture-20261120`，独立本地输出名 `bcep_v3_architecture_audit_20261120`。不以 main HEAD 代替正式源码。逐 worktree HEAD、源码哈希、资产检查见 [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) 和 [ASSET_PROTECTION.json](ASSET_PROTECTION.json)。

## 源码核验与原调用链

正式 v2 的 Provider 是5维运动输入、2层历史 Transformer、相对社交聚合、3层地图 CNN/全局池化和4层交替二维样条 Flow；M256 support → KMeans20 → 有界候选修正。Provider 和修正头没有修改/重训。原 spatial_encoding 的 pre-hook 读取真实 pooling 输入：128×32×32 特征，32m/256像素地图以最后观测位置为中心，world xy→pixel→stride8特征采样坐标；只有同一冻结权重/地图/变换才可复用，增广后必须重新编码。其细节见 [FEATURE_SIDECAR_CONTRACT.json](FEATURE_SIDECAR_CONTRACT.json)。

原下游链是：observed/map encoder → world-query scorer → 第一次 Hungarian/Sinkhorn ST → 八近邻（边界并列全部纳入）旧 compatibility feedback → 第二次 Hungarian → 同步 gather 终点和 keyed noise → 20步条件 DDIM。shared encoder 的 social_h 来自历史与社交分支；地图仍通过冻结 Provider 候选、bridge memory 与 CTF 进入模型，不能宣称 social_h 已融合地图。候选特征旧 hook 已改为同一前向的显式返回，没有重复 scorer。

原 objective 的配对分支使用预测候选的有效 final_coordinate ST；桥分支使用 GT endpoint 条件。实际 stages.joint_callbacks 对 agent_part/scene_part 按原人/场景分母分组 backward。v3 完整复用该函数及旧损失：diffusion、coarse、geometry、map、clean-path（0.05），未启用 candidate_cost/shared_world_bounded，也没有20世界完整路径训练 rollout。六层 FJAB 仍为3+CTF+3。正式 ctf_feedback_basis=legacy；decoder_energy 仍为可选择的固定解码能量启发式，并非严格 Jacobian。12个 latent 槽位不等于12个物理帧。

## 已实现数据流、张量与公式

| 模块 | 输入/输出 | 实际新增计算 |
|---|---|---|
| CSEE | old u:[B,N,A,192]，g:[B,A,N,2] → u:[B,N,A,192] | 每人一次K20 self-attention，4头；相对dx/dy/距离bias；零初始化out残差 |
| HCRA | P0_ST:[B,N,W,A]、u、候选对关系 → ΔS:[B,N,W,A] | E条有向邻边、每边20×20真实候选对；16维关系；每world一次跨人注意力；零初始化score出口 |
| GRG-CTF | 实际goal:[B,P,N,2]、粗物理path:[B,P,N,12,2]、social_h、t → gate:[B,P,N] | 每人每world一次202→64→1 MLP；原physical与future pair bias的有界缩放 |

W 是 world 行，A 是候选列，二者数值均为20但语义不同。`V[b,w,i,d]=Σa P0_ST[b,i,w,a]u[b,i,a,d]` 只混合隐藏特征。原初配评分与旧反馈仍用 old u；新增 CSEE u 仅服务 HCRA 增量分支。最终 `S=S_old_final+ΔS`，零出口不再乘第二个零门，恰好两次 Hungarian。未选候选只是增量评分query，不替代最终选中的真实终点。

参考路径 `ybar(s)=baseline(s)+(s/12)*(g−baseline(12))`，先为每个真实候选构建，再对真实邻居的每个(a,b)算12维关系：最小同步距离/发生时刻、均值距离、相对终点、参考相对速度模、观测相对位置/速度、基于该候选对参考速度的CPA时刻与距离。先编码 R[a,b]，再沿发送人 P0_ST 得 R[w,a]，再沿接收人 P0_ST 得 R[w]；不能先平均坐标再算非线性距离。该逻辑通过独立负对照验证。边分块64，旧反馈与HCRA复用同一邻居索引；新增聚合采用唯一键 index_copy 和确定性矩阵乘法，没有新增浮点 index_add。旧两次 Hungarian 的CPU调用仍在；新GPU合法性断言使用异步断言。

CSEE 精确使用候选相对观测/基线的位移及候选两两距离摘要。未执行历史M256支持池的无损回放，因此簇占比、簇离散度、Flow密度三字段明确缺失（值0、presence=false），非空sidecar直接拒绝。没有伪造概率，没有候选序号embedding。当前K20推导特征随候选置换等变；旋转/反射使用原增广，同步变换obs/maps/goals后重新推导。

GRG 使用 `g_i=alpha²*(1+0.5*tanh(r_i))`，有效行人 pair gate 为 `(g_i+g_j)/2`，无效行人行列清零。physical广播到[B,P,N,slot,D]；pair广播到[B,P,slot,heads,N,N]。门控输入没有未来真值轨迹参数，不借用推理配对token：训练取桥实际GT endpoint，推理取桥实际冻结候选endpoint，粗轨迹均来自当前已有前3块decode。r=0恢复原alpha²；不增加denoiser调用、DDIM步数或桥深度。

## 迁移、参数量与初始化

原生自有 Provider density epoch19/head epoch9 与 joint epoch26 权重均按Top5哈希核验。只以原生旧权重验证迁移，不把它作为后续新训练起点。不加载GDTS/教师权重。先对旧model使用strict=True完整载入，随后允许且仅允许三个新增前缀；旧334个张量逐项SHA不变，新增34个张量均有名称、shape和初始化SHA。[INITIALIZATION_ORIGIN.json](INITIALIZATION_ORIGIN.json)

参数量 v2 **7,596,022** → v3 **7,939,056**，增加 **343,034（约4.52%）**：CSEE151,364，HCRA178,613，GRG13,057。新增模块在局部CPU生成器种子3201下初始化，不扰动全局CPU/CUDA RNG；旧随机参数初态也逐项相同。最终出口为零；stage2随机初态训练计划与本次旧权重兼容审计分开。[POSTCHECK.json](POSTCHECK.json)

## 验收证据

- CPU **30/30通过**，包含新模块和正式v2回归、原Top5同分取早轮、CTF legacy/decoder_energy测试。
- RTX5070Ti、PyTorch2.8/cu128、FP32、严格确定性、math-SDPA；原deterministic pooling和bilinear_zeros保留。
- 实际epoch26非零旧反馈下，K20、N=1/2/9/57，完整20步DDIM的G/H轨迹最大绝对差全部为 **0**；初配/终配pi、对应keyed noise逐值相同，端点候选边缘不变。每次generate只调用一次assign，未在DDIM中重配。[MIGRATION_EQUIVALENCE.json](MIGRATION_EQUIVALENCE.json)
- 非零新增残差下的独立候选重排、行人重排、batch重排、mask、单人/无邻居安全性通过；N57观测位置完全并列时保留3192条有向边，输出有限。
- 真正调用未修改的stages.joint_callbacks与run_stage：连续4次更新 vs 2+2恢复的model、AdamW、scheduler、RNG、cursor/order、history/stale全部逐值相等。原原子checkpoint writer保留每个已完成epoch；少于五轮保留全部，原Top5测试另外验证五个最佳及同分规则。[RESTART_GRADIENT.json](RESTART_GRADIENT.json)
- CSEE首步梯度为0是下游零出口的预期结果，第2步起有非零梯度；最终连续第4步三个分支梯度L1和为 0.000426247 / 0.126278 / 0.0237899。逐个新增张量梯度均有限；注意softmax常量平移bias的极小梯度可能只是浮点残差，不能据此声称有效学习。新增注意力内部权重也有非零梯度。Provider不在joint optimizer中、requires_grad=false、逐值未变。
- 非法单位、帧映射、非有限候选、缺旧key、非空未核验sidecar、错误损失及真实数据入口显式拒绝。新版默认关闭真实训练；不存在自动恢复或扩折入口。

第一次随机初始化重排检查发现旧scorer近并列最优解会在微小求和差异后换Hungarian解，连旧feedback也变化。没有用宽松容差掩盖：保留原算法，用有分离度的评分测量token等变，并单列严格并列负对照。**精确/近并列Hungarian解不保证跨重排逐值稳定**，这是继承的离散选择边界；它仍保持合法双射和边缘候选集合。future实验需记录这类稳定性，不能据此宣称无条件端到端排列等变。

本轮GPU更新账本覆盖初版工程检查20次、候选对CPA收紧后的恢复重验8次、最终v3性能3次，共31次；最终额外前反向/迁移检查没有optimizer.step。被替代版本的时间不冒充最终结果。所有真实数据optimizer更新为0。[TEST_RECEIPT.json](TEST_RECEIPT.json)

## 计算成本（完整合成step）

包含CPU增广、实际分组forward/backward、clip与AdamW.step；256×256语义地图，当前GPU无其他本任务训练。测试物理group_max_slots=32；N57单窗沿用原超阈值单独处理规则。为遵守32次总预算，v2保留两次测量，最终v3每形状仅一次；表中比例只比较各自首个测量。单人初始化/缓存影响很大，**不能把这些比例当作稳定训练加速或减速结论**。

| batch×人数 | v2耗时s（两次） | v3耗时s（一次） | 首次相对差 | 首次峰值GiB v2/v3 |
|---|---|---|---|---|
| 1×1 | 0.9383, 0.1659 | 0.7873 | -16.1% | 0.249 / 0.252 |
| 4×8 | 0.7816, 0.7389 | 0.8081 | +3.4% | 1.666 / 1.668 |
| 1×57 | 1.2395, 1.1931 | 1.2715 | +2.6% | 2.299 / 2.500 |

复杂度：CSEE O(B·N·K²·D)；候选对关系 O(E·K²·12)，world attention O(B·K·N²·D)，dense relation O(B·K·N²·16)。距离并列可使E由通常约8N上升到N(N−1)。GRG另算当前P个world的物理同步距离，O(B·P·12·N²)，没有跨world注意力。该预算仅证明1×57可运行，不承诺正式训练速度或全数据峰值。[BENCHMARK.json](BENCHMARK.json)

## 方法差异与来源

| 原理参照（官方主会来源） | 本轮适配 | 明确不移植 |
|---|---|---|
| [SHARP, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Prutsch_SHARP_Short-Window_Streaming_for_Accurate_and_Robust_Prediction_in_Motion_CVPR_2026_paper.html) | 仅将附件提出的分解交互思想作为背景；具体world/candidate接口由v2约束推导 | streaming生成器、跨观测窗口缓存、双训练目标 |
| [SRefiner, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_SRefiner_Soft-Braid_Attention_for_Multi-Agent_Trajectory_Refinement_ICCV_2025_paper.html) | 从实际候选原型提取时空关系bias/value | soft-braid定义、车道结构、迭代轨迹refiner |
| [ModeSeq, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_ModeSeq_Taming_Sparse_Multimodal_Motion_Prediction_with_Sequential_Mode_Modeling_CVPR_2025_paper.html) | 显式候选集合依赖，仍输出一一置换 | 顺序模式解码、EMTA损失 |
| [U2Diff, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Capellera_Unified_Uncertainty-Aware_Diffusion_for_Multi-Agent_Trajectory_Modeling_CVPR_2025_paper.html) | 缺失证据显式标记；没有新增概率声称 | noise-NLL、RankNN、原运动生成网络 |

链接与摘要已通过CVF官方索引检索核对；直接打开页面曾返回403，未获取或复制外部网络源码。报告只借鉴一般原理，不宣称“首次双轴/拓扑注意力”，不把工程PASS当成论文创新性或SOTA。

## 能证明什么，仍缺什么

每人双射π保留经验终点边缘 `mu_i=(1/K)Σa delta(g_ia)`，所以同bank的边缘minFDE严格保持，跨人对应改变时JFDE可以改变。这是有限集合置换不变量，不是新的普适最优性定理。HCRA可能改变联合endpoint分配，但参考路径只是廉价proxy，不是生成路径，更不保证恢复真实未来拓扑或避免碰撞。

GRG在有限r时处于原alpha²的0.5–1.5倍，最多调制给定终点下的轨迹，不能直接改变JFDE，也不能保证改善JADE。潜在反例包括错误参考路径把交叉误认为不兼容、孤立人没有跨人关系、含噪地图/粗轨迹诱发不合适门控、近并列离散分配不稳定。支持池统计未验收，部分新分支梯度很小，只有一次seed的合成检验；这些都必须保留为限制。

**当前只支持“接口与工程实现可运行”，不支持“精度已提高”或自动扩折。** 唯一后续是等待审阅后单独授权的一次ZARA2 Stage2受控训练，计划见 [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md)；完整代码变化见 [CHANGE_MAP.md](CHANGE_MAP.md)。

## 复现与文件

以本次独立分支checkout及原主机环境执行；本地审计输出位置固定，已经存在的预算账本不会清零。重复执行restart不会覆盖旧完成状态，预算超限会拒绝。完整重放须在独立获准的审计资产副本中运行，不能删除历史账本/权重绕过预算。

```bash
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q tests/test_bcep_v3.py tests/test_bcep_integrated_v2.py tests/test_bcep_v2_formal_controller.py tests/test_ctf_decoder_energy.py -o addopts=''
python scripts/v3/audit.py --section migration
python scripts/v3/audit.py --section restart
python scripts/v3/audit.py --section benchmark
python scripts/v3/postcheck.py
python scripts/v3/report.py
```

首次完整默认GPU流程为8次恢复试验更新+12次性能更新=20次；本轮额外修正重验也已计入31次账本。最终性能复测使用 `--benchmark-repeats 1 --reuse-v2-benchmark <本地已核验v2基准JSON>`，详见BENCHMARK的各行重复数。所有checkpoint、原始bank、局部日志留在主机；GitHub仅代码、匿名合成统计、配置、哈希和报告。
