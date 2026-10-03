# Canonical Stage-A 完整 bank 与 marginal-preserving 配对审计

日期：2026-10-03（上海）。阶段状态：**COMPLETE**。这不是此前 synthetic-only / BLOCKED 报告的覆盖版。

## 1. 结论与证据边界

在冻结的 canonical A 完整 trajectory bank 内，只破坏多人 sample 配对、保持每人的 P20 完整轨迹集合，JADE 平均增加 **0.033918 m**，JFDE 增加 **0.077423 m**，RME 增加 **0.067510 m**；逐人 marginal 不变。五个 inference seed 的 JADE/JFDE 平均 delta 都为正。

因此，当前 bank 的原始配对相对预登记的随机配对 null 有实际 joint 指标优势。效果主要集中于 all-active 窗口；mixed 的 JADE/JFDE 接近零。不能据此确认 pair energy 中不可分解交互项的因果贡献，更不能宣布 held-out 泛化、结构创新或值得重训。

另有独立边界：中断后 seed 2036 的跨进程重跑不是逐位复现。完整新 seed 被整体采用，没有按指标择优拼接。原因未定位；本轮 COMPLETE 指 bank 来源、同次 forward parity、负控和真实配对分析完成，不表示跨进程 bitwise replay 已通过。详见第5节。

快速读取：[SUMMARY.json](SUMMARY.json)；完整原始结果：[RESULTS.json](RESULTS.json)。

## 2. 路径与实际源码

本轮只运行 A：原 Stage-A epoch13 checkpoint，canonical config，strict-no-z，全零 corrector 的既有 eval bypass。没有为导出关闭 correction 字段，也没有改变模型运算。

B 是同 A checkpoint correction off；C 使用 V2-A checkpoint **但 correction arithmetic 关闭**；D 才是实际 correction-on V2-A。本轮没有运行 B/C/D，也未重复已关闭的 A/B/C 认证。

| 来源 | 实际值 |
|---|---|
| 模型、配置及复用历史 helper 的等价来源 | `d1a1382ae324b9096ac3a6d8de812ffbbb8d9780` |
| 最初导出及复用 seed 2035 的生成提交 | `e4c36036be986d04f9a14a43ff73582ba85ac564` |
| 恢复 exporter、seed 2036–2039、CPU adapter | `aa7022ef8acd8fa500b4959588765a21e9daa004` |
| 输入 cache pin | `4b75110a571e6fd200938ff951a3a218ca4eb1e8` |
| checkpoint SHA256 | `699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb` |
| config SHA256 | `bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62` |
| frozen CPU kernel SHA256 | `80aeae22f7d0a488779e5920a9093519ce05329d82e9361b72ae6f080e4d3225` |

88 个模型/配置/helper 文件内容核对通过；src/、configs/ 没有相对身份修复来源的变化。工具先提交、后执行，原生产 loader 完整检查来源缓存，checkpoint strict load 及 exact-zero/eval 检查通过。未修改 pin、manifest、缓存、权重、sampler、历史指标或报告。仅新增/修改独立 tools 和新归档。

最终归档提交由本目录所在 Git commit 确定；发布后返回固定 commit URL，不在报告里猜测自引用 SHA。脚本及输入完整指纹在 [EXECUTION.json](EXECUTION.json) 和 [BANK_MANIFEST.json](BANK_MANIFEST.json)。

## 3. 完整 bank 与坐标合同

ETH validation 原139窗，inference seeds 2035–2039；每 seed P20/T12/K21，两轮 sampler。共有 **139 个物理窗口、695 条 scene-seed 记录**，每 seed 368 个有效 agent 实例。不是695个独立场景，也没有将五个 P20 拼成 P100。

GPU forward 沿用 CUDA/BF16 autocast；捕获的 Y 和 GT 实际均为原生 **torch.float32 → NumPy float32，无转换**。通用 BF16→FP32 无损存储分支另有合成验收，但不冒充这批真实 Y 的 dtype。

同一次 forward 后复用 velocities_to_predictions 积分一次，按生产 down_factor=8 与 scene.make_world_coord_torch 变换，截取 future [8:20]，存 Y[N,20,12,2]、GT[N,12,2]，world metres、dt=0.4。捕获/NPZ/JSON 序列化未消耗 Python、NumPy、Torch CPU/CUDA 采样 RNG。

numeric NPZ 含完整轨迹、GT、masks、frame/agent/window/source IDs、原样本索引、候选 IDs、joint goals、history-only graph、degree/components。K轴、P轴与额外 trunk 分离；原世界的 edge relation embedding 不当作 independently shuffled worlds 的关系特征使用。图不由 GT 未来重建。

## 4. 资源授权与 SDD

初次归档因 SDD 占用唯一GPU而 BLOCKED，历史报告原样保留。随后用户明确“现在停”，并批准保留桌面图形进程用于诊断，这是对原不干预 SDD 边界的明确覆盖。

19:33:40 对经核验的 SDD pipeline/trainer/worker 进程组发送 SIGTERM，19:33:51 确认退出。保留可读的 epoch5 checkpoint/optimizer/scheduler 备份；未保存的 epoch6进度不保留，checkpoint 无完整 RNG 状态，不能承诺无中断等价 resume。[停止记录](SDD_STOP_RECORD.json)。

新导出显式允许 allowlisted desktop G 进程，仍拒绝任何其他 compute owner；没有停止桌面。导出于20:07:15完成，随后 CPU 审计设置 CUDA_VISIBLE_DEVICES 为空。20:13:48 复核 GPU compute owner 为空。**SDD 未自动恢复；没有训练、D推理、能量消融或缓存重建。**

## 5. 中断、复用及非逐位恢复边界

最初导出随聊天中断后进程退出，保存了 seed2035 的139窗、seed2036 前36窗，未生成 COMPLETE manifest。确切退出信号与可能未落盘的 in-flight forward 数未知。

恢复认证旧代码、标记哈希、resolved args、所有175条保存记录及其 baseline；只逐字节复用完整2035，保留旧2036前缀但全部排除。没有窗口级 RNG 状态，故从 isolated seed2036 的起点重跑，而非接第37窗。之后完整生成2037–2039；进程使用独立 session，日志持久化。

最终 bank = 139复用 + 556新 forward = 695记录。两次尝试合计至少731个已捕获 forward；不是一次无中断695次。未知尾部未伪造计数。每 seed 的生成提交单列，不能全部标成 aa7022e 新生成。

旧/新2036前36窗比较：31条数组合同完全一致；另4窗仅原 edge relation embedding 出现最大约 8.94e-8 差异；window13 的候选配对和 Y 也改变，Y最大绝对差2.265572 m，JADE delta=-0.099445 m、JFDE delta=-0.099960 m。35/36窗 Y 与 baseline 一致。候选/模型输入与配置来源核验通过，resolved args 除 model_dir/save_dir 外相同；**差异根因尚未定位，不宣称已由“CUDA非确定性”解释。**

没有因此挑选旧/新更优窗口或额外重跑。新2036整体进入 bank，之后只回答该固定 bank 内的配对问题。[详细差异](RESTART_PREFIX_COMPARISON.json)、[旧记录验收](RECOVERY_PRECHECK.json)。

## 6. 来源、parity 与负控

- 真实695条记录全部完成文件 SHA256、数组 shape/dtype/byte 合同、finite、样本完整/等权、稳定身份、同源 GT/masks/graph、跨 seed 顺序与源码验证。
- CPU float64 adapter 与同次 forward evaluator 的逐agent、逐scene、汇总基线共 **4,890** 项对照通过。预登记 atol=rtol=1e-6，最大误差 **1.6902512500749367e-7**，未放宽容差。
- 100 master seeds 625872300–625872399；沿用原 PCG64 与稳定 SHA256 身份派生，common 和 independent 各69,500个 scene-seed/replicate 组合。
- 完整轨迹整体 gather，P轴 aux同步；inverse-gather bytes/dtype恢复、逐agent marginal、common/N1负控均通过，预登记 atol=rtol=1e-12。最大 common/N1误差5.551115123125783e-17。
- 独立回读695个明细NPZ哈希/shape通过；每条scene/replicate marginal指标变化恰为0，common所有指标最大误差5.551115123125783e-17。[明细验收](DETAIL_VALIDATION.json)。
- 真实 bank 的 original exact JADE tie records=0，independent tie率=0，common first-index tie变化=0。保留官方 first-index规则及tie-set mean/min/max诊断；没有改为近似tie。
- 本批全部metric mask完整，partial-mask RME省略数=0。对未来partial-mask仍沿用省略策略；没有在审计中修复生产RME口径。
- 原7类CPU合成验收和44项拒绝测试通过，包括真实数据未覆盖的多人E=0、partial-mask和exact ties；不将合成测试算作真实模型收益。

汇总 marginal/common delta 的约1e-16偏差来自归约浮点舍入，不是收益；原始 JSON 的 exact-comparison ECDF 可能因此在这些不变控件上显示0或1，**不应解释成 p值或显著性**。

## 7. 总体与逐 seed 结果

delta = shuffled − original；对于这些误差/碰撞指标，正值表示随机打散后的结果更差。marginal按有效agent实例加权，joint按scene归约。下面五seed汇总只是固定 bank 的描述性统计。

| 指标 | Original | Independent均值 | 平均delta | delta的5%—95%分位（非CI） |
|---|---:|---:|---:|---:|
| minADE (m) | 0.278201 | 0.278201 | 0.000000 | 0.000000 — 0.000000 |
| minFDE (m) | 0.386215 | 0.386215 | -0.000000 | -0.000000 — -0.000000 |
| JADE (m) | 0.413185 | 0.447103 | 0.033918 | 0.029744 — 0.038074 |
| JFDE (m) | 0.701908 | 0.779330 | 0.077423 | 0.068506 — 0.087379 |
| RME, legacy full-mask (m) | 0.297907 | 0.365417 | 0.067510 | 0.061675 — 0.073398 |
| JMM CRmean (比例) | 0.035513 | 0.041569 | 0.006056 | 0.004386 — 0.007899 |
| JMM CR-JADE first-index (比例) | 0.034387 | 0.038701 | 0.004314 | -0.002226 — 0.011754 |

JADE、JFDE 相对原值分别恶化约 8.21%、11.03%。CRmean平均增加0.605643个百分点；CR-JADE平均增加0.431403个百分点，但其置换分位区间跨零，不能给出统一避碰改善结论。

| 推理seed | ΔJADE (m) | ΔJFDE (m) | ΔRME (m) | ΔCR-JADE (比例) |
|---|---:|---:|---:|---:|
| 2035 | 0.029871 | 0.082145 | 0.069337 | -0.004584 |
| 2036 | 0.038430 | 0.090424 | 0.074771 | 0.014302 |
| 2037 | 0.034812 | 0.071465 | 0.059962 | -0.001999 |
| 2038 | 0.033062 | 0.070357 | 0.063898 | 0.012871 |
| 2039 | 0.033415 | 0.072724 | 0.069583 | 0.000980 |

JADE/JFDE 五个seed方向一致，不等于五次独立训练验证。CR-JADE在2035和2037的均值反向；保留混合证据。

## 8. 图活动分层

| 分层 | 物理窗口/seed | scene-seed记录 | ΔJADE (m) | ΔJFDE (m) | ΔRME (m) |
|---|---:|---:|---:|---:|---:|
| all-active | 62 | 310 | 0.075801 | 0.174382 | 0.147726 |
| mixed | 30 | 150 | 0.000498 | -0.001663 | 0.007496 |
| singleton | 47 | 235 | 0.000000 | 0.000000 | 0.000000 |

原始 E=0/mixed/all-active 覆盖为47/30/62窗；**47个E=0窗口全部N=1，本批没有真实多人E=0窗口**。单人不变符合负控，但不能外推多人E=0的joint指标也应不变。

主要优势出现在all-active；mixed 的JADE约+0.000498 m、JFDE约−0.001663 m，不能说各种图状态一致受益。activity只是描述性分层，N、E及场景难度相互关联，不能把分层差异解释为“边激活”的因果效应。N/E/max-degree/component-sizes完整分层及逐replicate数值保留在原始RESULTS。

## 9. 不确定性与不可推出的结论

这139窗均来自同一原始sequence：eth5/eth/val/biwi_eth.txt，独立sequence数=1；窗口有重叠，五个seed是推理采样重复。按预先固定source单位报告描述性效果，confidence_interval=null，不将 windows×5×100当作独立样本；置换分位不是总体泛化置信区间，ECDF也不作为因果检验p值。

ε_joint保持null，未临时选阈值。ETH valid/test缓存口径不能作为新增独立held-out证明。本实验不干预energy，不区分单边ranking项与不可分解交互项，不解释此前完整A/B差距，也不证明启用correction的V2-A与A相同。

只在保存的bank内重配完整轨迹：若真实重新生成多人的轨迹可能随关系上下文变化，本审计不回答那种反事实。跨进程重启差异另列为复现风险，不以同次forward parity掩盖。

## 10. 归档、持久位置与唯一下一步

网页端建议先读本报告及SUMMARY，再读原始RESULTS、BANK_MANIFEST、EXECUTION和RESTART_PREFIX_COMPARISON。大JSON若GitHub不预览，使用Raw下载；它们是原始产物的逐字节副本。

持久 bank（仓库相对路径）：

`outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_banks/canonical_a_recovered_20261003_aa7022e_seed2035_2039/`

CPU明细及原始结果：

`outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_bank_pairing_20261003_aa7022e/`

这些持久NPZ不提交到GitHub。manifest中的NPZ路径相对本地bank根，RESULTS.details相对本地CPU审计根，不能误解为本报告目录内有数据。明细逐文件hash在RESULTS，bank逐文件hash在BANK_MANIFEST。

| 原始产物 | SHA256 |
|---|---|
| BANK_MANIFEST.json | `ac6cd236d1f3aa1828dfe2548abe3a094ee7989dca1a7720d76771a872853ed1` |
| RESULTS.json | `0061d510cd0e047d7e12e5d97e729c0f63a7671c73514d6df3b3746f3e098367` |

其余归档文件见ARTIFACT_HASHES.json（不自包含其自身hash）。历史5a0ae41及a9a220c的BLOCKED/合成结果全部保留，不重写为成功。

**唯一下一步：将本轮“all-active集中优势、mixed弱效应、CR-JADE混合、重启非逐位一致”的证据交给网页端做只读go/no-go复核。** 不自动进入能量I消融、geometry6、Stage-B扩展、重训或SDD恢复；任何新实验需另行授权。
