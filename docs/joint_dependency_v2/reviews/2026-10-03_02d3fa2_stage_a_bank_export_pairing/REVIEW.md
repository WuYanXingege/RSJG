# Canonical Stage-A bank 导出准备与 CPU 配对审计验收

日期：2026-10-03（Asia/Shanghai）。阶段状态：**BLOCKED_RESOURCE_UNAVAILABLE**。

本轮已实现并提交独立 exporter、numeric NPZ loader、来源与 CPU parity 门禁、真实 bank 配对审计入口和 CPU 验收脚本。**没有启动 GPU 导出，没有完整真实 bank，没有真实配对收益结果。** 当前只有一张 GPU，SDD 仍持有计算进程；按任务要求停止在资源门槛，不改变 SDD。

本报告不是上一轮缺 bank 报告的重复：新增的是可执行、已做合成 CPU 集成测试的导出与验收实现。它也不是 GPU 导出成功或真实 bank 已认证的报告。

## 1. 提交与状态分层

| 项目 | 实际值 |
|---|---|
| 开始时 HEAD / 远端分支头 | 5a0ae411f401fe77b9ec4f114d5c4bfa45109a4e；tracked 工作树干净 |
| 独立工具实现提交 | 02d3fa2876e7a2dd663b9a43df934400cf0c1c99 |
| 模型/预处理源码等价版本 | d1a1382ae324b9096ac3a6d8de812ffbbb8d9780；相关 src/configs 无改动 |
| 输入 cache source pin | 4b75110a571e6fd200938ff951a3a218ca4eb1e8；既有配置中的显式 pin，未修改 |
| 本轮实际 export_source_commit | null，尚未执行导出；不能拿工具提交冒充数据生成提交 |
| 最终归档提交 | 由包含本页的 Git commit 与最终固定链接确定，不提前填自引用 SHA |
| 本轮真实覆盖 | 0 个已导出窗口、0 条 scene-seed 记录、0 个可用 inference seeds、0 次模型 forward |
| 目标覆盖，尚未获得 | 每 seed 139 窗，2035–2039 共 695 条记录；各 seed 保持 P20，不是 695 个独立 scene 或 P100 |

详细阶段字段见 [RESULTS.json](RESULTS.json)。[BANK_MANIFEST.json](BANK_MANIFEST.json) 是明确标为 blocked 的准备清单，bank_path / bank hash 均为 null；它不是伪造的完整数据 manifest，不能被真实 loader 接受。

## 2. 资源门槛与 SDD 保护

只读快照：2026-10-03 19:18:00 +08:00。

- 唯一设备：RTX 5070 Ti，UUID GPU-078a9326-675c-cb94-af4a-ac46eb77c484。
- nvidia-smi 报告总显存 16,303 MiB、已用 8,888 MiB；计算进程 PID 962176 持有 8,192 MiB。
- 19:18:04 的进程元数据显示，该 PID 仍为 gdts_paper_sdd_goal_pretrain_seed2025，父进程 962146，goal_pretrain、FP32、num_workers=1。
- 没有调整/暂停/终止 SDD，没有改变其 worker、线程、checkpoint、配置或 cache。这是时间快照，不是持续健康监控。

未执行 export 命令，也未设置后台等待/自动启动。CPU 测试仅限制自身进程：nice 10、单进程、BLAS/Torch 1 线程、CUDA_VISIBLE_DEVICES 为空；没有构造 GDTS 实例、没有反序列化 checkpoint、没有初始化 CUDA。

## 3. 回读、已有输入与来源兼容性

完整回读上一轮的 REVIEW、RESULTS、INPUT_MANIFEST、INPUT_SCHEMA、DISCOVERY、audit_cpu，以及索引、identity validation 和 post-fix paired 证据。上一报告 SHA256 为 8177bdfab37b146fc02f375f58bf38c008e4a5c1b7ffb3ad76386e194202c261；RESULTS 为 647d66130ea1517eea0a5fda004e702dcc3601af91d30338377525e53f4a9af7，均匹配。历史 BF16/FP32/paired JSON 的 SHA 仍匹配；没有重做 A/B/C 或 C/D 推理。

只检查已知 ETH 输出范围中新出现的 bank/NPZ 名称，没有重复上轮全盘式文件清单，没有外部新 bank 位置输入，未找到可复用的合格 bank。旧 D/JMM253 导出仍不能替代 A139。祖先、仓库及本轮工具/文档目录未找到适用 AGENTS.md。已有未跟踪 data/cache/runtime 工作全部保留。

提交后 metadata preflight 已核对：

| 输入 | SHA256 / 结果 |
|---|---|
| strict-no-z Stage-A epoch13 checkpoint 文件 | 699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb |
| 原 canonical A config | bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62 |
| 原 cache manifest 文件 | eadae55391b45224a3d7920e6128764db9a56298ffe70d723030ec00193786dd |
| manifest 的既有 stable JSON hash | 2d525a0f441a148dc6ff25c8f14eae3bd10d45b764cac17a25553ab76b93b949 |
| 物理 valid 文件名/部署 cache 记录存在性 | 139 个；顺序写入 PREFLIGHT |
| src/configs 与复用 helper 指纹 | 88 个文件内容与身份修复版本兼容，见 PREFLIGHT |

**边界：这些是 metadata 与文件字节 hash 验收。完整 source-cache split 内容重哈希、production manifest validator 和 checkpoint strict load 尚未运行。** exporter 在空闲设备上调用不变的 _active_evaluator；其现有 loader 使用配置中的精确 source pin，同时严格检查其他 manifest 字段。没有伪造当前源码 commit、修改 cache manifest、放宽 strict load 或调用 main.py 的 preprocessing 入口。输入异常将停止，不重建缓存。

## 4. 新增实现

| 工具 | 功能 |
|---|---|
| [jdv2_export_stage_a_bank.py](https://github.com/WuYanXingege/RSJG/blob/02d3fa2876e7a2dd663b9a43df934400cf0c1c99/tools/jdv2_export_stage_a_bank.py) | metadata preflight；GPU owner 门槛；唯一 A 路径五 seed 导出 |
| [jdv2_stage_a_bank.py](https://github.com/WuYanXingege/RSJG/blob/02d3fa2876e7a2dd663b9a43df934400cf0c1c99/tools/jdv2_stage_a_bank.py) | 原子不覆盖写入、numeric NPZ、来源/覆盖/指标验收 |
| [jdv2_audit_stage_a_bank.py](https://github.com/WuYanXingege/RSJG/blob/02d3fa2876e7a2dd663b9a43df934400cf0c1c99/tools/jdv2_audit_stage_a_bank.py) | 通过完整认证后，调用原 kernel 做 CPU 配对分析与明细归档 |
| [jdv2_stage_a_bank_cpu_checks.py](https://github.com/WuYanXingege/RSJG/blob/02d3fa2876e7a2dd663b9a43df934400cf0c1c99/tools/jdv2_stage_a_bank_cpu_checks.py) | 合成 capture→NPZ→loader→parity→置换集成测试 |

导出不修改 use_dependency_corrector，canonical config 中仍为 true；实际全零 eval bypass 由既有 model.py 执行。A=canonical Stage-A；B=同 checkpoint correction off；C=V2-A checkpoint correction off 基底；D=实际 correction-on V2-A。本轮不运行 B/C/D。

每窗口只调用一次 encode 与一次 ts_sample。保留与历史工具相同的分开的 BF16 autocast 上下文、isolated inference seed、loader 顺序及 sampler seed/window context。velocity 只经原 velocities_to_predictions 积分一次，再按生产指标执行 down_factor、逐 sample scene.make_world_coord_torch、相同 GT 变换，截取 future 并存为 Y[N,P,T,2]。BF16→FP32 仅为无损存储提升，记录原/存储 dtype。

同次 forward 保存 Y、GT、masks、frame/agent/source/window IDs、等权样本、history-only cache graph、degree/component、selected IDs、P 轴 goals、K21 candidates、单独 trunk、H、输入 fingerprint，以及生产逐 agent/scene 指标列表。per-edge relation embedding 保留为原 world 上下文；独立 agent 重排后该条件关系不再有对应意义，不冒充 shuffled relation 或参与新指标。可按 agent 重排的 P 轴 aux 全部同步 gather。

写入只允许新 stage_a_banks 子目录；临时文件 fsync 后通过同文件系统原子 hard-link 发布，拒绝覆盖，删除的仅是本工具自己的临时链接。每个完成窗口有不可变 sidecar；最终完整 manifest 最后发布。失败写 FAILED.json、已完成 scene/seed 键与 forward 数，不自动续采或覆盖。初始 INCOMPLETE 标记保留为审计记录，只有完整最终 manifest 才能建立完成状态。

## 5. CPU 验收：本轮实测

详见 [CPU_CHECKS.json](CPU_CHECKS.json)，以及 [EXECUTION.json](EXECUTION.json)。

- 7 类全 P20/T12 合成 fixture：singleton、多人 E0、mixed、all-active、partial mask、exact JADE tie、可控配对变化。
- 每类使用预登记 100 master seeds；新明细 adapter 与未改动的旧 kernel 逐项对照，随机协议/指标定义未重写。
- 28 个拒绝/保护测试通过，含非法 hash、单位、dtype、mask、样本权重、IDs、frame、graph/component、K/P 对应、非数值容器、错误 route/checkpoint、已存在输出和 busy GPU。
- 捕获使用生产 compute_model_metrics 函数与 ETH 世界变换函数；没有构造模型。与 float64 CPU kernel 对照的最大误差 **2.402860368455606e-7**，通过原 atol=rtol=1e-6。
- common/N1 控制最大误差 **0**，原阈值仍为 1e-12；inverse gather bytes/dtype、CPU RNG 不变、BF16 存储 round-trip、按 agent 加权汇总均通过。
- exact-tie fixture 的 common first-index CR-JADE 有 92/100 次变化；这是不同 stable-ID fixture 下的合成结果，不覆盖旧报告的 94/100，更不是实际 A tie 率。tie-set 附加指标保持不变。
- CUDA capture RNG 检查已实现，但**尚未执行**；真实 bank loader 的完整 695 记录来源闭环、真实 CPU parity/负控/效应均 **NOT_RUN**。

loader 首先核对 manifest/逐文件 bytes hash、代码 commit/fingerprint、输入 fingerprint、路线、单位、dtype/shape、IDs、GT/masks、graph、K/P axes、139×5 顺序与历史 graph/agent 覆盖。再进行逐 agent/scene/seed 汇总 parity；任何首个超差均停止，不放宽容差、不跳过异常窗后声称完成覆盖。partial-mask RME 省略，不修生产口径。

真实分析入口输出逐窗口/seed/replicate 明细、共同与独立置换、N/E/degree/component 分层和按 raw sequence 的汇总。五 inference seeds 与 100 置换都不是训练重复；重叠窗口不作为独立样本。按预登记 source-sequence 单位报告描述分布，不在 cluster 不足时生成 CI。ε_joint 继续为 null，不据临时阈值宣布结构优化 Go。

## 6. 待设备空闲后的可执行命令

以下命令**本轮未执行**，从 RSJG_JDV2_clean 根目录运行。无需修改模型配置。每次启动仍必须确认 GPU owner；脚本也会在模型 setup 前及每窗口检查其他计算进程。该检查不是硬件资源锁；检测到其他 owner 时只停止自己的导出，绝不干预对方。若目录已存在，停止并核对已完成记录，不能删除旧结果后盲目重跑。

~~~bash
stage_a_bank_dir=outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_banks/canonical_a_02d3fa2_seed2035_2039
nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.conda/rsjg/bin/python -B tools/jdv2_export_stage_a_bank.py export --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484 --output "$stage_a_bank_dir"
~~~

只有 exporter 输出完整 manifest 后，才运行：

~~~bash
stage_a_bank_dir=outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_banks/canonical_a_02d3fa2_seed2035_2039
stage_a_bank_sha256="$(sha256sum "$stage_a_bank_dir/BANK_MANIFEST.json" | cut -d ' ' -f 1)"
nice -n 10 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.conda/rsjg/bin/python -B tools/jdv2_audit_stage_a_bank.py --manifest "$stage_a_bank_dir/BANK_MANIFEST.json" --manifest-sha256 "$stage_a_bank_sha256" --output outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_pairing_audits/canonical_a_02d3fa2_seed2035_2039
~~~

规划的 bank 目录当前未创建，不当作可读取数据路径。目录名中的 02d3fa2 仅标记工具准备版本；实际 export_source_commit 会记录当时 HEAD，可能是后续文档归档提交。数据/NPZ 保存在持久输出，不提交 GitHub；后续报告需给路径与 SHA。

## 7. 结论与唯一下一步

本轮结果为 **BLOCKED_RESOURCE_UNAVAILABLE，CPU 准备完成、GPU 导出未运行**。不能由此推断 A 的配对收益，也不能据此证明 pair energy 的因果贡献或 held-out 泛化。

唯一下一步：**确认有空闲 GPU 后，执行上述 canonical A 五 seed 单路径导出，再通过同次 forward CPU parity 门槛完成真实配对审计。** 不自动扩展能量 I 消融、geometry6、Stage-B 或重训。SDD 保持原运行。

## 8. 归档指纹

| 文件 | SHA256 |
|---|---|
| [RESULTS.json](RESULTS.json) | ce684ae7051b21749de5cffd569c9ddd8dd1abf355311e91abfb79c1d59be030 |
| [BANK_MANIFEST.json](BANK_MANIFEST.json) | f1631fe274ab1659c53cd658d41a660a59e59f862749e5789e5719c7f7250aff |
| [CPU_CHECKS.json](CPU_CHECKS.json) | 7c9b9d09bb13c7046994e6080a3a07088ddde4be99ad730f9032d09b024cd49b |
| [PREFLIGHT.json](PREFLIGHT.json) | cd7a92d37ad89bf0ba368796fe601e8a846640f1185165502731b231c2043e93 |
| [EXECUTION.json](EXECUTION.json) | fcc8dc90bb8f3d5c4561b1a5ee07be34fc2ee26a39e55e5ab4a743ade17dd957 |

全部旧报告/JSON 保留，只新增本目录与索引一行。是否已上传及远端字节核验，以最终固定 commit 链接与回读结果为准。
