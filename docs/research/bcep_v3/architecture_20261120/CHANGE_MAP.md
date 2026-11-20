# v3 实际代码差异

固定父提交：`468c5eddde48bdeec47c4bb3c39418352be16bb9`。没有复制 SHARP、SRefiner、ModeSeq 或 U2Diff 的模型代码/权重。

| 文件 | 具体改动 | 不变量 |
|---|---|---|
| `ebjd/unified.py` | 将原 scorer 同一前向的 candidate_norm 输出显式返回为 `forward_features` | 原 `forward` 参数/返回和 state_dict 键保持；无第二遍计算 |
| `bcep_v2/joint.py` | 移除候选特征临时 forward hook，使用显式接口 | 原初配、一次旧反馈、终配和坐标 ST 完整保留 |
| `ebjd/denoiser.py` | 仅在安装 `grg_ctf` 时对原 physical feedback / future bias 使用新 gate | 无新增 FJAB、无额外 denoiser/DDIM；v2 分支不变 |
| `bcep_v3/evidence.py` | K20 相对几何偏置 self-attention；零输出残差；显式缺失统计 | 不移动候选，不使用序号 embedding，不接受未经验证 sidecar |
| `bcep_v3/geometry.py` | 观测 CV baseline 到每个实际终点的 12 帧参考路径；真实候选对几何 | 先计算 pair geometry 再按 P0_ST 加权；无平均终点几何 |
| `bcep_v3/coupling.py` | 一层同 world social attention；关系 bias/value；零初始化 ScoreDelta | 恰好两次 Hungarian；旧 feedback 先完整执行；最终双射与 keyed noise 同步 |
| `bcep_v3/gate.py` | 一次小 MLP，条件为实际 bridge goal、当步粗轨迹、observed social_h、时间和距离 | gate 0.5–1.5 倍 alpha²；对称 pair gate；无未来真值参数 |
| `bcep_v3/model.py` | 严格 v2 checkpoint 载入后只初始化三个新增前缀 | 每个旧 tensor SHA 不变；无 `strict=False`；局部 seed 不扰动旧 RNG |
| `bcep_v3/contracts.py`, `fixtures.py`, `stage2.py` | 输入单位/成员/帧映射检查；只允许内置 synthetic adapter 调原 joint_callbacks | 实际 grouped backward 与原 agent/scene 权重；真实训练入口关闭 |
| `scripts/v3/audit.py` | 原生旧权重迁移、20-step 对齐、实际 run_stage 恢复、计数受限的 GPU 性能测试 | 每次 AdamW.step 前先记账；总预算 32；无队列 |
| `tests/test_bcep_v3.py` | 初态、非零残差置换、mask、单人/57人、真实 pair geometry、负对照 | 明示 Hungarian 并列解边界；不以大容差掩盖近并列换解 |

`bcep_v2/objective.py`、`bcep_v2/stages.py`、`bcep_v2/state.py`、`bcep_v2/provider.py`、`ebjd/representation.py` 和 `bcep_joint/inference.py` 未改。正式锚点配置中没有名为 final_coordinate 的开关；它的有效实现就是最终配对混合坐标 ST。v3 将该名称显式锁定，并拒绝 candidate_cost/shared_world_bounded。
