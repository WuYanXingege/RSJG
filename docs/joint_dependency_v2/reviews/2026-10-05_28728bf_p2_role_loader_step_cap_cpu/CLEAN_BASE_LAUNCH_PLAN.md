# Fresh P2 clean base launch plan — NOT EXECUTED

当前训练与模型执行未授权。四配置均可走真实main的只读launch-check，但真实来源未核实，结果exit2。没有clean base不能阻止准备fresh goal；recording/map UNKNOWN必须阻断设备前执行。

| stage | 初始化/用途 | packing / upper bound | selection |
|---|---|---|---|
| goal | fresh原Goal U-Net；仅train梯度 | 556pack×≤150epoch；≤83400更新 | inner10pack masked goal BCE；strict改善，tie早epoch |
| joint | 仅合格P2 goal父权重；history/diffusion fresh | 556pack×≤250epoch；≤139000更新 | inner ADE；strict改善，tie早epoch |
| cache | 合格完整base；obs-only candidate | 4249deployment/3484train teacher | 不开outer指标 |
| Stage-A L1 | 同一合格base+fresh heads共同初态 | native3484，H≤500 | 无validation |
| L2 | 本轮未实现选择adapter | 未启动 | JADE/JFDE等需后续实现/授权 |

base packing保留35456 train agent-window exposures和621 inner exposures；source连续按64装包，不混source地图、不drop_last。此规则不是旧scene minibatch计数，不由Stage-A吞吐推算时间。

goal Adam1e−3/gamma.99；joint Adam1e−3/gamma.995；保留原optimizer/scheduler。运行模板未实测吞吐，旧96h/204h权限不继承。所有初态与真实checkpoint仍NOT_CREATED/null。

Goal U-Net encoder14→32→32→64→64→64、decoder64→64→64→32→32，原double conv、12帧logits；history单层LSTM8/256；diffusion两层Transformer d512/head4/FF1024和原ConcatSquash/tree。Stage-A Social128单GRU/一层message/M4/rank8/K21/P20；不启用scene z、geometry6、V4或Stage-B。网络/loss AST 对照见SOURCE_SCOPE。

P2 atomic checkpoint含manifest/data/source/recording/map证据、随机seed/初态hash/goal父SHA、train曝光hash、选择metric/epoch/tie、原epoch optimizer/scheduler payload、实际保存文件SHA sidecar。qualification默认PROVENANCE_REVIEW_REQUIRED。真实CUDA/globalRNG/worker/partialcursor续跑未认证；拒绝作为训练resume。完整源链认证后才能人工提供QUALIFIED父权重，不改旧文件metadata升格。

只读命令（已执行）：

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B main.py --p2_config docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu/CLEAN_GOAL_P2.yaml --p2_launch_check
```

同接口替换为 CLEAN_JOINT_P2.yaml / A0_BLOCKED.yaml / A1_BLOCKED.yaml，全部BLOCKED。不要去掉launch-check自动开训。

下一次先提交正面recording/session/namespace/转换release证据和semantic/map许可；按每window观测边界提供认证独立产物。随后另行授权goal→joint有界timing/训练。完整base合格后再给新cache/schema2产物，最后L1；不是直接A1。未来资格manifest应是新文件/新hash，不修改本轮UNKNOWN归档。
