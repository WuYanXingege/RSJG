# 独立 readiness：不被首个异常掩盖

所有阶段仍未获训练/GPU授权。LAUNCH_CHECKS.json 是在 CPU 护栏内实际调用现有 src.parser.main_parser 的 --p2_launch_check 分支，不是自行编造 CLI，也不构造 main.py 的 trainer/模型。goal、joint、cache、L1 A0、L1 A1 均先阻断于 BLOCKED_RECORDING_IDENTITY_UNVERIFIED，设备/权重/provider之前退出。下表是另外按全部 metadata 字段检查，不能把未触达条件说成 launch-check 已通过。

| 后续阶段 | 观察输入准备 | 独立缺口 |
|---|---|---|
| fresh goal | train3484+inner320，见 catalog 完整回读 | recording4249/map4249未合格；train3484 target、inner320 metric-only target；独立预算与启动授权 |
| fresh joint | 同上 | 上述来源/targets；合格 fresh goal=0；history/diffusion需fresh初始化；独立授权 |
| 新 cache | 全角色4249 observation准备 | 合格完整base=0；train-only target/teacher build授权；新deployment4249、train teacher3484未创建 |
| L1 A0/A1 | observation准备不等于两臂就绪 | 新cache/teacher、同一shared initial state、pair预算/失败账本均未就绪；现行500attempt、3484reference及两臂合同不改 |
| L2 | 无执行 | JADE/JFDE/min_delta/early-stop adapter未实现；需另行预算/授权 |

现行候选 runtime manifest 仅增加新 observation 引用和 preparation 状态，source rows、recording/map资格、role、seed3101顺序、parents/cache/shared_initial_state均与原版逐项对比。不更改UNKNOWN，不加入allow_unknown开关。source/member/converter、recording independence、map producer/weights/label-role/protocol仍有独立证据缺口。

真实 target/future/teacher 本轮均未查询/物化；outer target不是本轮或下一步默认需求。本轮全4249 observation允许准备，并不放行outer metrics。真实UNKNOWN metadata下outer selection在provenance gate先拒绝，provider计数0；生产角色策略本身未修改。

旧五份违规checkpoint未加载/重定性，qualified goal/base各0。成功生成obs不能认证clean base、GPU exact resume、MC目标收益、L1/L2效果或旧性能差距成因。
