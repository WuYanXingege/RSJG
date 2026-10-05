# 共同初态与 RNG：待授权，不创建模型

状态 TEMPLATE_BLOCKED；实际初始化路径/SHA 均 null；NOT_CREATED_AUTH_REQUIRED。旧 canonical/P1 A 不能作为 clean 共同 warm start。

每个 training seed s∈{3101,3102,3103}：验证同一个合格 base 全组件指纹→一次性新建 Stage-A heads→corrector末层严格零且冻结→保存一个 shared initial state 和逐组件/整体SHA→A0/A1加载同一字节产物。两次各自“设同seed”不替代共享初态。两臂 optimizer 都是新建空 Adam 状态，scheduler计数/progress相同，不恢复旧 optimizer。生成/保存/加载后再次核验 frozen weights，排除构造时覆盖。

## Namespace 与实际随机对象

| 用途 | 规则/产物 | 隔离要求 |
|---|---|---|
| 初始化 | namespace rsjg.p2.init.v1，seed=s；保存实际共享 state + RNG snapshot | 只建一次；主seed的随机调用次序不是证据 |
| ordered windows | rsjg.p2.batch-order.v1；以 compact JSON [namespace,s,source_id,id_sha256] 的SHA排序 | seed3101的3484条顺序已存 BATCH_ORDER_SEED3101.json，两臂前500条完全一致；不消费global RNG |
| MC target | SHA256("rsjg.p2.mc-neighbor.v1:"+str(s))前8byte big-endian mod2^63 | s3101=8257597550193927250；A1独立CPU Generator，post/prior共用一次draw；A0不建owner/不draw |
| train global/dropout | namespace rsjg.p2.train-global.v1，seed=s；在load共同初态之后恢复同一Python/NumPy/Torch CPU/CUDA初始状态 | MC不改变global；记录任何目标导致的额外随机调用，不能凭种子宣称逐步相同 |
| inner selection | namespace rsjg.p2.validation.v1；预登记validation_seed2035 | 验证保存/恢复train RNG，不消耗MC owner；两臂同一真实噪声 |
| deployment | rsjg.p2.deployment.v1；inference seeds2035–2039，按source/window/seed寻址 | 固定goal/Gumbel/diffusion实际noise payload/hash，非仅seed。需后续明确adapter；本轮未创建 |

两个loss目标不能以原始loss绝对值判断优劣。冻结base的train/eval mode和dropout政策也须相同。若真实噪声不能做到共享，明确不具备paired noise证据，不偷换为same-seed认证。

## Resume与step边界

沿用 format1 CPU / format2 CUDA 的objective/backend/AMP身份，不同身份不能同阶段resume。CPU synthetic认证不外推真实CUDA。
新run不从canonical optimizer续跑。A1记录generator state、draw count、pending状态；一次attempt内先draw/计算，再按现有成功optimizer更新事务提交，异常立即停止pair，不补抽窗口。
保存实际completed updates、ordered batch cursor、Python/NumPy/CPU/CUDA RNG、optimizer/scheduler/scaler和state hashes；原子checkpoint保存必须在后续实际入口验收。本轮未认证真实CUDA断点续跑。
