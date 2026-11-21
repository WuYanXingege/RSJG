# 候选身份规范化、逆映射与训练图

## 1. 什么是身份

原 `Bank` 的行收据包含 `index/member/identity_hash/sha256`，数组格式是FP32 `[20,N,2]`，排序声明为original KMeans center order。没有每候选的独立语义类别或概率。

可信的出现实例键是 `(bank_hash, row_sha256, member.agent_id, original_slot)`，其中slot是**经校验文件中的原位置**。它不表示第k近候选，也不表示某种行为。候选重排R必须同步执行 `G'=gather(G,R), id'=gather(id,R)`。把新数组再次标为0..19会丢失原身份。

坐标是物理端点值，出现实例ID是与bank/keyed-noise绑定的记录身份，数组索引仅是当前存储位置。三者不能混用。

## 2. 精确规范顺序

对每个有效多人场景行人独立按 `(FP32 x, FP32 y, original_slot if available)` 稳定字典序排序，得到

- `L[b,i,c]`：规范候选c在调用方数组中的下标；
- `L_inverse[b,i,a]`：调用方候选a在规范数组中的下标。

排序只作用于脱离梯度的固定端点。未对端点做舍入、软平均、移动或误差容限合并。若坐标互异，前两个键已构成全序；可选slot不改变排序。如果完全重合，包括数学上相等的+0/−0，则必须提供随重排传递的出现实例ID，否则抛出异常。FP32相邻1 ULP仍为不同键；NaN/Inf有效候选拒绝。padding不参与配对，保留原索引约定。

对纯重编号R，若`G'[a]=G[R[a]]`且ID同步，则

`R[L'(c)] = L(c)`，因此`G'[L'(c)] = G[L(c)]`。

同一设备、dtype、模型状态和运行设置下，进入原编码、attention、Hungarian与公共能量的张量逐值相同。保证来自相同输入计算次序，不来自“实数算术下attention应当等变”的理论替代。

## 3. 输出、评分、软代理

原内部计算得到规范索引 `pi_can[b,w,i]`。输出索引为

`pi_input[b,w,i] = L[b,i,pi_can[b,w,i]]`。

评分列映射为 `S_input[b,i,w,a] = S_can[b,i,w,L_inverse[b,i,a]]`。外部incumbent若提供，先用`L_inverse`映射到规范索引。新诊断details同时包含原输入索引与规范索引，能量trace中的候选编号是规范编号，不能直接解释为原bank槽位。

硬终点为规范候选的精确gather，等于按`pi_input`从原输入gather。proposal-ST和final_coordinate不变；其坐标代理在规范候选顺序内计算，再映射诊断列。排序/固定终点不建立坐标梯度，φ/ρ及原评分参数梯度由原计算图产生。内核的接受/拒绝逻辑、唯一性与FP64误差界均未修改。

## 4. 单人、padding和增广

单人场景遵守原native identity排列：重编号就是整个world列表重编号。比较单人输出时按同一候选身份映射world，不能要求未对齐的第w条轨迹不变。正式G/H适配器先恢复原bank槽位，保持原单人G=H检查。

padding仍使用identity，物理有效性由valid控制。没有把padding坐标作为真实候选身份。

观测与候选通过原`augment`同步旋转/反射；ID不变。不同增广可能改变xy规范顺序，但比较同一次增广前后候选重编号时规范输入仍相同。本轮不宣称旋转不变性。若FP32增广使不同点舍入重合，携带的原实例ID仍可区分；没有ID时停止，而不是猜测。

## 5. 原噪声与G/H

原噪声key包含原候选槽位k，因此独立重排后不能仍用新数组位置k重新生成噪声。`bcep_scene_energy.inference.generate`要求出现实例ID，先恢复原bank槽位，调用**未经修改的原生generate**。原G组合、噪声函数和六层20步DDIM全部保留。最后将H排列映射回调用方输入索引。

适配器没有重新选一次incumbent，没有将结果放回DDIM中重复配对。原generate仍只调用配对一次。无模型副本；`IdentityBoundModel`仅绑定本次输入身份，原参数、bridge、objective共用同一对象。

最小调用示例（只读前向）：

```python
from bcep_scene_energy.identity import bank_occurrence_ids
from bcep_scene_energy.inference import generate, IdentityBoundModel

goals, ids, namespace = bank_occurrence_ids(checked_bank, indices, batch)
# 原增广改变batch/goals，ids保持不变；重编号时goals与ids同步gather。
x = batch.to(device)
g, h, pi, receipt = generate(
    model, x.observed, x.semantic_maps, x.valid, x.scene_ids, x.agent_ids,
    goals.to(device), 2035, candidate_ids=ids.to(device))
```

受控合成梯度检查可将 `IdentityBoundModel(model, ids)` 传入原 `bcep_v2.objective.bridge_and_pair_loss`；这不授权训练，也不构造新损失。当前冻结bank坐标互异，旧原生回调的无ID默认路径与带ID路径规范顺序相同，已做分组反向回归。若未来更换bank或变换导致重合，须传递身份，不可关闭异常来继续训练。

## 6. 未解决的边界

固定候选规范顺序不固定行人token拼接顺序，也不消除坐标更新的物理访问顺序依赖。严格并列匹配仍可能有多个数学最优解；在相同规范输入下原Hungarian重复返回一致选择，不代表任意标签操作和硬件下都唯一或等变。新协调提案仍使用原严格唯一性及可靠能量下降检查，严格并列保留incumbent。

没有改变 Sinkhorn、proposal-ST 或final_coordinate的含义。非零且重编号一致的梯度不能证明离散分配的真实优化方向已经正确。
