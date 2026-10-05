# Expected conditional composite：CPU 输入与梯度合同

本轮新增独立库函数，未接入生产训练/推理。接口位于 src/joint_goal_loss.py:932；新增 Parameter、buffer、module state 均为0。

~~~python
expected_conditional_composite(
    unary_score, soft_goal_target, scene_index, edge_index,
    edge_cost_chunks, neighbor_ids, candidate_mask=None,
) -> scalar_tensor
~~~

## 输入

| 输入 | shape / dtype | 严格规则 |
|---|---|---|
| unary_score=u | [N,K] FP32或FP64 | N,K>0；全部有限，包括masked位置；可求梯度 |
| soft_goal_target=q | [N,K]，与u同dtype | endpoint固定标签，requires_grad=True拒绝；非负有限；行和正且与1绝对差≤1e−6(FP32)/1e−12(FP64)，rtol0；不重归一化、不detach |
| scene_index | [N] int64 | 标签允许不连续及负数；仅实际出现的scene计入B |
| edge_index | [2,E] int64 | 0≤src<dst<N；无重复无向边、自环或跨scene边；原边顺序不重排 |
| neighbor_ids=z | [S,N] int64 | S>0，外部给定；每个ID在所属agent的valid support，负索引拒绝；不要求来自模型预测 |
| candidate_mask | [N,K] bool 或None | None=全真；每人至少一个valid；q在invalid位置严格为0 |
| edge_cost_chunks | 一次性 iterable | 每项(start,stop,C)，Python int bounds，C[stop-start,K,K]与u同dtype；全部有限；完整顺序覆盖[0,E)，不重叠、不漏块、不允许空块 |

全部tensor限定dense strided **CPU**，允许non-contiguous。混合dtype、FP16/BF16、非CPU拒绝，不隐式搬设备/转精度。FP64运算与归约保留FP64；不复用内部强转FP32的旧helper。该CPU-only边界是本轮有意限定，后续CUDA集成须另行扩展并认证。

E=0合法，此时chunks必须为空（不是一个[0,0)空块）。不要求K≥20：这里是纯loss，不是P20 sampler。C的第一候选轴属于src、第二轴属于dst，重编号后若边方向翻转必须同步转置C。

## 数学与实现

对每个s、每条边(i,j)，向i累加C(k,z_j)，向j累加C(z_i,k)。每个draw维护[N,K] local；分块采用out-of-place index_add，先累计全部边，之后才计算log_softmax_K(u−local)。不做degree归一化，也不逐块分别计算CE。

\[
L=\frac1S\sum_s\frac1B\sum_{b\in\mathrm{occupied}}\frac1{N_b}
\sum_{i\in b}-\sum_{k\in V_i}q_i(k)
\log\operatorname{softmax}_{V_i}(u_i-A_i^{(s)})_k .
\]

scene权重由unique(scene_index)与实际counts得到，不用max(ID)+1或全agent均值。masked logits在normalizer前设−∞；log_softmax后先把masked log-prob改为0，再乘q，避免0×−∞。不接受输入masked位置本身的NaN/Inf。有限输入仍可能溢出，若累加、logits、log-prob或标量loss非有限则明确抛FloatingPointError，不clip或吞NaN。

q不参与梯度；**此规则不适用于通过C传入的relation teacher logits**。u与C不detach，cost的上游图保留。没有RNG、generator、sampler、model或solver调用，没有任何输入内容/版本修改。

## 内存与复杂度

对每个chunk逐S gather：[Ec,K] source及destination；没有S份[Ec,K,K] cost。live local列表总[S,N,K]；stack后的accumulated、logits、log_prob、安全mask副本等各为[S,N,K]，是多个同阶tensor，不声称只占一份。autograd还保留gather索引/边累加图，峰值并不严格只随当前Ec变化。

静态例S4/N128/K21、FP32：每个[S,N,K]为43,008 bytes；Ec256时一个[Ec,K] gather为21,504 bytes，两个方向43,008 bytes。上游单个cost[Ec,K,K]为451,584 bytes。FP64均加倍；以上是shape乘元素字节数，不是实测峰值。验证/遍历cost O(EK²)，条件gather/累加 O(SEK)，归一化 O(SNK)；未做运行性能benchmark。

## 后续不自动获准

未来集成须明示CPU→CUDA/AMP合同、成功optimizer step与独立训练RNG/resume、Cpost/Cprior同z、原KL/teacher梯度以及默认旧分支兼容性。本轮没有模型strict-load、训练稳定性、完整canonical rollout或收益认证。
