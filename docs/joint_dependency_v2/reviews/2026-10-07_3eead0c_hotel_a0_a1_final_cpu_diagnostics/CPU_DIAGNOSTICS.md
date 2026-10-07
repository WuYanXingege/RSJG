# CPU diagnostics

## 预算与输入封存

本轮限制为 4 threads、8 GiB RSS、2 h wall clock；实际 analyzer wall time 约 3 s。
只对四个有 `COMPLETED` 原子收据的臂加载 initial/best/last checkpoint；parent 加载一次，
总计 13 次安全 CPU load。使用 `weights_only=True` 和 `map_location='cpu'`，没有导入或
实例化模型。正在运行的 A0_3103 checkpoint 未加载。

原计划的分层 window 规则是：按稳定 source/time/window ID 排序，在 train 与 E_joint
各最多 64 个，覆盖 E=0/mixed/all-active 与 degree 层；local condition 最多 32 个，
decision margin 最多 8 window/16 次已有 CPU solver 调用。由于没有符合 schema 的
原生 tensor payload，选择集合为空，没有根据结果挑样本。

## Relation

源码语义如下：

- `relation_entropy`：obs-only base prior `p(r|X)` 的逐 edge entropy 后取均值；
- `teacher_relation_entropy`：train-only future teacher `q(r|X,Y*)` 的逐 edge entropy；
- `dynamic_relation_entropy`：candidate-conditioned deployable
  `p(r|X,g_i,g_j)` 在 edge/candidate 上的平均条件 entropy；
- `predicted_relation_usage_i`：同一 dynamic probability 的聚合 soft mass。

trainer 对普通字段按 window-equal 累加；只有 `_e_gt0/_e_eq0` 后缀字段按相应 scene
count 加权。因此当前日志不能给同一 edge/scene 权重下的精确 `H_cond`、`H_agg` 和
二者差。可确认的是：四个完成臂的 dynamic 条件熵从 1.01–1.13 nat 降至
0.010–0.019 nat；重归一化 aggregate mass 的最大分量达 0.9965–0.9982。mode winner
在 seed3101 与 3102 间发生标签置换，不能赋予跨 seed 固定物理语义。

**判定：**高度集中有日志趋势支持；“全局 collapse”仍缺严格同权重 edge-level
`H_agg`，标为条件性支持而非已证明。

## Pair-cost decomposition

没有 `[E,K,K]` deployable prior cost、合法矩形 mask 或稳定 window ID payload。
因此 centered C、a+b、I 的 RMS/Frobenius norm、重构、source/degree 分布和与
interaction-off 的关联均 `NOT_RUN_MISSING_NATIVE_PAYLOAD`。energy std 从约 0.10
增到 1.24–1.41 只证明 cost 非常量，不能区分 additive 与 interaction。

人工参考检查覆盖 full rectangular uniform support：additive cost 的 I=0、含交互
小例子的双中心行列均值为零、重构等式。4/4 unittest 通过。

## A1 expected-conditional gap

同一 checkpoint 的固定 q/u/cost/graph/logits 和历史 draw IDs 均未保存；不能用 A0、A1
两个不同模型的训练 loss 相减充当 Jensen gap。因此 q entropy、中心化 logit 方差、
sample gap、analytic-Ez gap 及 S=4 噪声覆盖均未执行。

人工参考检查确认：只产生所有 candidate 共同 shift 的邻居随机性给出 sample gap=0；
candidate-dependent 变化给出严格正 gap。检查不外推为真实 GPU 路径。

## 参数变化与有效学习路径

每个完成 selected checkpoint 相对其共享 initial state：

- frozen `goal_module`、`diffnet`、`registrar`、`var_sched` 和全部
  `jdv2_corrector` tensors 的 L2 delta 为 0；
- social encoder、base relation、future teacher、unary、dynamic relation 和 joint
  energy 均有 tensor 变化；
- 注册 Stage-A 参数为 331,845，其中 dynamic relation 的 64-parameter
  `relation_embedding` 在 strict-no-z Stage-A 路径没有变化，故有效 active 参数
  331,781；这与历史静态接线审计一致。

参数 delta 只能证明状态变化，不是梯度大小估计。没有 per-step gradient payload，
不能反推出 unary、relation 或 energy 的精确梯度主导关系。

## Decision margin 与 step0

没有 exact decision payload，CPU solver calls 为 0，assignment margin 标为
`NOT_RUN_MISSING_PAYLOAD`。预检初态的 JADE/JFDE 仅是不同证据层级的负面线索；
缺少同协议 paired step0 bank，状态保持 `MISSING_PAIRED_STEP0_EVAL`，没有追加推理。
