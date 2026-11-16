# BCEP测试结果

8项CPU测试通过；30个随机有限支持穷举在选择器测试中执行。包括：非平凡spline前反向与数值Jacobian（FP64，1e-9）；世界坐标Jacobian与静止frame；贪心逐步决策、收益界、子模性质与退化/无效权重；局部RNG、未来标签哨兵、padding、agent顺序与调用顺序；3次合成optimizer更新和严格恢复；短horizon及无图；H候选gather集合；endpoint指标与IID能量评分定义。

真实CPU接口检查2个ETH训练折窗口（3人和1人，分别来自internal train/dev），M256→K20，使用实际6通道语义crop，坐标米制。随机初始化，非正式候选bank已明确标记INTERFACE_SMOKE_NOT_FORMAL，不用于质量结论；无真实训练更新。固定seed输出一致，全局Torch RNG不变，输出及NLL有限。

训练入口默认拒绝运行已实测；仅自有随机初始化synthetic checkpoint被测试加载。完整真实训练/恢复、真实精度、校准、速度、五折和3训练seed均未执行。没有加载GDTS权重或teacher。旧六层回放属于恢复阶段，不能当成BCEP的性能结果。

复现：从独立worktree运行 `PYTHONPATH=. ../.conda/rsjg/bin/python -m pytest tests/test_bcep_core.py -q`。恢复主线入口与完整收据见 docs/restoration/20261116_sixlayer_provider/。
