# 联合与边缘轨迹修正：启动归档

当前为实现与启动记录，尚无本次新头训练结果，也没有已确认的GPU提速结论。

固定起点4969c12adb5675aa654b2e748dec8da0f6e32adf。先从ZARA1冻结B epoch6开始，train推理流2036、dev2035，头训练seed3201；宽64共享主干及common/centered-person读出，新增27,332参数。逻辑批32、AdamW lr3e-4、weight_decay1e-4、clip1、FP32、固定30轮，无学习率或种子搜索。

全体minFDE差≤−0.020m、JFDE差≤−0.005m、minADE/JADE差均严格<0，密集组JFDE/minFDE差均≤0；全部必须来自同一轮。通过轮选minFDE最低，再按JFDE和早轮决胜。未通过则选最大条件超限最小者，再按minFDE和早轮决胜，结束模型扩折；提速交付独立完成。

ZARA1通过才运行同初始化等参数P=J,a=0对照，以及各自训练集上的ZARA2/HOTEL；三折均通过再验证既有控制器释放的UNIV/ETH权重。测试集访问0。此前看过三个测试折旧模型成绩的事实公开说明，不用于本次特征、训练、提前停止或选模。

用户后续明确允许并行，因此新增作业与原主线共享GPU，最多一个辅助GPU进程，allocator上限40%，低于6GiB空闲则等待；原主线代码、配置、锁和权重不修改。微基准记录共享资源竞争。先按同机同数据固定容差验收，再以至少3次预热后配对中位数决定启用优化；不合格优化按名称关闭，不扩大容差。

运行：`CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m joint_marginal_residual.run --push-on-complete`。该入口支持同一任务断点恢复、条件扩折、完整观测开发集复核和自动归档推送。每阶段保留全部完整epoch、官方排序Top5、best、last以及完整RNG、optimizer、顺序和游标。

原始观测推理：`CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m joint_marginal_residual.inference --fold zara1 --checkpoint <选中checkpoint> --purpose selection_dev --position 0 --seed 2035 --output <host-output.pt>`。教师只解码一次；头只读取历史、冻结B几何和覆盖摘要。

已完成33项CPU接口/数学/恢复测试，以及真实ZARA1的density、旧head、coordinate、B依赖四阶段CPU接口微基准。正式GPU微基准与完整新头训练结果完成后，本页自动替换为实测中文报告。原始数据、缓存和权重留在主机。
