# BCEP v1：已实现的无GDTS权重原型

主工作区保持已回放的六层基线。此独立worktree只做原型与CPU测试，allow_real_training=false；无真实optimizer update。

观测 [N,Tobs,2] 与可选 [N,6,H,W] 语义crop → 128维两层时序编码、一次masked social attention、32/64/128浅层CNN → horizon条件4层交替二维RQS coupling（8 bins，线性尾部） → 世界坐标256个IID等权支持原子 → 欧氏最近距离风险贪心选择20个代表点。没有未来输入、GDTS teacher、GDTS权重或终点扩散链。

运动frame仅用最后3个观测位移求平均速度；速度<=1e-3 m/s用单位旋转。内部train逐horizon残差RMS拟合尺度，下界0.1m。world log_prob包括旋转/尺度Jacobian。多horizon是ceil(T/3)、ceil(2T/3)、T，辅助权重0.25，短预测期去重。未加入best-of-K训练损失、排斥、终点残差或bridge损失。

接口 BCEP.predict(observed, semantic_maps, valid, scene_id, agent_ids, seed) 返回[K,N,2]与metadata。只在单个同步scene内编码，多scene调用顺序不影响identity派生的本地随机流；candidate顺序为稳定greedy顺序。支持质量等权，不再乘密度。代表点不是IID分布样本。

地图来自既有协议语义资产，仅作特征条件，不限制终点范围；无图用显式可学习missing-map向量。语义图可能有分割模型训练来源，不能宣称完全无外部先验。训练runner仅用SceneManifestDataset；路径中存在GDTS资产目录不表示加载GDTS学习权重。

训练入口具备完整epoch模型/Adam/scheduler、epoch/update、Python/NumPy/Torch/loader RNG、best/stale与身份保存恢复；实际训练默认拒绝。原型checkpoint完整恢复尚未做真实数据训练续跑，只有合成optimizer保存/恢复测试。未来独立完整EBJD必须重新随机初始化coupling、六层桥与地图U-Net，不能用旧配置自动载入matched初始化。

已实现IID支持，未实现Sobol；未启用真实训练。原型没有性能与精度优势证据，不能称为已接受的主模型。
