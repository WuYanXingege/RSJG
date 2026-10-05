# 资源与依赖：阶段分别授权

全部为未执行计划，旧204 GPU-hour矩阵不是本轮授权。首先补clean base，而非先跑A1。

| 阶段 | 必须先有 | 最小产物/拟定停止上限 | 耗时证据 |
|---|---|---|---|
| CPU适配准备 | recording/ID、semantic来源证据 | P2 logical-role loader、train-only teacher、L1 cap、无validation、L2选择合同的最小patch及CPU测试 | 本轮仅审查设计，未改生产 |
| clean goal预训练 | 上述P2 loader，train-only统计 | 随机初始化U-Net；仅train3484来源，inner320 masked goal BCE选择；≤150epochs，和joint合计≤96h | 新P2 packing/吞吐UNKNOWN |
| clean joint base | clean goal最佳SHA | fresh history LSTM/diffusion，independent GDTS；train-only梯度，inner预登记ADE选择；≤250epochs，合计≤96h | 新更新数/吞吐UNKNOWN |
| 新cache | 完整base链合格且锁定SHA | 4249 observation-only deployment / 3484train teacher / 0protected teacher；拟首次timing≤15min，完整cache硬cap2h | UNKNOWN，不用smoke估 |
| L1 | base/cache/共同初态/cap/loader | seed3101两臂各≤500attempt，900s/arm、2700s全阶段、10GiBreserved | 只给cap，不预测 |
| L2训练 | L1通路验收后新授权 | 3seed×2arm，≤40epoch且≤4h/arm，合计≤24h | UNKNOWN |
| L2选择与正式部署 | inference noise合同、锁定模型分析 | 每epochinner、最终outer一次5seed；单独部署/验证拟定合计≤6h（不挤占L1） | UNKNOWN，首次inner timing≤15min计入cap |
| pair归因 | L2结果与独立因果授权 | full/zero/additive+gauge与固定bank；另拟≤4h含bank/solver | UNKNOWN，超限停止 |

cache2h+部署6h+归因4h合计≤旧12h子上限，base≤96h、主线L2≤24h；这是缩小后的条件性cap设计，不是重新继承整矩阵、不含B/AB/V4/nearest-method。所有计时包含阶段失败尝试；只允许空闲设备、不干预SDD。正式预算还需在授权前由实际timing和packing元数据细化，只能显式修订，不默默扩大。

## 最小clean base训练/选择设计

1. 保持原goal/U-Net、history LSTM、diffusion架构；拒绝已有违规base或旧A warm start。外部semantic可学习来源需先合格，否则当前pipeline仍不能声明全链clean。
2. P2 source membership锁定；obs/future独立存储或权限视图。train可读GT，inner仅预登记selection，outer不开标签。所有normalization/statistics只拟合train；固定单位/坐标/map变换。
3. goal阶段使用实际phase=goal_pretrain接口与masked goal_BCE_loss；Adam lr1e−3、ExponentialLR gamma=.99（源码已有），≤150epoch。best按inner BCE，exact tie早epoch；保留随机初始hash、optimizer/RNG、每epoch成员/selection账本。
4. joint阶段使用实际phase=train/goal_model_type=independent；通过goal_pretrain_checkpoint导入该clean goal；history/diffusion fresh，非pretrain_path违规全模型。Adam lr1e−3、ExponentialLR gamma=.995；≤250epoch，inner ADE作为预先选择指标、exact tie早epoch；锁定坐标单位/metric定义。
5. native base batch_size64/packing与Stage-A scene batch1不同：必须先生成实际packed membership、optimizer更新数/epoch、总曝光，当前UNKNOWN。不能用3484作为base更新数，不能用Stage-A吞吐预测base。
6. 新文件保存完整train/inner source SHA、祖先链、统计拟合证据、semantic外部来源、选择规则、real RNG/optimizer/scheduler与checkpoint原子保存证明；inner不能梯度更新。先CPU合同验收，再另行授权GPU训练。
7. base未在硬cap内达成可用合同则停止，不用半成品冒充clean。只完成goal还不够；history/diffusion会影响预测也必须clean。

当前clean协议硬编码P1，且main预处理包含所有physical splits；上述base plan **不是可直接运行的现有CLI**。必须先实现logical-role source/selection路径和label lock，不能靠existing clean_split_protocol开关解锁。

## 估时证据边界

旧P1 A0真实训练log：3790 batch完整训练loop02:52（172s，tqdm21.94it/s）；epoch诊断381.22035s包含validation。它只描述旧run，既不是P2 A1、base训练或完整diffusion吞吐，也不能当预算保证。上轮3.334s小N smoke不用于估时。没有同类型日志的项统一UNKNOWN；首次timing仍需授权且计入阶段cap，不能临时突破“不做outer评测”边界。
