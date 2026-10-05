# Grouped UNIV 新协议与工程边界

本轮起点 `7a0f83bb9341617b4f214cefc3563927961c08d6`。新 family
`p2_grouped_univ_hotel_v1` / `rsjg-p2-grouped-role-v1`，旧 P2 完整保留。
本轮未授权正式长训练、HOTEL target/metric、inner/outer 梯度或 teacher。

固定 original rows 和 947 成员摘要，先还原原逻辑角色执行旧结构验证，再对新角色、保守组、
rows/order/packing 单独验证。原 recording/session/ID-remap 字段仍 UNKNOWN。
UNIV 三源合为 inner 组；ZARA 三源合为 train 组；ETH=train、HOTEL=outer。
1267 个验证窗口重叠，不是 1267 个独立 recording。

新资格只认证“作者分发处理文件的精确 bytes 绑定 + 跨角色保守场景组隔离”，不认证完整原始
采集/转换历史。geometry 数值可逆不等于 camera 归属证明；后者需作者 loader 映射和资产匹配。
静态 raster 包成员匹配不证明 learned preprocessor 的训练标签范围。若该范围不满足新协议，
生产仍拒绝，禁止 generic allow_unknown、虚假 VERIFIED 或替换输入通道。

## 逐窗用途

| 角色 | observation | target | 梯度/teacher |
|---|---|---|---|
| train2537 | obs8 | t8..19 gradient preparation；之后真实 train loss | 仅来源合格后生产；独立 SMOKE_ONLY 授权例外 |
| inner1267 | obs8 | t8..19 metric preparation；生产必须预测完成后读取 | 永远拒绝 |
| outer445 | obs8 | 永远拒绝 | 永远拒绝 |

准备 grant 绑定 preparation manifest、全成员/角色/source identity；并非生产授权。每个 raw 数值查询
都绑定 `(window,t,frame,agent,purpose token)`。按 source 一次扫描仅是 I/O 优化，routes 仍逐窗；
未授权行只读 ID 字节，不解析 XY；全源 SHA 不计作坐标授权。不存在 full-pickle fallback。

窗口 ID 不随 role 变；947 份 wrapper source digest 随 role 变，必须重封装；其余只引用原文件。
验证旧 producer/asset/grant/file/content hash 和新回读。obs 与 target 独立保存；原 sidecar 不覆盖。
target 使用原 world2pixel、ETH/HOTEL swap、down8、Gaussian 和峰值归一化；固定 cohort 不按未来重选。
缺 slot、重复键、非有限、身份/half-write/hash 不符即停止。输出地图外坐标不 clamp、不丢窗。

## 新训练入口

main → p2 dispatch → grouped runner，仅对新 family 分派。默认旧模型/目标/训练器保持；模型唯一改动
是可选 dataset 注入，默认仍走旧构造器。新 GeometryScene 仅读绑定 H，不隐式读取 GT mask、raw 或旧 cache。
新 PackReader 的 inner 路径在 prediction callback 返回后才访问 target provider；旧 eager-target
loader 对新 base family 拒绝。outer 不进入 goal/joint loader。实际来源未合格时真实生产 loader 不称通过。

Goal 原 BCE 定义不变，跨 pack 按最后 loss mask 有效 agent-window 加权。joint 保留原 `ADE`
定义：原生像素、P20 best-of、原 metric mask；米制 `ADE_world` 不是这里的选择指标。
验证 seed3101 与训练 RNG 保存恢复分离；strict improvement，tie 保留早 epoch。

Goal 上限150×175=26250；joint250×175=43750，均保留尾 pack、无 augmentation、workers0、accumulation1。
原生 base packing 按注册源/窗口顺序，不混源；seed3101 train_order 专供逐窗 Stage-A。
LR goal=.001、joint=.0001，Adam / ExponentialLR gamma=.99、clip1，正式 FP32。
本轮不启动这些训练。初次正式 wall 上限 goal24h/joint48h 是新失败预算，不继承历史预算，不是时长承诺。
partial epoch 不验证、不步进 scheduler、不保存可恢复 checkpoint；本轮仅认证 FRESH_START_ONLY。

正式目录要求全新；实际父权重 hash/schema/state/protocol/data/architecture/initial/epoch/selection 均核验。
joint 只加载 goal_module，history/diffusion 全新；缺合格 goal 必须等待。smoke、旧五份违规 SHA 永不提升为父权重。

## 后续依赖和旧结论

Goal 可启动不要求提前存在 qualified goal/base。joint 等 qualified goal，Stage-A 等 qualified complete base、
新 deployment4249/train-only teacher2537、共同未训练初态。新 cache schema 独立；outer observation 的来源资格
只在真正消费它的 cache 阶段要求。L1 H<=500/reference2537，A0 mean-energy / A1 MC S4 独立 RNG；
不把新旧 L1 当同一实验。L2 selector 实现默认关闭，不生成真实 L2/outer metrics。

Stage-A objective/wiring 未改：复用 cb5160f/dcf5fde/c9dbfec/4f11f44 的有界验收，不重跑昂贵历史片段。
历史违规权重、旧配对 Δ、seed2036 非逐位续跑、历史首差因果 OPEN、上轮 import/native/RSS 覆盖缺口均保留。
CPU 单元 fixture、真实准备文件验收、SMOKE_ONLY 更新、真实生产资格分别报告。

`data_binding` 绑定不可变成员/角色组、实际资产、顺序和资格合同版本，不绑定可补充的证明文件或状态。
证据仍由每个生产阶段的严格 gate 独立验证，checkpoint 另保存当时 manifest/source；这不是 UNKNOWN bypass。
后续只补 outer 证明不误使已有 goal/base 数据身份失效；实际资产/成员/协议改变仍拒绝父产物。
smoke harness 对预选全部 train 成员显式检查 `PREFLIGHT_SMOKE` 用途后才允许 provider/model 访问，不能将一般 preparation 授权自动当成生产训练许可。
