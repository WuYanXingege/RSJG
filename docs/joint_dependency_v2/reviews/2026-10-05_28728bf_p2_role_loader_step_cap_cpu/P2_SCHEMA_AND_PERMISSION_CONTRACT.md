# P2 schema 与用途合同

family=`p2_hotel_uni_examples`；schema=`rsjg-p2-role-v1`；cache=`jdv2-p2-cache-v2`；默认 off。审计 schema 与新 runtime manifest 分离。当前 runtime 包含 UNKNOWN/null，因此可供 metadata 验证但不可执行训练。

注册 source rows SHA256：`a3df43f81bd1c198d369b1607bf3e02ed7104c9197a094e9d93fcf6a0254e4c1`。
实际 runtime 与 shard/order hashes 见 P2_RUNTIME_MANIFEST/ROLE_ADDRESS_DRY_RUN。新 window ID 表示法导致 order hash 字面值不同，原成员次序逐项相同；不按整数 index 去重。

| logical role / purpose | obs | target | teacher |
|---|---|---|---|
| train / gradient | 独立obs8 | 允许 | 禁止 |
| train / JDV2 gradient | 独立obs8 | 允许 | 允许 |
| train / teacher_build_train | 允许 | 允许 | 此处计算后写独立sidecar |
| inner / selection | 独立obs8送预测 | 仅之后的metric writer | 禁止 |
| any / candidate_build | 独立obs8 | 禁止 | 禁止 |
| outer / metrics | 读取前拒绝 | 读取前拒绝 | 永远禁止 |

完整 metadata 授权 → typed provider 的 path/hash/window/source_identity/kind 验证 → payload → 用途适配。FileProviders 没有 full pickle fallback；hash等于原批文件、path等于原批文件均拒绝。独立产物须由可信/获授权分离流程认证；不能仅靠手填 kind 标签证明内容来源。

记录包括 source SHA、alias、physical source/split/index/cache_filename/file SHA、frame/agent身份、recording/namespace证据、8/12边界、map来源以及artifact引用。同地址/内容身份或跨角色已知 recording 拒绝。当前对同 recording 跨角色采用保守拒绝，未引入时间片豁免。

map PROVIDED_STATIC_ASSET 必须有证据hash和协议许可；LEARNED_PREPROCESSOR 另需train-only训练来源；UNKNOWN拒绝。不以文件名不同证明独立。

训练full20只在授权后组合；obs-only input adapter不填零future、不把target送 prepare_inputs。标签时间窗只按当前window，不用全局frame并集。augmentation当前显式关闭。新P2 base按source连续agent64装包；Stage-A按保存seed3101窗顺序、scene1。

入口依赖：goal需source/许可/obs+train/inner target，无base依赖；joint增加clean goal；cache增加完整base和obs4249/train target3484；L1增加deployment/train teacher/shared init/cap。当前outer metric无放行流程，不能借用P1 lock。

legacy/P1仍旧schema消费；P2消费者仅新schema。新cache producer需实际文件SHA且合格链，旧5违规SHA显式拒绝。新checkpoint写出的provenance sidecar待独立审核，不能自动成为合格父权重。
