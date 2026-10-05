# 独立 observation preparation 合同

本轮授权来自 RSJG_P2_Observation_Provenance_CPU_Prompt_bda1ae1.md；固定输入/实际开始 HEAD 均为 bda1ae147d5912e8847642db36b5822584420563。不是训练授权。PREPARATION_GRANT.json 绑定原 manifest、全部有序 source rows 与 window IDs；UNKNOWN recording/map 只能进入这个独立 preparation gate。Registry.require_provenance、expected_role、REGISTERED_ROWS、生产 loader、旧 train_order 均不改。

工具默认关闭，必须显式给出 --enable-observation-preparation --device cpu、CUDA_VISIBLE_DEVICES=''、原 manifest 及其摘要。当前工具是这次固定授权的执行器：grant 与累计 wall 起点固定，不能用于隐式延长或继承下一轮授权。

## 查询权限与成员

全 4249 窗沿用离线 benchmark 注册 cohort，不重做 future 完整性筛选，也不认证在线 agent 选择。source bytes 可完整流式 hash；所有行仅解析 frame/agent ID。仅命中某 window_id 的前 8 帧及有序 agent_ids 才解码 x/y，并路由到该窗口 t<8 的槽位。一次 source scan 可以共用原始行，但不是 source-level frame 并集权限。某行可同时是 A 的 future 与 B 的 observation；本轮没有以 A-future 用途查询该行，不能据此宣称全局所有 future 坐标从未读过。

允许 artifact_preparation/artifact_validation，覆盖 train3484/inner320/outer445。禁止 target/future/teacher、full-source pickle、模型构造/执行、优化器、GPU、outer metrics。生产消费者的权限与 provenance gate 完全保留。

## 认证及续做

wrapper 仅包含 window_id/kind/source_identity/payload；sidecar 绑定 source identity/raw SHA、静态资产摘要、数值 producer_hash、grant、obs_steps=8、内容与文件 SHA、qualification。环境与 CLI/guard 的完整代码指纹在 ENVIRONMENT_AND_ARTIFACT_BINDING.json 通过全 catalog shards 的 SHA 绑定到每一个产物，不伪称 sidecar 内已有这些字段。

写临时二进制文件→flush/fsync→原子替换→sealed sidecar→weights_only CPU 消费者回读。文件 bytes SHA 与 sorted tensor content SHA 分开；不承诺跨环境 torch.save 字节一致。已有文件只验证、不覆盖；只有 payload+sidecar 全部通过才复用。半成品会阻断，不被计作通过；本轮仅一个 writer，没有认证并发 writer。受保护原路径或旧 full-batch 同 SHA 改名均不能作为 observation。CPU guard 仅允许通过路径及 SHA allowlist 的 observation load。

32 GiB 新产物、8 GiB RSS、90 分钟 wall、最多 2 工作进程/4 计算线程。导出器使用 1 进程/1 计算线程；测试或 metadata 检查至多再用 1 进程/1 线程。预算起点 2026-10-05 07:46:57 UTC，包括准备、重试与回读，不重置。检查在操作之间进行，不是 OS 级强制内存隔离。

## 边界

坐标越过 raster 几何边界不等于越过授权时窗。保持原 Gaussian 全栅格计算与 peak normalization，不 clamp、丢窗、padding 或改成员；若原公式 underflow 导致非有限值则阻断。canary 有 1 个 raster 外观察槽位，但公式有限且 peak=1，无额外边界处理。全量精确计数见 RESULTS。原始 recording↔map 几何对应仍 UNKNOWN；本轮仅认证现代码数值合同。

精确定义：字段out_of_raster_obs_slots比较的是下采样Gaussian栅格的采样中心范围 x/8∈[0,w−1]、y/8∈[0,h−1]，不是原图连续像素区域[0,W)×[0,H)的独立越界证明。全量1129个此类window-local槽位，无clamp/丢弃。不能由此计数直接断言原始地图错配。

前述 UNKNOWN 不因观察产物完整而升级。状态只能为 OBSERVATION_ARTIFACT_PREPARED_PROVENANCE_PENDING；真实 target/base/cache/初态仍未创建。
