# UNIV 统一配对条件桥启动合同

- 训练 seed 3101，30 epochs；hidden 192、4 heads、FFN 768、3+3 FJAB。
- FULL、CTF、GT 一致条件桥、固定终点、`proposal_eta=0`、Raw。
- pair weight 1.0；Sinkhorn temperature 1.0、20 iterations；pair-loss temperature 0.1。
- 主干/配对学习率 `1e-4`，可训练地图 U-Net `1e-5`；1–5 轮 warmup，6–30 轮 cosine decay 至 0.1 倍。
- rollout 与 paired actual-step projection 全程关闭；不早停、不扫描参数。
- 每轮使用 seed 2035、20 worlds、20 DDIM steps 验证 G/H；最终冻结同一 checkpoint 后评测 2035–2039。

## 绑定身份

| 资产 | SHA256 |
|---|---|
| Goal-pretrain U-Net | `9fe5b2a7ae1e765d2706763b342d461c11ac0e736a2b597b6e18d63381652229` |
| matched GDTS | `ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e` |
| train manifest | `6f6fca362ec1e591bf9a79c9b1a0204123947a0c7b754992f131432af40dfd5d` |
| validation manifest | `46c52540aac6f0313865f01f53d7c735d91cb8116fdbee39cb65f0cf545c7d8e` |
| test manifest | `aabb35de82e936495fcd948a7ba0416d92d4fc93366151b99573a4d7683ee7e0` |
| matched GDTS five-seed result | `04f8d43f92b94ec09037bb267949f66092fa2dd2f42fe2b3e4388b33602c3477` |

训练折为 4,067 个同步窗口、15,186 个有效行人出现，平均 3.734、最大 16 人；验证折为 947 个窗口、24,334 个有效行人出现，平均 25.696、最大 57 人。所有真实行人保留，表示尺度只从训练折拟合。

旧 UNIV epoch73 与 Control epoch81 不加载，只保留历史身份。冻结候选提供器来自 matched GDTS 的 Goal U-Net；主模型内可训练地图 U-Net只加载同折 Goal-pretrain 权重，两者是独立参数副本。

GPU 硬预算为 24 小时，至少预留 3 小时用于最终评测。精确运行身份、配置哈希、累计成本和恢复状态以持久实验目录中的收据为准。
