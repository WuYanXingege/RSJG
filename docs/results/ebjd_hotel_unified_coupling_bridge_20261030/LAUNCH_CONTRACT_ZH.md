# HOTEL 统一配对条件桥启动合同

## 固定方法与训练合同

- fold：HOTEL；训练 seed：3101；训练 30 epochs。
- 模型：hidden 192、4 heads、FFN 768、3+3 个 FJAB、FULL、CTF、Raw。
- 固定原生终点；GT 一致条件桥；`proposal_eta=0`。
- 配对：pair weight 1.0、Sinkhorn temperature 1.0、20 iterations、pair-loss temperature 0.1。
- 学习率：主干/配对 `1e-4`，可训练地图 U-Net `1e-5`；epoch 1–5 warmup，epoch 6–30 cosine decay 至 0.1 倍。
- rollout 与 paired actual-step projection 全程关闭。
- 每轮使用 seed 2035、20 worlds、20 DDIM steps 完整验证 G/H；其余 seed 只用于冻结 checkpoint 后的最终评测。
- 不早停、不扫描参数、不加载旧 HOTEL EBJD 主模型或 optimizer。

## 绑定资产

| 资产 | SHA256 |
|---|---|
| HOTEL Goal U-Net epoch 96 | `f6a52cf228d733bd684aef043d843ba7d942e40495d5eec354ce1e4e3507ec13` |
| HOTEL matched GDTS | `5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b` |
| train manifest | `2a5f1909d928e5277d426ba04e3488940e34a5f71fad9da6331e5e673ec1966c` |
| validation manifest | `90fb42545ad043410f7c12b9ca91a1a0b218895b19252650e62f7fbefdb77d6d` |
| test manifest | `84c95cb431caf7214d1b85e8e7dbfeaf48c9fe301b1cda7f3201c3e8cf199c78` |
| matched GDTS five-seed result | `df9aa35cd26296558c864a2bcda792a8411bbc83a530376c45d3429a768eae6c` |

训练集为 4,249 个同步窗口、37,702 个有效行人出现；验证集为 445 个窗口、1,197 个有效行人出现，平均 2.6899 人/窗口。表示尺度只从训练折拟合。

## 流水线边界

流水线依次执行：资产核验、最小终点提供器提取、A/C 有序候选回放、train/validation proposal bank 导出、30 轮统一训练、固定选模和 2035–2039 五推理种子评测。任一前置检查失败会停止后续阶段。

独立提供器使用 matched GDTS 中的 Goal U-Net 权重、原生预处理、TTST 和坐标变换；不会在正式 bank 生成阶段实例化 GDTS 轨迹去噪器。冻结提供器和主模型内可训练地图 U-Net 是两个独立参数副本。

正式运行状态、精确命令、配置哈希和 GPU 成本以持久资产实验目录中的 `RUN_MANIFEST.json`、`GPU_BUDGET.json` 与 `configs/RESOLVED_HOTEL_UNIFIED.json` 为准。
