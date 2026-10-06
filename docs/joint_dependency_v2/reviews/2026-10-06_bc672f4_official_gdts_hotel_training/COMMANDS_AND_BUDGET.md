# 官方 GDTS HOTEL 后台启动合同

```bash
../.conda/rsjg/bin/python -B tools/official_gdts_hotel_launch.py \
  --official-root ../GDTS_official_297d508_HOTEL \
  --smoke outputs/joint_dependency_v2/standard_gdts_hotel_297d508_20261006/OFFICIAL_CUDA_SMOKE.json \
  --output outputs/joint_dependency_v2/standard_gdts_hotel_297d508_20261006/formal_launch
```

启动器重新核对上游 commit、兼容补丁 diff、全部关键源码、8个 HOTEL split 文件、591/19 cache 数量、CUDA smoke receipt、GPU 空闲和不存在旧正式训练产物。通过后用 `setsid + nohup` 启动。

正式命令为官方 HOTEL `train_test`：seed2025、FP32、batch64、Adam lr0.001、250 epochs、每10轮验证/保存、20 samples、数据增强开启。WandB 关闭，仅省略外部日志；模型、数据、优化与选模语义不变。

正式产物写入：

```text
../GDTS_official_297d508_HOTEL/output/hotel/
```

独立启动日志写入：

```text
outputs/joint_dependency_v2/standard_gdts_hotel_297d508_20261006/formal_launch/official_gdts_hotel.log
```

官方 checkpoint 仅保存 model state，不含 optimizer、scheduler 或完整 RNG，不能宣称 exact resume。中断后应作为新运行重新开始，不能把权重加载等同于严格续训。
