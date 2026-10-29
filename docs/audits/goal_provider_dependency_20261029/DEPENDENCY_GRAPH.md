# 当前原生终点候选 G 的真实依赖

## 实际数据链路

```text
同步窗口原始数据
  └─ legacy dataset / GDTS.prepare_inputs
       ├─ 6 通道语义图 tensor_image
       ├─ 8 帧观测轨迹热图 input_traj_maps[:, :8]
       └─ scene 像素—米制坐标变换
             │
             ▼
cat([semantic6, observed_heatmaps8])                 [N,14,H,W]
             │
             ▼
GDTS.goal_module (UNet: encoder → decoder → head)   [N,12,H,W]
             │ 仅使用最后一个未来通道
             ▼
sigmoid(goal_logits[:, -1:])                         [N,1,H,W]
             │
             ▼
TTST_test_time_sampling_trick
  ├─ 每人 multinomial 采样 1000 点
  ├─ 每人 k-means 聚为 20 个中心
  └─ 原函数追加 argmax 为第 21 项；正式导出保留前 20 项
             │
             ▼
map-grid G × down_factor(8)
             │
             ▼
scene.make_world_coord_torch                         [20,N,2] metres
```

## 被调用但不影响 G 的模块

`GDTS._legacy_encode(..., if_test=True)` 在生成候选后，会针对 21 个候选逐个调用 `encoder.encode_hist` 生成历史上下文。该上下文用于后续轨迹生成，但没有回流到 `auxiliary["goal_point"]`。因此历史编码器是参考函数中的额外计算，不是 G 的数据依赖。

`diffnet`、DDIM 轨迹扩散/去噪和最终轨迹头在训练 proposal bank 导出路径中均未被调用；它们不是 G 的运行结构依赖。验证 endpoint exporter 为同时取得最终轨迹而调用完整路径，但捕获的 proposal 仍在完整轨迹去噪之前由上述链路产生。

## 权重来源链路

```text
Goal-pretrain checkpoint
  └─ strict 初始化 goal_module
       └─ matched GDTS baseline stage
            ├─ goal_module 进入 optimizer（未冻结）
            ├─ registrar 进入 optimizer
            └─ diffnet 进入 optimizer
                 └─ selected matched checkpoint 中的 goal_module
                      └─ 当前正式 G 的实际权重来源
```

两折的 Goal-pretrain 与 matched checkpoint 都含完整且同构的 46 个 Goal U-Net 参数张量；但 matched baseline 之后 46/46 张量均已变化。当前网络没有 BatchNorm，因此该 Goal U-Net 没有 running mean/variance 等运行 buffer。

## 三个验证对象

- A：完整 GDTS 对象加载 matched checkpoint，调用真实 `_legacy_encode`。
- B：独立 `NativeGoalProvider`，仅 strict-load Goal-pretrain 的 46 个张量。
- C：同一独立提供器，仅 strict-load matched checkpoint 的 46 个 Goal U-Net 张量；不实例化 registrar 或 diffnet，也不从旧源码树导入网络/TTST 实现。

A 的参考数据加载仍使用现有冻结 GDTS 数据预处理，以保证输入张量和坐标变换与正式导出完全一致；B/C 对同一份已准备输入进行配对回放。
