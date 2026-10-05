# 实现 / 层与 shape / 验收口径

```text
immutable original rows → conservative role migration → new preparation grant
  obs4249: verify old bytes/content → rewrap moved947 or explicit reuse
  train2537/inner1267: window-slot future reader → original Gaussian target12
  outer445: observation only; target route absent

qualified goal entry → obs8 prediction → metric-only target12 → weighted BCE
qualified joint entry → only qualified goal weights + fresh history/diffusion
qualified complete base → grouped cache/teacher → common fresh heads → A0/A1
```

新 family 的 train/inner 原生 agent64 packing 分别175/391 packs；逐窗 Stage-A order 单独 seed3101 登记。
所有读者的 role/purpose 在 provider 前检查；production 与 preparation/smoke grant 不互通。

| 模块 | 保持的结构 / 输入输出 |
|---|---|
| Goal | semantic6 + obs maps8 →14 channels；encoder32/32/64/64/64；decoder64/64/64/32/32；double conv，无BN；输出12帧 logits；613772参数 |
| history | 单层 LSTM input8 / hidden256；每 agent 独立，goal-relative history |
| diffusion | context256+时间3；ConcatSquash；2层 Transformer d512/head4/FF1024；原 tree sampling |
| 注册 template | diffnet.layer 另注册一个2102784参数模板，不是第三个有效 Transformer 层 |
| GDTS | 7826588注册参数，5723804 forward 可达；原 loss=diffusion MSE+20×goal BCE |
| Stage-A | Social128、单GRU/单message、M4/rank8/K21/P20/两轮、strict_no_z；新增参数0 |
| MC | S4、独立目标 RNG；默认 mean-energy 不改，已有接线/损失证据复用 |

上述预期表必须与实际 MODEL_SMOKE 的 architecture/shapes/layer_specs 对照；实际执行失败不以表格代替。
网络文件唯一允许变化是可选 geometry-only dataset 注入，默认旧调用保持。CPU AST 回归会移除该
精确条件赋值/唯一新可选参数后比较旧模型/损失节点；不允许笼统跳过整个 constructor 或网络文件。

训练工程独立新增：

- `p2_grouped.py`：保守分组、不可变成员校验、阶段范围资格；语义需要资产级独立证明，不能拿 ZIP 证明替代。
- `p2_grouped_artifacts.py` / `grouped_artifacts.py`：plan/canary/export/verify，原子 wrapper、用途与逐窗授权。
- `p2_grouped_training.py`：geometry-only 场景、延后 metric target、原损失、加权选择、epoch/状态/父产物合同。
- `grouped_preflight.py`：各条件矩阵、正式 main launch-check、只读父权重资格工具。
- `grouped_pipeline.py`：依赖检查与未来共同初态准备接口；无依赖时退出，不自动运行后续阶段。
- `grouped_model_smoke.py`：CPU/CUDA FP32/BF16 真实路径，SMOKE_ONLY；未训练工程 goal 不作为 joint 正式父权重。

Replay 分开报告模型/optimizer/scheduler/RNG 身份、固定输入和固定 noise 的真实 eval loss 是否逐位一致、
以及预注册容差是否满足。它不是任意中途 CUDA exact resume，不认证历史 index_add 首差、收敛或性能。
真实 inner selection/full391 只有实际来源合格才能执行；train-only 工程预测/损失不是 inner 验收。

正式 fresh 初态在 smoke 后重新 seed3101 构造，optimizer/global/独立 RNG 不继承，路径独立；不是 clean parent。
本轮正式长训练0与有界真实 smoke 更新不矛盾，报告必须给实际次数，不写“真实训练0”的笼统模板。
