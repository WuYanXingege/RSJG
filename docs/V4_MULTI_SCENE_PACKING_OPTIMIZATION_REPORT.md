# V4 Multi-Scene Packing 工程优化报告

更新时间：2026-09-10（Asia/Shanghai）

## 1. 结论

本次优化已经完成并通过测试。V4 的网络结构、轨迹对关系定义、pair energy、pair-specific gate、multiway synchronization、Hungarian 投影、全部 loss 数学定义、K=20 语义及 JADE/JFDE/minADE/minFDE 定义均未改变。

新路径由三部分组成：

1. 冻结 Stage0 上游的多 seed 原始轨迹库；
2. variable-N 多场景动态 packing；
3. 严格逐场景归约的 scene-balanced loss。

真实 UNIV 小规模基准中，同一窗口与同一 seed 的在线结果和缓存结果最大逐元素误差为 **0.0**。相对旧的在线单场景训练，缓存单场景吞吐提高约 **11.75 倍**，缓存加多场景 packing 提高约 **26.24 倍**。

## 2. 旧训练停止与现场保留

- 停止的进程组：PGID/SID `89479`；主 Python PID `89482`，DataLoader 子进程 PID `92172/92173`。
- 停止方式：向已经核验的进程组发送 `SIGTERM`，随后确认所有进程退出。
- 原实验目录：`output/univ/joint/runs/multiway_coupling_v4_stage0`。
- 停止进度：Epoch 2，约 `1430/2982`，即约 48%。
- 已完整保留：Epoch 1 `best_model.pt`、配置、evaluation protocol、validation diagnostics、曲线及完整训练日志。
- 新 packed 实验没有续训这个临时 checkpoint，而是从固定 Stage0 上游权重重新初始化 V4。

## 3. Frozen trajectory-bank cache

实现文件：

- `src/trajectory_bank_cache.py`
- `src/trajectory_bank_builder.py`

每个同步窗口保存：

- `window_id`、`scene_name`、`frame_ids`、`agent_ids`；
- `obs_world [N,T_obs,2]`；
- `gt_future_world [N,T_pred,2]`；
- `future_mask`、`agent_mask`；
- `raw_trajectory_banks [S_seed,N,K,T_pred,2]`；
- 上游 checkpoint SHA-256/path/type、K、时长、dt、坐标系、缓存版本等 metadata。

没有缓存任何可训练 V4 输出：trajectory feature、relation posterior、pair score、pair gate、Sinkhorn/permutation 都在训练时在线计算。

### Seed 规则

使用 SHA-256 实现稳定 hash：

```text
seed(w,q) = hash(cache_version, seed_base, scene_name, frame_start, q)
```

正式配置为 `num_cached_seeds_per_window=4`、`trajectory_bank_seed_base=42`。训练时每个窗口每轮只选一个 q；同一次 forward 只有 K=20。验证固定 q=0；正式测试分别运行 q=0..3，再报告 mean/std，绝不拼成 K=80。

### Manifest 与恢复

`trajectory_bank_manifest.json` 记录数据集、held-out set、K、seed 数、Stage0 checkpoint SHA-256/size/mtime、DDPM/DDIM/trunk 参数、坐标约定和内部验证划分。任何不一致都会明确失败。只有显式 `--force_rebuild_trajectory_bank True` 才会重建，且旧目录先被重命名为带时间戳的 stale 备份。

每个窗口采用临时文件加 `os.replace` 原子发布。构建中断后再次执行会校验并复用已有完整窗口，继续生成其余窗口。

## 4. Multi-scene packing

数据加载采用 map-style cache dataset、动态 batch sampler 和自定义 collate：

```text
obs_world       [N_total,T_obs,2]
gt_future_world [N_total,T_pred,2]
raw_future      [N_total,K,T_pred,2]
scene_index     [N_total]
scene_ptr       [num_scenes+1]
```

同时保留 scene name、window id、frame ids、agent ids、所选 seed index 和 seed value。

正式 packing 限制：

- `max_agents_per_pack=32`
- `max_edges_per_pack=256`
- `max_scenes_per_pack=8`

Sampler 用每个窗口完整图的边数上界做贪心打包，因此真实 radius+TTC 图不会超过估算值。同步窗口绝不拆分；若单个窗口本身超过限制，只能作为明确标记的 oversized singleton pack。

## 5. 图隔离与稀疏张量

每个场景单独调用 interaction graph。得到局部 `[2,E_s]` 后才加 agent offset，再拼接成 `[2,E_total]`。运行时断言每条边两端具有相同 `scene_index`。即使两个不同场景的坐标完全相同，也不能产生跨场景边。

V4 始终只构造 `[E_total,K,K,...]` 的稀疏 pair 张量，没有构造 `[N_total,N_total,K,K]` 全连接密集张量。日志会给出 scenes/agents/edges、pair elements 与估算内存。

## 6. Packed V4 forward

一次 packed forward 中：

1. 对 `[N_total,K,T_pred,2]` 编码完整轨迹；
2. 仅在场景内的稀疏边上构造 trajectory-pair geometry；
3. 在线计算 relation posterior、pair energy 和 pair-specific gate；
4. synchronizer 按 `scene_index`/connected component 独立同步；
5. hard evaluation 对每个分量独立做 Hungarian；
6. 最终轨迹只通过整数 gather 重排，不插值、不移动坐标。

## 7. Scene-balanced loss

`compute_scene_balanced_multiway_coupling_loss` 不重写原 loss。它先按场景切出 node/edge tensors，将 edge index 映射回局部编号，然后逐场景调用原 `compute_multiway_coupling_loss`，最后对每个 loss component 求场景均值：

```text
L_pack = (1 / S) * sum_s L_scene_s
```

因此 alignment、pair score、pair assignment、no-harm、permutation entropy、relation prior KL 和 gate regularization 都不会被大 N 或大 E 场景额外加权。测试逐分量验证 packed loss 与对应 single-scene losses 的算术平均严格一致。

## 8. 旧路径兼容

下列开关默认均为 False，旧 online/single-scene V4 路径仍然存在：

- `use_trajectory_bank_cache`
- `use_multi_scene_packing`
- `scene_balanced_loss`

正式 packed 路径要求 cache=True、packing=True、scene-balanced=True；非法组合由 parser 直接拒绝。`coupling_grad_accum_steps` 与 packing 独立；正式 packed 实验使用真实多场景 pack，设为 1。

## 9. 测试结果

完整测试：**181 passed**。

覆盖内容包括：

- deterministic window/seed；
- K=20 与 seed 轴不合并；
- variable-N packing 与三种限制；
- graph offset 和无跨场景边；
- 相同坐标的不同场景仍隔离；
- packed/single trajectory feature、relation posterior、pair score、gate、soft P、Hungarian、aligned trajectories 等价；
- scene-balanced 每一项 loss 精确等价；
- N=1/E=0 有限且可反传；
- hard permutation 为双射；
- minADE/minFDE 的边际轨迹集合保持不变；
- packed 端到端反向传播、validation、日志与 checkpoint 保存。

真实在线/缓存等价测试：

- scene：hotel
- frame start：10550
- seed index：0
- seed：193265491
- K：20
- `atol=rtol=1e-6`
- max absolute error：**0.0**

## 10. A/B/C benchmark

基准文件：`output/univ/joint/analysis/v4_packing_benchmark_v2.json`。

| 路径 | windows/s | agents/s | 平均 scenes/pack | 峰值 allocated | 相对 A |
|---|---:|---:|---:|---:|---:|
| A online + single | 2.017 | 4.54 | 1.0 | 164.2 MiB | 1.00x |
| B cached + single | 23.690 | 53.30 | 1.0 | 89.0 MiB | 11.75x |
| C cached + packed | 52.913 | 119.06 | 6.0 | 167.7 MiB | 26.24x |

短基准中 C 的平均 pack 为 13.5 agents / 10.5 actual edges，GPU 采样时间很短，因此 utilization 数值只作烟测参考；正式 cache 构建启动后实测 GPU utilization 为约 94%，显存约 1.6 GiB（ETH 与 UNIV 两个缓存任务合计）。

## 11. 正式 ETH/UNIV 流水线

统一启动脚本：`train_packed_multiway_v4.sh`。

### ETH

- pipeline PID：`96599`
- 状态：`cache_building`
- run：`output/eth/joint/runs/multiway_coupling_v4_packed_cache_v2`
- cache：`output/eth/joint/trajectory_bank_cache_v1/stage0_epoch100_s4_k20`
- train log：`packed_v4_train.log`

### UNIV

- pipeline PID：`96606`
- 状态：`cache_building`
- run：`output/univ/joint/runs/multiway_coupling_v4_packed_cache_v2`
- cache：`output/univ/joint/trajectory_bank_cache_v1/stage0_epoch100_s4_k20`
- train log：`packed_v4_train.log`

两个 pipeline 都由 `nohup + setsid` 启动。当前先生成完整的 4-seed trajectory-bank cache；cache manifest 标记 complete 后，脚本会自动从 Epoch 1 启动独立的 30 轮 packed V4 训练，随后用 best checkpoint 对 4 个缓存 seed 分别测试并写 mean/std。

可用以下文件检查阶段和日志：

```bash
cat output/eth/joint/runs/multiway_coupling_v4_packed_cache_v2/pipeline_state.txt
cat output/univ/joint/runs/multiway_coupling_v4_packed_cache_v2/pipeline_state.txt
tail -f output/eth/joint/runs/multiway_coupling_v4_packed_cache_v2/trajectory_cache_build.log
tail -f output/univ/joint/runs/multiway_coupling_v4_packed_cache_v2/trajectory_cache_build.log
```

正式训练会使用固定 Stage0 `epoch_100.pt` 作为上游，并从头随机初始化 V4；不会加载旧单场景 V4 的 Epoch 1 权重。
