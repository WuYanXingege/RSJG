# ZARA1 新桥＋H 主方案长期训练就绪评审

## 结论

最终分类：**READY_FOR_BOUNDED_LONG_TRAIN**。

新桥＋H 值得作为当前 ZARA1 主方案进入一次最多新增 30 轮、带早停的 warm-start 续训。理由是：它相对 matched GDTS 的两项边缘均值不退化、两项联合均值严格改善，而且四项指标均为 5/5 个推理 seed 同方向；完整在线链路为 GDTS 推理时间的 `2.404×`，处于本轮约 `2.5×` 的工程资源参考内。该结论只表示值得进入有上限训练，不保证长期训练必然继续改善，也不构成跨折结论。

此前“新 H 的 JADE 距 Candidate 不超过 0.005 m”的预注册门槛仍为 FAIL，本报告没有改写该历史。这里采用的是不同且预先明确的主目标：相对 matched GDTS 保持边缘指标并改善联合指标。

## 冻结五种子结果

均值 ± 样本标准差，越低越好。五个 seed 是同一训练 checkpoint 的推理随机重复，不是五个训练 seed。

| 方法 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
| matched GDTS | 0.188556 ± 0.001004 | 0.285608 ± 0.001405 | 0.318124 ± 0.002240 | 0.585493 ± 0.004152 |
| 新桥＋G | 0.184354 ± 0.000906 | 0.274816 ± 0.000666 | 0.302674 ± 0.000727 | 0.572203 ± 0.003532 |
| 新桥＋H | **0.183750 ± 0.000796** | **0.274816 ± 0.000666** | 0.287978 ± 0.002386 | 0.535355 ± 0.005775 |
| 原 Candidate | 0.192292 ± 0.000934 | 0.309826 ± 0.003587 | **0.275205 ± 0.000737** | **0.510408 ± 0.003630** |

### 新桥＋H − matched GDTS（逐 seed）

负值表示 H 更好。

| seed | ΔminADE | ΔminFDE | ΔJADE | ΔJFDE |
|---:|---:|---:|---:|---:|
| 2035 | -0.004025 | -0.010572 | -0.031935 | -0.044086 |
| 2036 | -0.004839 | -0.010183 | -0.026531 | -0.045201 |
| 2037 | -0.005947 | -0.009561 | -0.033535 | -0.058654 |
| 2038 | -0.005001 | -0.011673 | -0.029735 | -0.051074 |
| 2039 | -0.004214 | -0.011970 | -0.028996 | -0.051675 |
| 均值差 | **-0.004805** | **-0.010792** | **-0.030147** | **-0.050138** |

四项均为 5/5 seed 改善。故 `marginal_pass=true`、`joint_pass=true`、`joint_direction_stable=true`，主性能合同通过。

### 其余固定比较

| 比较（左−右） | ΔminADE | ΔminFDE | ΔJADE | ΔJFDE | 改善 seed 数 |
|---|---:|---:|---:|---:|---:|
| 新 G − GDTS | -0.004201 | -0.010792 | -0.015451 | -0.013290 | 5/5、5/5、5/5、5/5 |
| 新 H − 新 G | -0.000604 | 0 | -0.014696 | -0.036848 | 5/5、0/5（5/5相等）、5/5、5/5 |
| 新 H − Candidate | -0.008542 | -0.035010 | +0.012773 | +0.024947 | 5/5、5/5、0/5、0/5 |

新 G 已达到原研究目标；MPJC 的 H 在不改变逐人终点集合时进一步改善联合指标。相对自由终点 Candidate，H 换回了更好的边缘覆盖，但付出 JADE/JFDE 代价。固定 H 后 minFDE/JFDE 由 bank 决定，后续桥训练主要可能改变 minADE/JADE，不能承诺靠桥训练超过 Candidate 的 JFDE。

## 完整系统成本

计时口径为相同 705 场景、2356 行人、K=20/20 DDIM；阶段按实际串行在线链路相加，而不是只报告缓存 H 后的桥推理。

| 系统 | 实际在线组成 | 秒 | 相对 GDTS | 峰值 CUDA allocated |
|---|---|---:|---:|---:|
| GDTS | 原生完整推理 | 290.246 | 1.000× | 250,549,760 B |
| 新 G | 原生 proposal provider＋新条件桥 | 493.900 | 1.702× | 692,324,352 B |
| 新 H | provider＋Candidate reference＋MPJC＋新条件桥 | 697.622 | **2.404×** | 847,810,048 B |

新 H 的分项均值为 `290.246 + 206.337 + 0.223 + 200.816 s`。在线总参数 `22,399,316`：未来可训练的新桥 `7,286,364`，冻结 provider 与 Candidate reference 合计 `15,112,952`；MPJC 无可训练参数。峰值按串行阶段最大值报告，不把互不同时驻留的峰值相加。

这不是“只训练三轮”的完整方法：proposal provider 依赖已完成的 GDTS goal/baseline 训练；Candidate reference 又继承 epoch96 主干和原生先验短阶段；新桥在其上完成 E3。后续 30 轮仅是新桥的 warm-start 续训，provider/reference 始终冻结且训练时不在线运行。

本轮新增 GPU 成本 `1667.811 s`：补齐 GDTS 2036–2039 为 `1153.346 s`，GDTS seed2035 资源回放 `302.777 s`，新 H seed2035 资源回放 `211.688 s`。未进行 optimizer update。

## 唯一有界长训合同

- 从 E3 checkpoint `9ecf25...288cb` 的完整 optimizer、CPU/CUDA RNG 和 loader RNG 状态恢复；父状态为 epoch 9/update 810。
- seed 3101；最多新增 30 轮；主干/U-Net 初始 LR `2e-5/2e-6`，在 30 个端点含括的 epoch 上余弦下降至初始值的 `0.1`。
- 192 hidden、4 heads、FFN768、6 FJAB，FULL、legacy CTF、Raw；rollout 与 paired actual-step projection 始终关闭。
- 训练条件只用增强后 GT endpoint；验证使用冻结 H seed2035 bank，不读取验证 GT，不运行 provider/reference。
- 每轮完整 validation；相对 GDTS seed2035 四项满足合同才 eligible；在 eligible 中选最低 JADE。E3 本身作为初始候选保留。
- JADE 实际下降即保存 eligible 最低点；下降至少 `0.0005 m` 才重置 patience；连续 8 轮未达到该改善停止。
- 训练后只对唯一选中 checkpoint 做一次 2035–2039 正式评测。GPU 墙钟上限 12 h，达到边界保存完整状态，不自动延长。

E3 最近一轮实测每轮含训练和 seed2035 验证约 `946.20 s`；30 轮约 `7.88 h`，最终 H 五种子约 `0.28 h`。checkpoint 保存开销未完整包含，因此这是估计而非完整 epoch 保证，但在 12 h 内留有明显余量。

配置：[zara1_gt_bridge_h_bounded_long_v1.yaml](../../../configs/research/zara1_gt_bridge_h_bounded_long_v1.yaml)

实际入口：[run_zara1_gt_bridge_bounded_long.py](../../../scripts/run_zara1_gt_bridge_bounded_long.py)

启动时把当前冻结 HEAD 写入运行身份并由入口再次核对；该命令不会在本次评审中执行：

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_Paper
rsjg_commit="$(git rev-parse HEAD)"
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python -u \
  scripts/run_zara1_gt_bridge_bounded_long.py \
  --plan-config configs/research/zara1_gt_bridge_h_bounded_long_v1.yaml \
  --output /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/zara1_gt_bridge_h_bounded_long_v1 \
  --source-commit "$rsjg_commit" \
  --device cuda
```

CPU preflight 已 strict-load 模型与完整 optimizer 状态（301 个状态项）、恢复 RNG、验证 H bank 身份；结果为 `PREFLIGHT_OK`，optimizer updates=`0`。本轮未启动长期训练，当前无本任务训练进程。

