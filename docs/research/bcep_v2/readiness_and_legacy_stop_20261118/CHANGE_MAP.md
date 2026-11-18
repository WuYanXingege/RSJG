# 改动与冻结合同

| 入口 | 修补及归约 | 验收 |
|---|---|---|
| `bcep_v2/bank.py::export, Bank` | 窗口边界检查暂停。原子提交候选及其行收据，收据绑定窗口身份、bank 身份和 SHA256。只在完整有序成员集通过后写完成标志。孤立数据文件须原身份回放相符才能补收据。 | 合成首行、末行、两个 bank 的间隙、篡改拒绝与逐行恢复 |
| `bcep_v2/workflow.py::execute, run_command, cli` | 每个 bank 前后及核验已提交行时可请求暂停。暂停写独立收据，进程退出 75。 | 子进程真实 SIGTERM；shell 条件链的下一阶段未启动 |
| `bcep_v2/stages.py::head_callbacks` | 按窗口缓存一次冻结编码，每个 seed 分别读候选、查询空间特征并计算小头。按各自有效行人数计算 minFDE，再取五 seed 等权均值。 | 2 个窗口、3 次行人出现，10 → 2 次编码；小头仍调用 10 次 |
| `bcep_v2/provider.py::head_loss_components`、`stages.py::density_callbacks` | 小头累计最近终点距离、每人 20 个位移平方的均值和实际总损失。密度直接取已有 final/h4/h8 NLL 分子和行人计数。 | 验证分子、分母、尾批与已有反向计算 |
| `bcep_v2/joint.py::clean_path_terms`、`objective.py::bridge_and_pair_loss` | 复用既有单次去噪结果。记录 clean-path 原始、按信号加权及进入目标的贡献，保留 diffusion、geometry、map、pair 分项。 | 物理分组和累积按原系数加总；测试核对各分项之和 |
| `bcep_v2/state.py::run_stage` | 每更新平均、原始分子和计数分别累计；完成一轮时按各自分母派生指标。部分轮次断点随游标一起持久化。 | 旧状态 schema 拒绝；实际三阶段重建后逐值恢复 |
| `bcep_v2/joint.py::neighbors, build_joint`、`bcep_v2/runtime.py` | 严格确定性下通过排序取同一邻居距离阈值。v2 CTF 使用 PyTorch 双线性采样分解式，MSCE 的等尺寸池化窗口使用相同边界的普通平均池化。共用基类各增加一个可替换算子。 | 同区域池化及梯度、双线性值及双输入梯度；CUDA 重复 receiver 的归约与实际模型恢复 |
| `tests/test_bcep_integrated_v2.py`、`tests/test_bcep_v2_readiness.py`、`bcep_v2/check_readiness.py` | CPU 状态/CLI/排列检查与独立合成 GPU 入口。GPU 入口只从已核验哈希的成员元数据读取最大人数。 | 预算、逐项结果及失败重试见 [验收收据](TEST_RESULTS.json) |

**配置身份。** 模型仍使用 `configs/bcep_ebjd_v2/RESOLVED_CONFIG.yaml`，文件 SHA256 为 `0d4034a54443a40f4b046a8e88a423c64f2b7d3ad4e7f14f2b7de0d0be535985`。两个训练开关均为 false。四层 Flow、M256/K20、KMeans++ 五次初始化、最多 100 轮、1e−6 m 容差、0.5 m 修正半径、一轮反馈、六层条件桥、20 步 DDIM、数据划分、各项损失权重及三阶段选模沿用该配置。更新的是运行时、bank 行收据和训练状态格式，原基础参数张量的名称与形状兼容。

**统计口径。** `train_nll_final_agent_micro`、`train_h4_nll_agent_micro`、`train_h8_nll_agent_micro` 是训练中参数逐步变化时的在线统计。小头的分母也是有效行人出现次数。clean-path 的原始和信号加权项分别按有效场景求均值；`train_mean_per_update.clean_path_term` 保留进入反向目标的 0.05、物理组场景比及累积系数。diffusion、map 按物理组行人占比，geometry、pair 按组内场景占比。原 `train_mean_per_update.loss` 继续表示复合目标。完整 epoch 记录同时保存 `train_raw_sums` 和 `train_counts`。
