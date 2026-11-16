# BCEP 首轮训练前定点修复与验收

结论：**READY_FOR_SEPARATE_AUTHORIZATION**。已修复社交几何缺失、复合 loss 与终点 NLL 混称、导出成员不受内部划分约束三项问题，并补齐步边界预算与完整恢复。原有 8 项及新增 17 项 CPU 测试全部通过。**真实数据 optimizer updates=0；未启动 pilot、训练队列或 supervisor。没有得到新的预测精度结论。**

执行依据为 `BCEP_PrePilot_Fixes_Package.zip`；审查基准为 `62f8cc6cbf13233293805c14440703f37c53c385`。开发仅发生在 `RSJG_EBJD_BCEP` 的 `research/bcep-prepilot-fixes-20261116` 分支。恢复主工作区仍为 `5d07940b4fd0565d37af4f7219277cbba1b074fe`。

## 1. 复现与最小修改

| 已确认问题 | 修复 | 证据与边界 |
|---|---|---|
| 旧社交注意力只接收每人去掉最后位置的运动编码；邻人整体平移不可见 | 在原单层聚合中加入相对位置、相对速度、距离平方的 edge MLP，同时进入 attention bias 和 value | 实际旧源码 FP64 反例：邻人距目标从 0.5m 变为 8.5m，目标 context 最大差 **0**；改邻人速度则为 **0.186455**。修复后相同输入构造的位置干预差为 **1.260431**，共同平移差仍为 **0**。这是表示依赖检查，不是准确率提升证据 |
| 原 `model.loss()` 是 final NLL 加辅助 horizon NLL，却登记成 train_nll/dev_nll，并按窗口平均选模 | 新增 `loss_components`，保留训练目标；单独记录 final sum/count、各辅助项及 Jacobian 标识；选模固定为 `dev_nll_final_agent_micro` | N=1 与 N=3 合成统计夹具给出 micro=2.5、macro=2.0；实际 BCEP 新旧复合目标公式逐值一致 |
| 原 exporter 遍历整个 manifest，不使用 dev_indices，且固定写更新数0 | 必须指定 subset/purpose；按锁定 indices 导出并校验完整身份；分别记录 checkpoint_training_updates 与 export_optimizer_updates | 真实 train/dev 划分未变；实际 CLI 的错误 manifest、split、训练成员混入、遗漏、checkpoint 来源、协议、purpose、smoke/截断冒充完整等负控均被拒绝 |

原 exporter 问题是首次真实训练前的风险：本轮没有证据显示已经发生过性能结果污染。修复前源码和诊断保存在本轮资产目录，Git 中保留小型复现收据。

## 2. 几何输入的含义

`bcep/context_encoder.py` 中仍是 128 维、4 heads、2 层时序编码和 **1 次**社交聚合；地图 CNN 保持 32/64/128，flow 保持四层二维 RQS。新增 edge MLP 为 5→32→32，32→4 attention bias 和 32→128 value projection；当前 BCEP 总参数量 626,144。

输入为 `p_j(last)-p_i(last)`、最后三个观测间隔平均速度的差、相对位置平方和。位置按 1m、速度按 1m/s、距离平方按 1m² 固定归一化，无 dev 拟合。NaN padding 在构造两两几何前置零，invalid key 被 mask，invalid query 输出置零；保留 self，使单人场景有效。edge value 避免“邻人运动相同，单改权重仍无法改变消息”的退化。

测试覆盖共同平移、agent 置换、无地图、NaN padding、单人，以及非 identity 初始化 flow 的真实输入依赖和 edge 分支梯度。没有声称旋转不变或物理约束；relative geometry/attention 是恢复必需信息，不作为原创算法贡献。新增参数仅属于未训练的 BCEP 原型，不修改、迁移或加载旧六层/GDTS/八层权重。

## 3. 监督、选模与导出合同

训练仍按每场景一次 update 优化 `final_NLL_mean + 0.25 × auxiliary_horizon_NLL_mean`，未改变场景权重。dev 同时记录：

- `dev_loss_total_scene_macro`：原复合目标的窗口平均，仅诊断；
- `dev_nll_final_agent_micro`：最终世界坐标 NLL 的有效行人 sum/count，唯一主选模量；
- `dev_nll_final_scene_macro`、final sum/count、各辅助 horizon sum/count 与 micro/macro。

真实改善更新 best；相等保留较早 epoch。完整 best checkpoint 记录 selection_metric_name、selection_value、selected_epoch 和 split_hash；不使用 coverage 成绩选密度 checkpoint。

`evaluate_bcep_bank.py` 必须显式选择 `internal_train`、`internal_dev`、`interface_smoke` 或 `historical_reference`，并匹配 purpose。原源 manifest、每个 shard、split、逐场景 sequence/scene/frame/agent、训练/开发分离、尺度来源、checkpoint schema/source/code 和单位/时间协议均被校验。历史 manifest 目前没有登记，故该入口会拒绝，需未来明确只读绑定。

每个 scene/seed **只抽一次 M=256 IID 支持集**。IID20 为其前20个原子；coverage20 为同一支持集上的贪心选择。五个固定推理 seed 为2035–2039；身份局部 RNG 不随调用顺序前移。GT 在两个候选输出完成后才用于评分。输出 minFDE、Recall@0.25/0.5/1m、P50/P90/P95、行人数及窗口数，并分 N=1/N≥2；NLL 来自原密度，energy score 来自 IID256。均值/样本SD和逐seed配对差也会保存，不将coverage代表点视为IID概率样本。

随机初始化、limit、smoke、未训练权重、合成数据明确标记 `NONFORMAL`；`--require-complete` 拒绝这些输入。bank 记录训练更新数和导出更新数，不再把“导出没有更新”等同于“checkpoint未训练”。任何 internal_dev 结果均明确不是独立外层最终测试。

## 4. 有界 runner 与恢复证据

`bcep/pilot_runner.py` 为 CLI 与合成测试共用 runner。预注册上限未变：10 epochs、3600秒、patience3、seed3101、lr1e-4、lambda_aux0.25、M256/K20；新增总更新上限 **37,900=3,790×10**。`configs/research/bcep_prepilot_v2.yaml` 的 `allow_real_training` 仍为 false。

预算是累计**训练流程墙钟**，含数据合同准备、模型/optimizer构造、训练、验证和保存，并非 GPU active time。每个训练/验证场景前检查；已耗尽预算或更新数的恢复不会再开启新训练步。允许最多一个在途场景和保存开销越界，不声称硬实时抢占。checkpoint 与 RUN_STATE 收据用临时文件加原子替换保存；正常恢复合并已记录保存开销。

部分 epoch 保存 shuffled order、train/dev cursor、统计累加器、Adam、scheduler、Python/NumPy/Torch/loader RNG、best/stale、epoch/update 和累计时间；只有完整验证结束才推进 scheduler 和 completed_epoch。CPU 合成数据连续8步与3步保存后再5步，对参数、Adam、scheduler、数据顺序、全部相关RNG及计数逐值一致，跨越两个完整 epoch。预算耗尽、更新数耗尽、选模平局分别有定向测试。

这验证的是受控步边界中断恢复；没有承诺任意断电、SIGKILL 或磁盘损坏都能保留尚未落盘更新。CUDA数值及GPU恢复未测，不把CPU验收外推为GPU逐位一致。

## 5. 数据身份与验收范围

划分保持 **3790 internal_train / 320 internal_dev**，dev sequence=`uni_examples`。split SHA256：`ebe0e117ee8e1a36e72028cfb5e6b3759df7b96821b94ea8a9a478fcbf149188`。完整来源和成员hash见 `DATA_AND_EVAL_CONTRACT.json`。

本轮真实数据仅取锁定 train/dev 各1个窗口、合计4次有效行人出现，CPU无梯度、随机初始化、五推理seed接口导出。其数值不进入性能报告，不满足320窗口正式dev验收。完整真实训练、GPU数值/速度、预测精度、校准和外层泛化均未执行。详细25项收据见 `TEST_RECEIPT.json`；旧8项测试文件未更改、容差未放宽。

旧 matched 模型接触过当前内部dev，不能作为公平胜负对照；历史0.5811m/64.1%也不属于此评测身份。未来首轮唯一预注册筛查是同checkpoint下 coverage20 相对IID20：dev五seed平均minFDE改善≥0.01m，Recall@0.5不降，P90不升。通过只支持继续研究这一选择器组合，不能证明flow胜过其他密度或已替代旧provider。

独立小GMM/MDN、多训练seed、更多内部留出序列及全系统从零训练仍为后续计划，本轮没有实现或执行这些实验。

## 6. 保护与交付

恢复主工作区 HEAD 和完整 `git status --short --branch` 前后相同；两个cleanup目录及原未跟踪研究文档保持原样。五折正式选中六层 checkpoint 与 resolved config 共10项 SHA256 均与恢复登记一致（仅字节哈希核验，不反序列化权重）。没有读取GDTS/旧八层模型参数、教师热图或教师bank，没有删除资产。主机检查未发现项目训练/supervisor进程；系统服务未干预。

本轮大型/临时产物仅在 `RSJG_EBJD_assets/analyses/bcep_prepilot_fixes_20261116_190248/` 和pytest临时目录；Git仅提交代码、定向测试、版本配置与小型报告。最终建议是：**修复已足以进入单独授权的有界BCEP pilot，但当前不启动；在新dev合同下得到训练证据前，继续保留已恢复六层方案作为正式结果来源。**
