# 正式入口、监测和恢复接口

## 固定执行链

`bcep_scene_energy.formal` → 授权/配置文件SHA/源码/准入收据校验 → 独占本次运行和历史受保护控制锁 → 原`FormalInputs` → 同seed3201的Energy-only工厂 → 原`joint_callbacks` grouped backward → 原AdamW/epoch调度 → 原`run_stage`/原子checkpoint/早停/Top5。

新增文件只有 `formal.py`、`formal_contract.py`、`formal_callbacks.py`、`formal_monitor.py`、`formal_validation.py`。固定基点已有的SSEC网络、候选身份处理、v2/v3入口、损失、优化器及状态控制器未改。新增代码没有`optimizer.step()`调用；未来经单独授权后，唯一真实更新仍由原`run_stage`执行。

正式CLI只接受 `prepare / train / status` 和 `--frozen-sha`，没有fold、seed、变体、测试集或重启新训练的覆盖选项。目录固定为独立的 `zara2_ssec_energy_stage2_20261121`，恢复只读取该目录的`joint/last.pt`。

## 双重授权与初始化

`train`必须同时满足：环境变量`RUN_ONE_ZARA2_SSEC=true`、精确的`FROZEN_CONFIG.json`**文件字节SHA256**、当前源码逐文件哈希、与配置绑定的READY收据。缺标志、缺SHA、错误SHA、错误来源或未通过验收，均在正式数据/模型/优化器构造前拒绝。其后再次构造并核对输入与随机初态，防止冻结文件与现场数据不符。

冻结配置始终保存`allow_real_training=false`和本轮零更新事实；未来授权单独记录于本运行的`AUTHORIZATION.json`，不会改写冻结配置或旧审计记录。原审核用途的`bcep_scene_energy.interface`仍拒绝训练；仅新`formal`入口有受控执行能力。

模型由原v2随机下游工厂构造，保留同seed的共享编码器/S0/六层桥，再构建已验收φ/ρ。没有读取训练后v2/v3权重作为初始化。Provider完全在优化器之外，Stage2只读取其已冻结refined bank。

## 候选身份如何通过原生分组训练

原`joint_callbacks`的微批、时间/噪声抽样、同步旋转反射、人数压缩与损失加权一行未改。这个冻结实现只改变坐标或截取场景/行人，**不重排候选K轴**。因此由已验证bank读取的当前K下标仍是原bank槽位。

`SlotModelView`绑定这些原槽位，再调用已验收的SSEC身份规范化；不是把任意重排后的数组重新标成原ID。外部重排输入必须传递原有ID，正式数据路径没有此操作。全套输入、增强、分组源码均写入冻结清单，源码变化会在授权闸门处拒绝。

验证通过 `bank_occurrence_ids` 携带真实bank实例身份，由既有SSEC推理适配器恢复原槽位、调用原20步DDIM。配对只调用一次；原noise key `(scene,agent,original_slot,2035)`保持不变。真实N=1/2/9/57探针捕获了实际sampler噪声，逐值核对，不仅检查生成期望值的函数。

## 每轮输出

唯一选模返回值仍是原 `MetricAccumulator` 的全590窗H-JADE。每个完整epoch记录：

- `dev/epoch_NNN_windows.csv`：匿名窗口索引/人数、H-JADE、原始G/初配I/最终H的JFDE、I−H、候选边缘minFDE、SSEC决策、评分和成本。
- `dev/epoch_NNN_groups.json`：all和N≥9；JADE/FDE按原规则归约，I/H-JFDE按窗口等权；候选minFDE另保留agent-micro与scene-macro。
- `dev/epoch_NNN_costs.csv`：1、2–8、9–19、20–41、42–57人数区间的前向/DDIM时间及allocated峰值。区间42–57在本开发集中只有42人窗口，不冒称开发集有57人。
- `MODULE_UPDATES.jsonl`：每次实际原生更新的preclip梯度、post-step实际参数差、逻辑组前反向时间、显存及分配决策。φ/ρ与共享编码器分别记录。注册的是原AdamW的post hook，不替换更新公式。
- `dev/epoch_NNN_training.json`：完整轮的168次更新汇总。恢复后保留所有日志尝试，按逻辑update编号取最后一条；缺失完整轮记录则拒绝输出完整轮汇总。

接受/拒绝/严格并列/可靠下降/舍入不确定计数针对**协调提案**，不等于S0首配唯一性。可靠下降候选数和最终接受数是不同概念。能量记录采用原FP64核验数值，不是米单位的JFDE。

`final_proposal_effective_score_RMS`及匹配gap来自该人的最后一次实际协调提案评分；其余人后来可能更新，不能把它解释为“在最终完整状态下重新求解的全局最优间隔”。熵是相同评分下的原20次Sinkhorn训练松弛；单人旁路按identity计熵0。没有把硬排列熵误当软熵，也没有调整温度。

训练人数成本保留物理组的成员人数和耦合前向时间，按组内最大人数分箱；不把一个混合组的整次反向时间随意归因到每个行人。完整逻辑组时间、原epoch训练/验证总时间同时保留。

## 恢复和保留

实际更新、clip、AdamW、5轮warmup/100轮调度、minimum30/patience15/max100、完整epoch原子保存、best/last及H-JADE Top5全部使用原`bcep_v2.state`。同分取较早epoch；保存所有完整epoch，因此不会将最近5轮错误当成最佳5轮。

本轮真实恢复测试进入原生resume分支，以暂停请求在首次backward/step前退出。使用本轮独立的零更新checkpoint，严格恢复模型、AdamW槽位/param groups、scheduler、CPU/CUDA/Python/NumPy及order/view RNG、待处理顺序、计数和统计；下一批及下一次随机抽样一致。为覆盖非空优化器状态，零动量槽位由测试显式构造，绝非训练产生的动量。

Top5和早停测试执行原控制器的实际writer函数体和实际条件表达式，使用明确标记的测试夹具；不是运行伪训练，也不将这些fixture记成完成的训练epoch。未验证“连续多次真实AdamW更新与中断恢复后的训练轨迹等价”，因为本轮禁止任何step。原控制器历史实现保持不变。
