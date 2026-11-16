# BCEP冻结支持诊断与单次MDN8对照

冻结BCEP回归：PASS（五种子逐值相同）；MDN预检：PASS；MDN训练：FINISHED / PATIENCE_STOP；四格评测：COMPLETE。
PROVIDER_REPLACEMENT=NOT_ESTABLISHED；FULL_EBJD_INDEPENDENCE_PERFORMANCE=NOT_ESTABLISHED。

本次判断：flow的平均端点误差较好，但必要性尚未建立；选择器收益在两种密度上均出现。

BCEP在0.5m处：选中点命中75.27%，支持池命中但选择遗漏15.62%，本次有限支持池未命中9.11%。后两者分别占未覆盖出现次数的63.15%和36.85%。
池未命中不证明密度不存在该模态；选择遗漏不证明贪心实现错误，其目标不使用单条真实未来。

## 冻结身份与实验控制
BCEP固定epoch6 best（22740次更新，SHA cc71f02c3315bb7e9596600689740c644150b44d62f21b24df2fea6e51ec03a4）；没有重启旧run或纳入其第7轮619次更新。
train=3790窗口；dev=uni_examples的320窗口、621次有效行人出现。米制、dt=0.4、预测12帧，观察长度来自原manifest。M256/K20；推理seed2035–2039。每一行IID20取该行同一M256的前20点，coverage20用原选择器。不同密度的支持坐标不相同。
MDN8是标准控制：8个二维全协方差Gaussian分量，context128+horizon_seconds，两层128 SiLU，48维输出；softplus对角+0.001，非对角不约束，稳定logsumexp和三角解。Linear默认随机权重及偏置打破对称，没有GT初始化、额外正则或调参。
编码器初值逐位相同，hash=`77ccaedaa4bff1dc36d6c8670498096ff5d7fc6ec32030693f970058487d0573`；各自独立随机构造，未加载BCEP/GDTS/旧EBJD训练权重。其余motion frame、Jacobian、aux0.25、Adam1e-4、gamma1、场景加权、loader独立Generator和NLL选模规则保持一致。
新增独立MDN schema、runner与导出入口；原bcep/及原train/export身份校验未修改。runner仅替换模型/schema/import，终轮达到更新上限后仍可完成验证（在墙钟预算内）。
同协议不代表参数量、FLOPs或墙钟相同；同推理种子不代表两种密度共享相同支持。

## BCEP支持池与压缩诊断
五seed平均：e_pool=0.285644m，e_cov=0.549166m，mean(gap)=0.263522m。逐seed验证e_cov=e_pool+gap及子集关系。
M256整池最近点是只读诊断，不是K20预测结果、部署选择器或桥输入。参考诊断用float64距离，原K20指标沿用float32；末位舍入差不是候选变化。分位数不作加法归因。

| seed | selected_hit@0.5 | pool_hit_selection_miss | pool_miss | 合计 |
|---:|---:|---:|---:|---:|
| 2035 | 465 | 104 | 52 | 621 |
| 2036 | 460 | 99 | 62 | 621 |
| 2037 | 471 | 94 | 56 | 621 |
| 2038 | 466 | 98 | 57 | 621 |
| 2039 | 475 | 90 | 56 | 621 |

| 半径m | selected_hit | pool_hit_selection_miss | pool_miss |
|---:|---:|---:|---:|
| 0.25 | 36.17% | 46.31% | 17.52% |
| 0.5 | 75.27% | 15.62% | 9.11% |
| 1.0 | 90.34% | 4.22% | 5.44% |

上表比例为五种子等权平均（每seed分母621）；JSON同时保存3105个“出现×seed”的合并计数及分组分位数。它们不是3105个独立行人或独立训练重复。

## 冻结四格结果
均值±样本SD（ddof=1），距离单位m，Recall为比例。SD仅反映同一选中模型的推理seed变化。

| 密度 / 选择 | minFDE | Recall@0.25 | Recall@0.5 | Recall@1 | P50 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flow / iid20 | 0.644648 ± 0.005425 | 0.349114 ± 0.013871 | 0.678583 ± 0.019108 | 0.865378 ± 0.001440 | 0.344916 ± 0.013929 | 1.247080 ± 0.029318 | 2.218317 ± 0.104300 |
| flow / coverage20 | 0.549166 ± 0.009747 | 0.361675 ± 0.020672 | 0.752657 ± 0.009292 | 0.903382 ± 0.003013 | 0.316540 ± 0.012600 | 0.959336 ± 0.034691 | 1.888678 ± 0.080734 |
| mdn8 / iid20 | 0.688871 ± 0.005075 | 0.334944 ± 0.011443 | 0.656683 ± 0.006381 | 0.837681 ± 0.004321 | 0.350821 ± 0.016980 | 1.444676 ± 0.019926 | 2.454314 ± 0.118606 |
| mdn8 / coverage20 | 0.566333 ± 0.012099 | 0.339452 ± 0.012338 | 0.730757 ± 0.005389 | 0.902415 ± 0.004351 | 0.322351 ± 0.006686 | 0.988202 ± 0.013456 | 1.772871 ± 0.070665 |

| 密度 | 世界终点NLL | IID256 energy score | 参数量 | IID20重复率 | coverage20重复率 |
|---|---:|---:|---:|---:|---:|
| flow | 3.421934 | 1.043580 ± 0.001322 | 626144 | 0.000000 | 0.000000 |
| mdn8 | 3.261944 | 1.076174 ± 0.001216 | 520500 | 0.000000 | 0.000000 |

重复率定义为逐行人(20−精确不同坐标数)/20再取均值；未对近邻点施加人为合并阈值。NLL仅用于原连续密度，energy仅用于合法IID256，不在coverage代表点上计算。

| 分组 | 密度/选择 | minFDE | Recall@0.5 | P90 |
|---|---|---:|---:|---:|
| N_eq_1 | flow/iid20 | 0.781607 | 0.604545 | 1.674800 |
| N_eq_1 | flow/coverage20 | 0.630852 | 0.707576 | 1.093267 |
| N_eq_1 | mdn8/iid20 | 0.772830 | 0.590909 | 1.488667 |
| N_eq_1 | mdn8/coverage20 | 0.634531 | 0.674242 | 1.061962 |
| N_ge_2 | flow/iid20 | 0.607677 | 0.698569 | 1.137672 |
| N_ge_2 | flow/coverage20 | 0.527116 | 0.764826 | 0.869782 |
| N_ge_2 | mdn8/iid20 | 0.666208 | 0.674438 | 1.416539 |
| N_ge_2 | mdn8/coverage20 | 0.547924 | 0.746012 | 0.968917 |

## 选择器与密度贡献分开报告
- selector gain(flow)：+0.095481m，5/5种子改善（正数表示coverage更好）。
- selector gain(mdn8)：+0.122538m，5/5种子改善（正数表示coverage更好）。
- density contrast(iid20)=MDN−flow：minFDE +0.044224m，P90 +0.197595m，P95 +0.235996m，Recall@0.5 -2.19个百分点。误差负数为MDN更好。
- density contrast(coverage20)=MDN−flow：minFDE +0.017167m，P90 +0.028866m，P95 -0.115806m，Recall@0.5 -2.19个百分点。误差负数为MDN更好。

## 实际训练和选模窗口
MDN完成4轮 / 15160次真实更新；选中epoch=1，dev NLL=3.2619442757393045；停止原因=PATIENCE_STOP。
早停规则为patience3，best只来自完整320窗口NLL；未用coverage指标挑选epoch。没有第二次run或追加训练。
MDN未完成6轮，比较包含实际训练/选模窗口差异，不能称为严格相同6轮预算完成。选中MDN epoch1仅经历3790次更新，flow best epoch6经历22740次；两者遵循相同停止/选模规则，但实际训练暴露量不同。

| MDN epoch | update | train final NLL | dev final NLL | 累计秒 |
|---:|---:|---:|---:|---:|
| 1 | 3790 | 2.399179 | 3.261944 | 211.34 |
| 2 | 7580 | 2.225407 | 3.328281 | 417.24 |
| 3 | 11370 | 2.137980 | 3.360272 | 623.69 |
| 4 | 15160 | 2.118957 | 3.499400 | 838.33 |

## 定点例子（post-hoc failure visualization）
固定seed2035、0.5m、每类按scene_id/agent_id排序取前2项；仅作诊断，不参与候选选择或证明总体提升。完整支持坐标见资产EXAMPLES.json。

| 类别 | scene_id | agent_id | pool误差m | coverage误差m | gap m |
|---|---|---|---:|---:|---:|
| selected_hit | eth:train:uni_examples:0-190 | 1 | 0.02127 | 0.30244 | 0.28116 |
| selected_hit | eth:train:uni_examples:10-200 | 2 | 0.01803 | 0.16558 | 0.14755 |
| pool_hit_selection_miss | eth:train:uni_examples:1140-1330 | 21 | 0.05584 | 0.52788 | 0.47204 |
| pool_hit_selection_miss | eth:train:uni_examples:1170-1360 | 19 | 0.28576 | 0.77947 | 0.49372 |
| pool_miss | eth:train:uni_examples:1160-1350 | 19 | 0.72658 | 1.48598 | 0.75940 |
| pool_miss | eth:train:uni_examples:1550-1740 | 25 | 1.56750 | 1.91894 | 0.35144 |

## 成本、复现和保护
本轮合成/接口预检：11.525s /600s（35项CPU测试含原25项；GPU合成16次更新、真实无梯度N=1/3/57）。合成GPU连续8步与3+5恢复逐位一致。
MDN训练调用墙钟：838.398s /3600s，含数据检查/准备、训练、完整验证和保存；峰值allocated显存 0.600GiB。
本轮两阶段只读累计：178.848s /1800s，CPU执行、optimizer updates=0。未额外进行三人数层级的分项延迟benchmark；导出墙钟不当作模型在线延迟。
历史BCEP原run训练耗时3600.153s，包含未进入best的第7轮619次更新；其第6轮完成累计3506.802s。历史成本与本轮新增MDN成本分列，不借旧预算续训。
实际训练代码：`4246a7df0e46a110eb0931327aec796183c9375e`；证据基线：`725f6a783308bfcd21e97df446005c820bc0e571`。模型与训练/导出源码冻结后未改变，report-only提交另列。
资产目录：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_postpilot_control_20261116_210845`。best/last、完整支持bank与哈希见ARTIFACTS.json。Git无权重、大bank或原始日志。

```bash
python scripts/bcep_postpilot_readonly.py --plan docs/research/bcep_v1/bcep_postpilot_control_20261116_210845/PLAN_AND_AUTHORIZATION.json --stage flow-diagnostics
python scripts/mdn_control_gate.py --config configs/research/bcep_postpilot_control_20261116_210845.yaml
python scripts/train_mdn_control.py --config configs/research/bcep_postpilot_control_20261116_210845.yaml
python scripts/bcep_postpilot_readonly.py --plan docs/research/bcep_v1/bcep_postpilot_control_20261116_210845/PLAN_AND_AUTHORIZATION.json --stage mdn-evaluation
```
上述为已实现入口的执行记录，使用项目`.conda/rsjg/bin/python`；run claim与累计预算会拒绝重复启动。CPU命令：`python -m pytest tests/test_bcep_core.py tests/test_bcep_prepilot.py tests/test_mdn_control.py -q`。

## 结论边界
只使用内部dev；原matched已接触此dev，未被调用为公平胜负对照。历史0.5811m不作为本次门槛。MDN是Bishop (1994)的标准混合密度控制，不是本项目新创新。
支持池/选择两部分误差并存，不能仅凭整池GT最近点诊断宣称可部署改善，也不能从一次短控制推广为所有MDN不如flow或flow必不可少。
本轮结束后没有BCEP续训、第二个MDN run、seed/fold队列、联合桥训练或后台监控。最终方向由实际四格结果决定，见本报告补充判读。

## 实际判读与唯一优先方向
coverage20下flow相对MDN平均minFDE减少0.017167m（以MDN为分母约3.03%），5/5推理seed同向；Recall@0.5高2.19个百分点。
但flow的P95比MDN高0.115806m，5/5 seed均是MDN较低。MDN的NLL较低而energy与平均minFDE较高，说明这些评价刻画不同方面，不能用单一NLL推出有限K覆盖更好。
单人组coverage minFDE仅相差约0.0037m，多人组约0.0208m；这是本开发集分组观察，不能据此认定某个网络模块的因果贡献。
实际早停和选中更新数不同，加上只有一个训练seed，当前不能把这点差距当作flow容量优势的充分证据，也不能用MDN更短成本宣称等质量加速。
唯一优先方向：先将方法核心表述收敛为“可替换密度提供器上的有限预算覆盖选择”，把flow保留为候选实现、MDN8保留为简洁控制；暂不据此增加flow复杂度或启动联合桥。此处没有自动把MDN升级为主方案。
