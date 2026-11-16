# BCEP GPU验收与唯一有界Provider试训

GPU_GATE：**PASS**；TRAINING_STATE：**BUDGET_EXHAUSTED**；EVAL_COMPLETENESS：**完整320窗口×5推理seed**；SELECTOR_SCREEN：**PASS**；PROVIDER_REPLACEMENT：**NOT_ESTABLISHED**。

本轮仅训练独立BCEP Provider；未训练配对器、条件桥或完整EBJD。没有更改模型、目标、数据划分、选模与筛查规则；没有加载GDTS/旧EBJD权重初始化，没有第二次真实试训。

## GPU工程验收
单卡FP32。GPU gate墙钟 12.283s，加CPU回归共 16.298s，上限600s。25项原CPU测试完整保留。
GPU合成16次optimizer更新（连续8步与3+5），真实数据更新0。恢复逐位一致：True。
CPU/GPU固定 rtol=1e-4、atol=1e-5；GPU恢复固定 rtol=1e-6、atol=1e-7。报告同时保留近零量的最大相对差，不将其单独当成失败；判定采用预注册绝对＋相对容差。
真实无梯度接口预先按人数/索引选取 N=1,3,57，不依据未来误差筛选。IID20和coverage20确实共享一份M256支持；同一支持上的CPU/GPU选择结果一致。

## 唯一试训与选模
已完成epoch：6；optimizer updates：23359；当前/下一epoch：7；phase：train；未完成epoch的train cursor：619。
best selected epoch：6；final world NLL（agent micro）：3.421933929125468。只有完整320窗口验证才参与选模；未使用coverage精度选择checkpoint。
配置：seed3101；Adam lr1e-4、gamma1；lambda_aux0.25；每场景一次update；最多10epochs/37900updates/3600秒累计流程墙钟，patience3。历史默认仍禁用真实训练，仅本run配置获授权。

| epoch | train复合loss(场景平均) | train最终NLL(行人平均) | dev最终NLL(行人平均) | 累计流程秒 |
|---:|---:|---:|---:|---:|
| 1 | 1.739282 | 2.240727 | 3.507618 | 586.43 |
| 2 | 1.374868 | 2.005666 | 3.571757 | 1185.27 |
| 3 | 1.127023 | 1.848955 | 3.614913 | 1784.05 |
| 4 | 0.964574 | 1.743860 | 3.486718 | 2354.94 |
| 5 | 0.795155 | 1.649318 | 3.467967 | 2928.76 |
| 6 | 0.664150 | 1.580823 | 3.421934 | 3506.80 |

## 冻结best的内部开发集结果
单位：距离m；NLL为世界坐标密度的负对数值（nats），不是米制误差；Recall为0–1比例。均值±样本SD来自五个推理seed，不是五次独立训练。开发集320窗口/621次行人出现；N=1为132窗口，N≥2为188窗口。GMM/MDN、旧provider及外层测试未在此进行公平对照。

| 指标 | IID20 | coverage20 | coverage−IID |
|---|---:|---:|---:|
| minFDE | 0.644648 ± 0.005425 | 0.549166 ± 0.009747 | -0.095481 |
| recall_0.25 | 0.349114 ± 0.013871 | 0.361675 ± 0.020672 | +0.012560 |
| recall_0.5 | 0.678583 ± 0.019108 | 0.752657 ± 0.009292 | +0.074074 |
| recall_1.0 | 0.865378 ± 0.001440 | 0.903382 ± 0.003013 | +0.038003 |
| P50 | 0.344916 ± 0.013929 | 0.316540 ± 0.012600 | -0.028376 |
| P90 | 1.247080 ± 0.029318 | 0.959336 ± 0.034691 | -0.287744 |
| P95 | 2.218317 ± 0.104300 | 1.888678 ± 0.080734 | -0.329639 |

| 分组 | 方法 | minFDE | Recall@0.5 | P90 |
|---|---|---:|---:|---:|
| N_eq_1 | iid20 | 0.781607 | 0.604545 | 1.674800 |
| N_eq_1 | coverage20 | 0.630852 | 0.707576 | 1.093267 |
| N_ge_2 | iid20 | 0.607677 | 0.698569 | 1.137672 |
| N_ge_2 | coverage20 | 0.527116 | 0.764826 | 0.869782 |

冻结best的CPU final world NLL：3.421934；IID256 energy score五seed均值：1.043580。NLL不对coverage代表点定义；energy只来自合法IID256。
预注册筛查：{"status": "PASS", "checks": {"minFDE_improvement_at_least_0.01m": true, "Recall_0.5_not_lower": true, "P90_not_higher": true}, "scope": "same-checkpoint selector screen only; not provider replacement evidence"}
coverage的minFDE在5/5个推理seed改善；逐seed/分组数值见JSON和CSV。

## 学习证据与解释边界
按冻结runner的CPU构造、seed3101及train-only尺度严格重建初始模型；在同一完整dev、CPU精度下，初始final NLL=3.789218，best=3.421934，变化=-0.367284。该初始模型只作只读诊断，不伪装为训练checkpoint。
NLL变化只支持密度学习诊断，不能等同于覆盖或校准全面达标。上述候选绝对误差、尾部和Recall需单独看待。

## 成本与身份
gate：12.283s GPU进程墙钟（不是GPU active秒）；训练wrapper本次调用：3600.192s；runner累计（含验证/保存）：3600.153336802963s；只读CPU阶段：114.697s /1800s。三个预算不互相挪用。
训练峰值allocated显存：0.601GiB。FP32、TF32关闭、cuDNN benchmark关闭/确定性开启、math SDPA；具体环境见GPU_GATE_REPORT及执行身份。
同M256支持集的IID20/coverage20比较不是等推理成本比较；CPU导出时间不是GPU推理时间。
单个预定2人dev窗口的CPU总coverage延迟（编码＋M256生成＋选择）均值：0.031502s；仅三次只读测量，不能外推全数据吞吐。

执行源码提交：`a9fa4d4ba7c8673ab23125542d54ed3c645a0653`。分析基线：`46f51f46a9e37889333c955d0b4da895e4f6f0ed`。
资产目录：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets/experiments/bcep_gpu_gate_one_pilot_20261116_194040`。best/last、bank及SHA见ARTIFACTS.json；Git不包含权重或bank。
保护主工作区HEAD与status前后相同，旧五折selected checkpoint及配置哈希复核见PROTECTED_ASSETS_CHECK.json。没有删除或覆盖历史资产。

## 结论
选择器筛查仅检验这个冻结密度上M256预算选点的价值。本轮未证明独立provider替代旧provider、flow优于其他密度或完整EBJD优于原六层。旧matched见过当前内部dev，历史0.5811m/64.1%不作为本次门槛。
本run结束后不建立后续队列，不启动第二次训练、配对器或条件桥。

本次结论：PASS。coverage20的minFDE比IID20降低14.81%，Recall@0.5增加7.41个百分点；单人和多人分组的minFDE方向一致。
仍有24.73%的行人出现未被20个候选覆盖到0.5m内，P95为1.889m；这说明尾部误差仍未解决，不能只凭平均误差宣布提供器已达标。
训练NLL与dev NLL来自不同成员，而且训练统计是在不断更新的参数下累积，不能仅凭两列的差距判定过拟合。当前结果支持密度确实有所学习、同支持集预算选点有效；尚不支持跨fold泛化、独立训练种子稳定性或替代旧provider。
唯一建议：将当前冻结模型作为独立provider候选保留，后续先取得数据暴露严格可比的替代验证，再决定是否接入联合桥；本轮不执行该后续实验。

## 可复核入口
以下为本次执行记录；单次启动器会拒绝重复run，预算耗尽也不能重新计时。
```bash
python scripts/bcep_gpu_gate.py --plan docs/research/bcep_v1/bcep_gpu_gate_one_pilot_20261116_194040/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/bcep_gpu_gate_one_pilot_20261116_194040.yaml
python scripts/bcep_one_pilot.py --plan docs/research/bcep_v1/bcep_gpu_gate_one_pilot_20261116_194040/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/bcep_gpu_gate_one_pilot_20261116_194040.yaml
python scripts/bcep_pilot_readonly.py --plan docs/research/bcep_v1/bcep_gpu_gate_one_pilot_20261116_194040/AUTHORIZATION_AND_PILOT_PLAN.json --config configs/research/bcep_gpu_gate_one_pilot_20261116_194040.yaml
python scripts/summarize_bcep_pilot.py --plan docs/research/bcep_v1/bcep_gpu_gate_one_pilot_20261116_194040/AUTHORIZATION_AND_PILOT_PLAN.json
```
执行Python为项目`.conda/rsjg/bin/python`；CPU回归：`python -m pytest tests/test_bcep_core.py tests/test_bcep_prepilot.py -q`。GPU gate的精确参数见GPU_GATE_REPORT及授权计划，训练/导出命令亦保存在执行身份中。
