# NO_TOKEN 固定终点路径监督短试验

## 结论

- 配对与完整性：**PASS**；两臂各771 updates，固定epoch28，逐update输入/RNG/LR一致，rollout/projection为0。
- 候选标准：**FAIL**。
- 原合同：**FAIL**。
- 单训练seed3101；五个推理seed不是独立训练重复，`independent_test=false`。

| 输出 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| 父NO_TOKEN epoch25 Fixed-anchor | 0.4322 ± 0.0051 | 0.5789 ± 0.0092 | 0.4862 ± 0.0035 | 0.7077 ± 0.0081 |
| Control epoch28 Fixed-anchor | 0.4118 ± 0.0021 | 0.5789 ± 0.0092 | 0.4650 ± 0.0038 | 0.7096 ± 0.0086 |
| Candidate epoch28 Fixed-anchor | 0.4118 ± 0.0021 | 0.5789 ± 0.0092 | 0.4650 ± 0.0038 | 0.7096 ± 0.0086 |
| matched GDTS | 0.4158 ± 0.0052 | 0.5771 ± 0.0101 | 0.4927 ± 0.0048 | 0.7492 ± 0.0086 |

Candidate−Control：minADE +0.000029, minFDE +0.000000, JADE +0.000014, JFDE +0.000000。逐seed改善：minADE 0/5, minFDE 0/5, JADE 0/5, JFDE 0/5。

Candidate−父模型：minADE -0.020398, minFDE +0.000000, JADE -0.021258, JFDE +0.001906。Candidate−GDTS：minADE -0.003935, minFDE +0.001809, JADE -0.027752, JFDE -0.039587。

ρ=0 minFDE最大浮点差 `4.912e-10`，不解释为训练收益。GPU总墙钟 `5496.5` 秒，峰值 `3.28` GiB；其中正式训练/验证 `4649.6` 秒、五种子评估 `825.8` 秒、预检 `21.1` 秒。

唯一下一步：停止该λ=0.10路径监督分支，不调权重、不延长训练。
