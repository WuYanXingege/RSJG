# HOTEL Future Social Attention gamma=0.5 冻结推理

状态：**COMPLETE**。唯一新增变量为六个 FutureBlock 内部固定 `gamma=0.5`：`A_self + 0.5 × (A_full − A_self)`。使用原 HOTEL epoch45 checkpoint、Raw 输出和 test manifest；FULL 与 matched GDTS 直接复用已绑定五种子结果。

| 五推理种子均值 ± 样本标准差 | minADE ↓ | minFDE ↓ | JADE ↓ | JFDE ↓ |
|---|---:|---:|---:|---:|
| FULL | 0.138007 ± 0.001258 | 0.202037 ± 0.002553 | 0.168566 ± 0.001605 | 0.276314 ± 0.003906 |
| HALF | 0.138828 ± 0.002388 | 0.206117 ± 0.003603 | 0.177058 ± 0.001861 | 0.296070 ± 0.004233 |
| matched GDTS | 0.134384 ± 0.001351 | 0.191589 ± 0.002872 | 0.208469 ± 0.002341 | 0.357055 ± 0.004002 |

## HALF − FULL 逐 seed 差值

负值表示 HALF 更低。

| seed | minADE | minFDE | JADE | JFDE |
|---:|---:|---:|---:|---:|
| 2035 | -0.001040 | +0.001796 | +0.008212 | +0.017701 |
| 2036 | +0.002798 | +0.011035 | +0.008496 | +0.023208 |
| 2037 | +0.002352 | +0.003545 | +0.009587 | +0.021829 |
| 2038 | +0.000132 | +0.001368 | +0.007746 | +0.016745 |
| 2039 | -0.000138 | +0.002660 | +0.008418 | +0.019293 |

HALF 优于 FULL 的 seed 数：`{'minADE': 2, 'minFDE': 0, 'JADE': 0, 'JFDE': 0}`。

跨折方向：**NOT_SUPPORTED**。未满足项：`['half_minADE_mean_below_full', 'half_minFDE_mean_below_full', 'minADE_at_least_4_of_5', 'minFDE_at_least_4_of_5']`。

原 HOTEL 性能合同（相对 matched GDTS 四项均不退化，零容差）：**FAIL**。未满足项：`['minADE', 'minFDE']`。

本轮新增 GPU 墙钟：**792.28 / 1800 秒**。

证据边界：这是单训练 checkpoint 的五次推理随机重复，且 HOTEL validation/test 为镜像窗口；不是五训练种子、统计显著性检验或重新训练的结构消融。原 epoch45 模型及正式结果保持不变。
