# 实现映射

以下路径相对于仓库根目录。

| 模块 | 输入 → 输出 | 训练和冻结 |
|---|---|---|
| `bcep_formal/optimizations.py::provider_loss_batched`、`bcep_v2/stages.py::density_callbacks` | 单窗历史、地图、有效 mask、训练标签 → h12 NLL + 0.25×mean(h4,h8 NLL) | 每四场景按行人实例加权，训练原 BCEP |
| `bcep_v2/provider.py` | 历史、观测图、身份、seed → 256 个样本 → 20 个中心 → 20 个修正候选 | B 阶段只训练候选头；缓存导出全程冻结 |
| `bcep_v2/stages.py::augment` | 同场景坐标、地图、候选 → 增强后的数据、变换和场景身份 | 训练时重新编码上下文；验证使用规范坐标 |
| `bcep_v2/bank.py` | 冻结模型、成员、目的、seed、视图、几何 → 独立版本的候选缓存及逐窗收据 | 生成接口只读观测；下游按成员、顺序及 SHA 校验 |
| `bcep_v2/joint.py` | 全局打分 → 初次硬分配 → 邻居兼容性 → 最终分配 | 训练时梯度经过两处分配；正式推理只输出硬结果 |
| `bcep_v2/objective.py`、`bcep_v2/stages.py::joint_callbacks` | G、观测、训练标签 → 配对损失及六层桥损失 | 阶段二训练编码器、配对与桥。Provider 通过冻结 bank 提供候选 |
| `bcep_v2/state.py`、`workflow.py`（同包） | 解析配置、数据身份、预算 → 阶段断点、验证排名、冻结收据 | 只恢复同一身份完整状态。验证最优值相同则优先较早轮次 |
| `bcep_v2/evaluation.py`、`bcep_joint/inference.py` | 同折所选模型、逐 seed 的 G → 原顺序或重配后的 20 条轨迹 | 先冻结五种子观测推理输出，再读取标签计算 G/H 及打乱配对对照 |
| `tests/test_bcep_integrated_v2.py`、`bcep_v2/check_interfaces.py` | 合成 fixture / 至多两个真实观测窗口 → 数学、软件及接口验收 | 已有模型只作 strict-load 迁移检查。未来正式训练独立初始化 |

[新增参数键、计数与检查结果](TEST_RESULTS.json)
