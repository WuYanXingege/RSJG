# 代码与数值审计记录

本次新增代码仅在 `scripts/joint_score_audit/` 和一份原型测试文件中；`bcep/ebjd/bcep_v2/bcep_v3/bcep_joint` 正式网络与控制器相对基点无修改。没有新增正式训练入口。

## 原生路径核对

1. 旧 v2 必须导入 468c5edd 原工作区；加载前校验 HEAD 与 `joint.py/objective.py/unified.py/runtime.py` 文件字节。三个 checkpoint strict load，`requires_grad` 名称集合逐项对照 checkpoint 的 trainable 清单。
2. 正式 `FormalInputs` 核对原划分、train/dev 文件哈希、冻结 Provider density19/head9、bank seed2035/K20/单位/成员顺序及队列状态。真实前向只读取 selection_dev；训练文件用于身份校验，未训练、未读取 heldout/test 样本。
3. `ebjd.unified.LearnableJointCouplingHead` 的候选 LayerNorm 与 `scene_cross_attention` 输出用只读 hook 采集。v3 缺失特征采集在 score MLP 前中止，完全不重复已有评分、Hungarian 或 DDIM。旧 v2 缺失评分经原生两次 assignment 采集，590/590 窗初配与终配均复现历史排列。
4. `bcep_v2.joint.assignment` / `bcep_v3.coupling.assignment` 仍调用原 `hard_assignment_indices`。硬解为每人20候选双射。原生 `log_sinkhorn` 为 FP32、20 次行列 logsumexp、温度1；单人用 identity。
5. v3 旧反馈和 HCRA 都以初次 initial_ST 为条件；HCRA 不是在旧反馈的 B 分支重匹配之后才计算。本次梯度图保持该真实路径。
6. `pair_loss` 使用固定终点的世界平均欧氏误差，softmin 温度0.1 m。`final_coordinate` 的反向是 `hard_goals + soft@G − detach(soft@G)`；未替换成 candidate-cost 或 shared-world 损失。
7. pair loss 通过最终 soft→score→旧反馈/HCRA/候选特征/共享编码器传播，另经 initial_ST→初配 Sinkhorn→S0 返回；它不通过轨迹 denoiser/GRG。模型的其他训练损失不在本次受控梯度中。

## 可辨识性与数值计算

- 原始评分均保持 FP32，不改变原求解输入；分析双重去均值、目标求和、SVD 使用 FP64。
- 有效矩阵已知存在常数向量零空间。直接对大行列偏置相消后的矩阵按极小 FP64 相对阈值计算 rank，会产生虚假的第20维。本次用 Helmert 零和基 Q∈R²⁰ˣ¹⁹ 分解 `Q.T @ center(S) @ Q`，补回数学零奇异值。只修正统计方法，不改评分或硬解。相应回归测试通过。
- 次优完整分配采用禁止每条已选边的精确枚举（20次20×20 Hungarian），不是逐行差或随机采样；v3 复用前次同算法结果。
- 有效数值秩、稳定秩、熵秩、前1/2/5能量、95%能量维数、FP32舍入尺度分别输出，避免把谱能量集中误写成精确秩1。
- 重编号检查使用固定行反转、列循环移位，映射回原编号比较。v3 复用前次结果；不增加随机噪声解除并列。
- 所有连续主统计先在窗口内平均行人，再等权平均窗口；并列计数、gap中位数和分布表明确以行人矩阵为单位。不能将全体/多人/N≥9或两种分母互换。

## 受控梯度与技术修正

`gradients.py` 每模型使用36个预先规定的 dev 窗；直接 `torch.autograd.grad` 获取 VJP，不调用 `loss.backward()` 累积参数 `.grad`，不构造优化器。

第一次尝试 eval 模式反传遇到 `cudnn RNN backward can only be called in training mode`，没有任何更新。改用原 train 模式建立原生图；逐模块确认 dropout=0、无 BatchNorm，参数与缓冲哈希前后不变。108个最终样本 S0/最终评分、初配/终配和缓存逐值相同，因此这次训练模式修正没有改变被审计的评分或硬选择。

初配间接梯度很小，两个FP32总梯度相减会相消。首版差值记录保留在主机 `gradient_v1_subtraction/`，最终主表改为：原图测得 initial_ST 的伴随梯度，输入原初配 Sinkhorn 独立 VJP。仍保留差值和链式一致性残差，给出相对直接梯度和相对间接梯度的两种尺度；没有用不稳定相减结果证明断路。为此重跑相同冻结样本，未改变模型或数据、未训练。

所有最终参数梯度有限；多人相关模块非零，单人旁路为零；小范数只报告，不直接判失败。该梯度仅为 pair loss 的单窗局部响应，不能代表 AdamW 的实际参数更新、完整多损失 minibatch、训练增广或离散 JFDE 方向。

## 优化器禁令与资产保全

`common.setup` 禁止 `torch.optim.Optimizer` 构造和已导入原生 optimizer 类的 `.step`；任何调用立即抛错并计数。采集、梯度、原型收据均为构造0/step0；新代码没有任何 optimizer 调用，也没有训练控制器或队列启动命令。CPU原型没有可训练参数，测试只验证确定性代数。

前后全量核对 v2、v3、candidate_cost、sharedworld 和上一轮评分回放的文件清单及 SHA256。42,650 文件全数一致，新增审计位于独立目录。最后状态分别为 PAUSED_RESUMABLE、COMPLETE、SESSION_BUDGET_PAUSED、COMPLETE；v3历史30个完整epoch/5040更新未变。不将已完成的运行误登记为暂停。

GitHub交付仅脚本、CSV、配置摘要、数学设计和报告。全部 checkpoint、bank、原始轨迹、原始特征/评分数组留在训练主机。

## 原型验收边界

10项CPU单元测试：替换代价与暴力重算一致、低秩全局公式等价、双射/候选边缘保持、顺序公共能量不升、非行列可加交互、携带incumbent的候选重编号、统一world重编号、严格并列弃权、同时邻居修改冲突反例、单人旁路、零空间谱计算。其中双射与单调性共用一项测试；完整列表与运行输出以 `PROTOTYPE_TESTS.txt` 为准。

57人、20候选、1596条无向边的CPU检查使用固定随机代数张量，不使用真实样本拟合。正式 φ/ρ、可微ST sweep、GPU反传性能与噪声键接口尚未实现；报告没有把这些未完成项称为正式网络验收通过。
