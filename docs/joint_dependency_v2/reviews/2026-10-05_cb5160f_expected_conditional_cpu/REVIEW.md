# Expected conditional loss：纯实现与CPU验收

日期：2026-10-05（Asia/Shanghai）。输入/设计commit：cb5160fa8783cd7c90f0f0453809bf3e84665ed6；输入src tree：a2db08541129ba93e16d288867e7ec4fbd8e8c95。分支research/joint-dependency-v2-clean。开始时无tracked用户改动；原有未跟踪data/outputs保留，未读取其中bank/payload/checkpoint。

**状态：CPU_LOSS_IMPLEMENTATION_CERTIFIED_NOT_INTEGRATED。最终92/92 CPU测试通过，其中78项新增、14项已有纯loss/日程回归。新增参数0，旧函数正文与模型默认路径未改。可以讨论下一阶段授权，但不自动接线或训练。**

## 1. 实施范围

新增src/joint_goal_loss.py的expected_conditional_composite，外部传入固定z，按scene等权计算expected-hard-neighbor conditional CE；不抽随机数，不实例化网络。新增tests/test_jdv2_expected_conditional.py及本目录验收脚本/记录。

严格遵从本轮收窄授权：**没有修改model.py、parser.py、trainer.py，也没有新增开关**。前一设计中提到的接线、RNG checkpoint/atomic-save、clean-base及204 GPU-hours计划全部未执行。CPU-only接口不是已部署的主线A。

函数合同见[INPUT_CONTRACT](INPUT_CONTRACT.md)，详细结果见[RESULTS](RESULTS.json)、[最终逐测试记录](CPU_RUN_2.json)、[首次记录](CPU_RUN_1.json)、[源码隔离核验](SOURCE_SCOPE.json)。完整固定命令见[COMMANDS](COMMANDS.md)。

## 2. 验收证据

| 项目 | 实际结果 / 限定 |
|---|---|
| 非对称方向、chunk、重复receiver | source列切片/destination行切片与独立scalar循环值及梯度一致；degree8星图跨4块通过；孤立agent、单边/小链图覆盖 |
| 旧目标one-hot | S=1/4/7与原FP32 helper的u/C梯度和值一致；不声称旧helper有FP64精度 |
| 退化/scene归约 | E0、N1、K1、C全0、unequal scene sizes及不连续/负scene ID通过；packed等于逐scene等权均值 |
| 精确枚举 | N3/K2/E2的8状态、N3/K3链图27状态；非均匀q按product q加权；简单均匀平均负控明确不同；有理概率重复32配置精确平均通过 |
| 固定S4 MC | seed527015，1024重复×4draw；loss与14个u/C梯度分量均满足预登记5×SE+1e−10；未调seed/容差 |
| FP64梯度 | u及非对称cost通过gradcheck；逐项参考比较采用atol1e−10/rtol1e−8；已记录误差最大2.220446049250313e−16 |
| FP32值/梯度 | atol1e−6/rtol1e−5；已记录误差最大1.1920928955078125e−7 |
| M4 raw leaf链 | post→teacher/energy/unary，prior→deploy/energy/unary，原KL→teacher/deploy均存在；完整双CE+.07KL与独立参考吻合；没有网络实例化 |
| mask/重排/隔离 | 单一valid、部分mask、agent重编号连同cost转置、逐agent candidate重排、非连续tensor，值和梯度合同通过 |
| RNG/输入 | 两次相同z输出与梯度逐位相同；global CPU RNG与输入内容/版本不变；独立sampling helper不污染global RNG |
| 错误输入 | 最终56次有意非法调用均拒绝，包括缺/重/乱chunk、bad shape/dtype、trainable q、负ID、cross-scene/self/duplicate edges、空support、masked NaN/Inf、数值overflow |

MC loss exact=1.1511866389562833；样本均值=1.1484943456785737；绝对偏差0.00269229327770959，SE=0.003379341913059336，预声明界0.01689670966529668。这是**估计器检查**，不是JADE/JFDE、训练收益或泛化结果。每个梯度分量的均值/SE/偏差均完整保留，不只报通过的分量。

Jensen例中单节点mean-energy CE=2.1269280110429727，expected conditional CE=2.3556485542388774；当前surrogate仍合法。新loss保持normalizer在MC平均以内，不消除未来真实训练/部署邻居分布差异。

## 3. 执行账本

第一次88/88通过，存在一次“requires_grad tensor转标量”的测试断言warning；随后仅给断言加detach并增加4项边界验收。没有修改纯loss实现或统计阈值。第二次92/92通过、无warning。两次统计试验seed相同，所以不能把它们合称2048次独立重复。

两次pytest主体共6.156154秒（3.320835+2.835318），远低于CPU≤1小时；脚本启动/源码静态检查另计，未进行压力benchmark。总计180个test-call完成、0失败、0未知；最终独立测试案例数仍为92。

新loss总forward尝试2311，完成2200，预期非法拒绝111，未知0。成功gradient API请求2148（包含独立参考），loss输出backward hook访问2149（含gradcheck内部VJP）；二者口径不同，不伪装成真实训练steps。单次最终运行的各计数见JSON。

runner在执行测试前拦截nn.Module forward、CUDA初始化和torch.load；三者尝试均0。模型/full-A/encoder/message/sampler/diffusion/corrector、solver、GPU、训练、真实bank读取/重放、exact-margin审计均0。SDD未干预/恢复，Skills/Apps未改。

## 4. 原路径未变的证据

verify_scope.py逐一比较原23个函数/类定义的源码正文，全部逐字相同；旧alias、归约与dtype实现未改。src内新函数调用者为0，仅定义和export；model.py/parser.py/trainer.py/模型目录/配置/历史设计归档无diff。

准确表述是“**新增loss库代码，现有生产训练/推理语义未改**”，不是“生产源码没有变化”。checkpoint schema未新增state的判断来自新增纯函数/模块未变；没有执行真实checkpoint load或模型strict-load测试。

## 5. 下一阶段门槛

可以另行授权训练接线与独立RNG/checkpoint合同，但应先规定CUDA/AMP扩展验收、默认旧分支保持、端点q与relation teacher不同梯度边界、Cpost/Cprior共用z及成功step计数。CPU合成通过不替代生产训练图认证。

当前停在本实施单元。性能、learned pair energy因果贡献、新颖性和历史29IDs因果链均OPEN；旧bank的ΔJADE/ΔJFDE不改，不用本轮标量数学解释历史差距。不自动开始GPU、训练、geometry6、V4或其他诊断。
