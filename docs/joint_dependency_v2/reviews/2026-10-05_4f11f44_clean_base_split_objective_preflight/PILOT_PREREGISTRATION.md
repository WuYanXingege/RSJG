# 两臂试验预登记补充（无执行授权）

P2来源固定：HOTEL outer445 / uni_examples inner320 / train3484；recording身份待解，8文件不等于8独立样本。旧来源暴露不可撤销。
主线只改objective，不引入z/geometry6/V4/Stage-B，不detach teacher，不改变0.5post+0.5prior或beta·KL。

## L1：仅真实更新通路门槛

前置：合格base+新cache+P2loader/teacher权限+safe attempt cap；共有初态字节相同；全部身份hash锁定。当前全部未齐，TEMPLATE_BLOCKED。
seed3101，native Stage-A batch_size=1 scene（YAML legacy batch_size64不控制此分支），accumulation1，3484更新/完整epoch，因此H=min(500,3484)=500。
唯一固定order见BATCH_ORDER_SEED3101，A0/A1同前500窗；记录每步N/E、成员、曝光hash、attempt与completed计数。

参考训练长度1 native epoch，total_steps3484；attempt t=1..500前completed=t−1、progress=(t−1)/3484、beta=.1 min(progress/.2,1)。首beta0，末beta0.07161308840413318；成功结束progress500/3484=0.14351320321469574。详见L1_PROGRESS_TABLE的500行。
此短诊断不采用smoke .25，也不是300epoch或L2的warmup。
Adam lr1e−4，betas(.9,.999)，eps1e−8，weight_decay0，amsgradFalse，clip1；其他installed defaults见PILOT_PLAN（由已安装签名静态读取，未构造optimizer）。
ExponentialLR gamma=.995，只在完整epoch边界执行；H不足3484，因此scheduler steps0。

BF16策略一致；A1 loss有效FP32，backend cuda_fp32_bf16_v1，独立CPU target generator S4。A0无MC owner。
每臂GPU墙钟≤900s，总GPU阶段≤2700s，peak reserved≤10GiB；首次仅空闲GPU，无任务抢占。不改精度、不截断场景规避OOM。
任何nonfinite、skip、pending异常、加载身份不符、资源上限触发立即停止整个pair；不补抽以凑500成功。
不跑sampler/diffusion/完整轨迹validation、不解封outer标签。保存首尾分量、有限梯度、更新范数、frozen hash、实际步数与MC draws/pending。
没有validation分支/attempt cap的旧trainer不允许运行此计划；本轮未实施该patch。
L1通过只说明固定预算内update通路可运行，不证明JADE改善，不自动增加epochs/seeds。

## L2：后续独立授权的效能实验

三training seeds3101–3103，各自重新共同初始化、paired顺序；同一base/cache/预算。每arm最多40epochs，每epoch仅inner选择；patience5、min_delta .001m，primary JADE、JFDE然后earliest epoch tie-break，guardrails预先应用。当前Stage-A入口不完整支持此规则，需要adapter及合成CPU选择测试。
若不提前停止，native总更新139360/arm；6arm最多836160次。各seed的L2 progress以完整40epoch预算固定，不能沿用L1分母。选择/部署teacher诊断两臂必须同等禁用；prediction只依赖观测/map/共同噪声，GT仅在独立metric writer比较。
固定inference seeds2035–2039；inference seed不是训练重复。模型/analysis/config hash冻结后，单次正式outer5seed评测，不反馈调参。

沿用原预登记：
- JADE改善≥.01m且≥2%，至少2/3 training seed同方向；来源不足不报告confirmatory总体CI。
- minADE恶化≤.003m且≤1%；minFDE≤.005m且≤1%；JFDE不恶化>.01m。
- CRmean绝对增量≤.005；全agent20/20候选ID覆盖不减；finite/scene isolation通过。
- CR主阈值.2m，连续线段最近距离；.4m仅预声明敏感性。CR-JADE first-index exact tie并报告tie范围。
- 报告marginal/JADE/JFDE、relative path error、collision、diversity/duplicates、goal与第21候选排除变化、solver/forward成本。
- 未达门槛拒绝当前效能主张，回到A0；不加S/rank/epochs或换seed。

单outer source且重叠窗口很多；block分析仅作为依赖敏感性，不冒充独立source泛化。历史HOTEL暴露必须写入论文边界。

## Pair energy归因：只设计，不执行

同checkpoint、同G和实际common noise，对照 full C / 真正C=0 / 仅a+b。
有效cost C=−logsumexp(log p−E)，sampler logits由unary减cost；legacy energy_weight未被JDV2 sampler读取，设0不是energy-off。
对于有效K×K pair cost，均匀候选测度：mu=mean(C)，a=rowmean(C)−mu，b=colmean(C)，I=C−a−b；I行列均值为0。无效候选需固定product support，不能事后换测度。
full与I的gauge负控：每个节点unary减去其所有incident单边项（按edge朝向取a或b），使sum(u)−sum(C)保持相同。固定原Round0/共同noise/同tie路径；不补偿初始unary-only Round0会改变算法，不可声称纯gauge等价。
仅a+b意味着删除I、原unary不变；它检验交互增量，不是能随意混同的unary-only。
goal/sampler干预会改每人P20支持集，必须报告marginal与第21候选；只有固定完整trajectory bank内双射/独立置换严格保持每人的有限multiset。固定bank配对null优势不等于learned energy因果贡献。
本轮真实C/solver/exact-margin payload/bank生成均0；连合成分解实验也未新增。
