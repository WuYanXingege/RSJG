# 当前与 proposed 逐层架构

源码锚点均相对 79ea3fa5e94afe6818f8c8299b89aba1d677589e。[完整checkpoint每层shape/计数](STATIC_CHECKS.json)与RESULTS.current_architectures互补：state tensor并非全是参与forward的参数，下面区分注册量与有梯度量。没有执行任何网络来补形状。

## 0. 符号、坐标与共同约定

B=packed场景数，N=agent总数，E=同scene规范无向边数，Tobs=8/Tpred=12，dt=0.4s。X_world[N,8,2]，G_world[N,Kg,2]，Kg=21，P=20，Ky=20，M=4，R=8，dh=128。strict_no_z下Z不存在，不是Z=4。实际packed为[N,...]+scene_index[N]/scene_ptr，不是补零稠密[B,N,...]；每scene单独生成图/归约，禁止跨scene边。

world几何是m/m·s⁻¹；GDTS内部为down_factor=8的map coordinates，速度是每帧map displacement，不能直接称其m/s。prepare_inputs(model.py:1380)顺序：[相对末观测位置2、dt=1差分速度2、加速度2、绝对map位置2]。LSTM用[位置−该goal2、前述相对位置/速度/加速度6]。diffusion输入输出在map速度空间；先累积积分、乘8、scene坐标变换，再取未来12帧评测。

H/W随场景及预处理而变；已存window13 tensor_image为[6,60,80]，不是所有数据固定60×80。U-Net每个encoder block后maxpool，返回pool前skip；最后一次pool结果不用。空间层级为H,W → floor(H/2),floor(W/2) → … → floor(H/16),floor(W/16)，decoder双线性×2，必要CenterCrop skip，末尾插值回H,W。不能给整个数据集虚构统一分辨率。

所有Linear默认有bias，LN默认affine两向量、eps=1e-5；GRU/LSTM单层单向且有两套bias；未另列激活的输出层为linear。train/eval只有明确dropout分支差异；frozen参数仍须eval，不能仅requires_grad=False。mask非finite/跨scene一律应拒绝；canonical每人21个有效candidate。当前soft-target helper全false有candidate0 fallback，但exact P20要求至少20个有效候选：不能把fallback当成全pipeline合法合同。

## A. GDTS independent 基座（代码有效，非JDV2）

来源：model.py构造/1872 _legacy_encode/3778 ts_sample；model_utils/U_net_CNN.py:6,29,46,71,103,121；hist_traj_rnn_encoder.py:28；diffusion.py:84；common.py:51。

| 顺序/层 | 精确网络、输入→输出 | 参数 | 激活/norm/residual；冻结与梯度 |
|---|---|---:|---|
| Goal encoder e0 | Conv2d14→32 k3 s1 p1 d1；Conv32→32同规格；[N,14,H,W]→[N,32,H,W] | 13312 | 各ReLU，无BN；每块后MaxPool2 s2 |
| e1 | Conv32→32×2 | 18496 | 同上，H/2 |
| e2 | Conv32→64；64→64 | 55424 | 同上，H/4 |
| e3/e4 | 每块Conv64→64×2 | 各73856 | 同上，H/8,H/16 |
| d0/d1 | 各Upsample bilinear×2 align_corners=False+Conv64→64 k3；concat skip64=128；Conv128→64→64 k3 | 各147648 | 两个double-conv ReLU；没有激活在upconv后 |
| d2 | up Conv64→32；concat32=64；Conv64→32→32 | 46176 | 同上 |
| d3 | up Conv32→32；concat32=64；Conv64→32→32 | 36960 | 同上 |
| heatmap head | Conv32→12 k1 s1 p0；[N,32,H,W]→[N,12,H,W] | 396 | raw logits；endpoint另sigmoid；masked BCEWithLogits训练 |
| history | LSTM input8 hidden256 layers1 batch_first；[N,8,8]→last[N,256] | 272384 | internal sigmoid/tanh；encode_hist F.dropout(training=True)，但实际调用keep_prob=1所以无dropout |
| context-conditioned input | ConcatSquash 2→512，context259=256+[β,sinβ,cosβ]3 | 267264 | (Wx+b)⊙sigmoid(Wg c+bg)+Wb c，无额外ReLU |
| time position | sinusoidal512 max_len24；dropout0.1 | 0；12288 buffer值 | [T,N,512]，只沿时间位置；eval禁用dropout |
| temporal Transformer | 2层，每层d512/head4/head_dim128/FF1024；QKV512→1536，O512→512，FF512→1024→512 | 各2102784 | ReLU；post-LN，两残差，dropout0.1；attention沿T=12，不沿N |
| output projections | ConcatSquash512→256、256→128、128→2，context均259 | 264192 / 99328 /1296 | 每层同三Linear与gate公式，无其他非线性 |
| 注册但不调用 | diffnet.layer：额外TransformerEncoderLayer模板，TransformerEncoder深拷贝它 | 2102784 | state有权重，forward调用encoder克隆层，不调用模板；不能计作第三有效层 |
| schedule/sampling | DDPM100线性β1e−4→.05；树trunk100…31；branches30,25,20,15,10,5 | 0 | ε预测；70+20×6=190次denoiser；初始CPU randn一次+branch randn_like120次 |

Goal总613772；GDTS注册总7826588（不含positional/schedule buffers），实际forward可达参数5723804（7826588−2102784）。未通过额外forward计数。独立基座训练阶段Goal BCE及noise MSE可更新三模块；Stage-A/B/C全部冻结基座。H/W输入下trunk一次[N,12,2]；branch共享每agent的trunk latent，但不是所有agent使用同一噪声值。20输出不是21；额外argmax trunk goal仅内部condition。loss=diffusion MSE+20×Goal BCE是既有GDTS joint训练，不是本轮新目标。

## B. canonical Stage-A strict_no_z（当前冻结主线）

| 模块 / 文件:锚点 | 层序与输入输出 | 注册参数 | 梯度、精度、边界 |
|---|---|---:|---|
| radius+TTC图 / interaction_graph.py:139 | 最近位置/速度；距离≤6m 或 valid t*=−r·v/‖v‖²∈[0,8s]且最近距离≤6m；src<dst；edge14 | 0 | 仅观测、weight=1；不adaptive；不能看到图外未来碰撞 |
| Social输入 / social_encoder.py:133 | [N,8,4]=位置相对末帧2+速度2 →Linear4→128/ReLU | 640 | FP32 island；Stage-A train |
| Social时间 / :140 | 单层GRU128→128，last hidden[N,128] | 99072 | 观测8步、单向；无dropout |
| Social消息 / :16 | concat h_i128+h_j128+edge14=270；Linear270→128/ReLU→128 | 51200 | 两方向共享MLP，reverse14；FP32 |
| Social更新 / :27 | 256=[h128,weighted mean128]→128/ReLU→128；Dropout0；h+update→LN128；outputIdentity | 49408+256 | 两次index_add_，除sum(weight).clamp_min(1)；E0 aggregate=0仍执行update/LN，不是GDTS identity |
| base relation / relation_inference.py:12,103 | 270→128/ReLU→128/ReLU→4；反向logits均值→log_softmax_M | 51716 | [E,4]；共同学，无关系标签；FP32 |
| unary / unary_goal.py:13 | goal4=[g−last(2),distance,log_prior]→64/SiLU→64/LN；agent128→64/LN；concat context64,goal64,product64=192→64/SiLU→1 | 25409 | [N,21] score=logprior/1+residual；AMP residual可BF16，scoreFP32；末头初始化0，但checkpoint64weight+1bias均非零 |
| dynamic relation / dynamic_relation.py:21,92,149 | geom4→32/SiLU→16；query[4,16] dot/√16+bias[4]+base_logits→softmax_M；embedding[4,16]给选中关系 | 820 | FP32；[E,21,21,4]；其中embedding64在A endpoint loss无梯度，B冻结读取；实际有loss路径331781而非331845全部活跃 |
| energy candidate / joint_energy.py:27 | source [g_i−last_i(2),g_i−last_j(2)]4→64/SiLU→64；destination反向同网 | 4480 | **不是unary geometry4，不读取logprior** |
| energy pair / :30 | [h_i,h_j,edge]270→128/SiLU→64/LN；反向同网 | 43072 | AMP线性层；构造左/右context64 |
| energy fusion / :37 | pair64+candidate64+relationEmb[4,64]→LN64→SiLU→Linear64→8 | 256+128+520 | [E,4,21,8]左右；dot/sqrt8、混合与归约FP32；energy模块总48456 |
| future teacher / future_teacher.py:50 | GTdescriptor6→64/SiLU→64/SiLU→4，反向logits平均→log_softmax4 | 4868 | 仅train；[E,4]；post CE与KL双侧回传，无detach；没有scene posterior/scene GRU |
| sampler / joint_sampler.py:336 | 初始Gumbel-top20；两轮[S_i(P,K)=u−ΣC]→per-agent injective assignment | 0 | hard无梯度；FP32score→CPU arbitrary integers rectangular exact→GPU int64 IDs |
| frozen generator接口 / model.py:1059,1095 | IDs[N,20]→Gselected[N,20,2]；补argmaxtrunk成21；21次goal-relativeLSTM→context[21,N,1,256]→树Y[N,20,12,2] | 见A | inference无GTteacher；canonicalcorrector全零eval绕过literal base |

edge14顺序：[r_x,r_y,v_x,v_y,d,heading_cos, speed_j−speed_i, capped_TTC,valid_TTC,closest_dist,min_history_dist,var_history_dist,formation_x,formation_y]。反向符号翻转索引0,1,2,3,6,12,13，其余不变。具有不同物理量纲，并非全归一化特征。

dynamic geometry4：
[‖g_j−g_i‖/6，cos(g_i−x_i,g_j−x_j)，‖(g_j−g_i)−(x_j−x_i)‖/6，|‖g_i−x_i‖−‖g_j−x_j‖|/6]。静止cos=0，安全denominator；四维均swap不变。query缩放与softmax以源码为准；full和selected调用同geometry/net，数学上切片一致，不承诺不同运算shape的GPU逐位一致。

teacher descriptor6：[未来最小距离、argmin_step/(Tpred−1)、最终相对位置2、相对末观测队形变化2]；后四维reverse取负。输入描述符固定无可训练参数≠teacher网络冻结。single GT不能唯一命名关系模式。

Stage-A注册可训练331845，基座7826588冻结，corrector30851冻结；总注册8189284。A encoder/social与base relation FP32，unary/energy MLP遵循现有AMP，relation/energy乘积、logsumexp、loss、聚合FP32。所有candidate/cache几何detach；h、teacher logits不能缓存为不更新特征。

代码存在但strict-no-z未构造：scene_latent.py:35的SceneLatentPrior，attention Linear128→64/Tanh→1，经scene segment-softmax和weighted sum得128，与log(1+N)拼成129→128/SiLU→128/LN，再128→64/SiLU→4。该独立类静态50245参数（8321+33408+8516），**当前checkpoint对应参数为0**，不加进A总量。future_teacher的scene posterior支路也未启用；配置scene_modes=4不恢复它们。上表与JSON的所有层形状均以严格无z state为准。

## C. Stage-B V2-A（冻结corrector路线；不是canonical A）

来源dependency_corrector.py:25，component_residual_projection.py，model.py:3229训练/3778采样。基座+A均冻结，只corrector30851训练。nonzero corrector推理保留现有执行语义。

| 层 | 输入→输出/结构 | 参数 | 细节 |
|---|---|---:|---|
| state | directed[2E,12,21]，relative position2+relative velocity2+distance1+selected relation16 →21→64/SiLU→64/LN64 | 1408+4160+128 | noisy world velocity积分到位置，不是x0估计 |
| time | sinusoidal32→Linear32→64/SiLU→128 | 2112+8320 | split gamma64,beta64；末层零初始化；FiLM (1+γ)h+β |
| gate | 64→32/SiLU→1/Sigmoid | 2080+33 | ×edge_weight；时间逐点、边共享 |
| value | 64→64/SiLU→64 | 8320 | gate-weighted index_add_到receiver；除sumgate+1e−8 |
| output | aggregate[N,12,64]→64/SiLU→2 | 4160+130 | 末层weight/bias全零初始化，degree0强制0 |
| projection | 每connected component同一步 residual减agent均值 | 0 | FP32；不混scene；singleton0；不是每人时间积分为0 |
| combine | εbase+projected Δε →当前DDIM表达式 | 0 | nonzero active走correction；inactive literal base，shared noise |

训练：冻结StageA采20joint goals→GT端点scene-oracle挑一支（不是trajectory-oracle）→frozen context→从{30,25,20,15,10,5}为每scene抽t→noisy GT map velocity→base ε无梯度→corrector→投影→active noise MSE +0.05 sparse relative position MSE。time维无GRU/attention，但cumsum位置含此前噪声速度，不能称完全无时间信息。

全零头的Stage-A eval bypass仅在joint_goal+eval严格判断；Stage-B训练走_jdv2_dependency_losses，不能套用bypass，否则output头也无法学习。component_zero_mean约束的是residual，后续各agent base非线性、map变换和多步反馈下不能宣称最终world centroid严格不变；更不能声称端点/trajectory multiset保持。

## D. 已有 V3 / V4（非当前Stage-A；C proposed复用V4）

V3来源trajectory_aligner.py:105：无时间GRU；每步[位移2,差分速度2]4→128/SiLU→128；time mean+max concat256→128/SiLU/LN；agent dh128→128/SiLU/LN；edge context256+14+relation_dim，gate→128/SiLU→1；pair加geometry5，score→128/SiLU→128/SiLU→1；keep5→64/SiLU→1。legacy relation_dim由model构造的M决定，常用4；该配置下参数155331（静态公式；relation_dim=0时154307），并非本轮核验的V3 checkpoint总量。2轮alignment、8 Sinkhorn、温度.15、threshold.6、topk4是类默认；reference/anchor机制与V4全图不同，不把旧运行当canonical。无当前V3运行配置认证；复现需pin实际参数。

V4来源trajectory_pair_relation.py:267,348；trajectory_pair_energy.py:134；permutation_synchronizer.py:113；multiway_trajectory_coupler.py:62；multiway_coupling_loss.py:754,883。

| 模块 | 层序（C也完全复用） | 输入→输出 | 参数/梯度 |
|---|---|---|---:|
| path step | Linear7→128/SiLU→128/SiLU | [N,20,12,7]→[N×20,12,128] | 1024+16512 |
| temporal | 单层单向GRU128→128，last；LN128→Linear128→128→LN128 | →[N,20,128] | 99072+256+16512+256；encoder总133632 |
| relation | concat path_i128,path_j128,edge14,geometry12=282→128/SiLU→128/SiLU→4，双向均值 | [E,20,20,4] posterior softmax(logq0+residual) | 53252；Stage0 q0均匀，不冻随机prior头 |
| energy candidate | 128→128/SiLU/LN128 | [N,20,128] | 16768 |
| energy pair | path候选mean_i128+mean_j128+edge14=270→128/SiLU→128/LN | [E,128]左右 | 51456 |
| factors | context128+candidate128+relationEmb[4,128]→SiLU→128→8 | [E,4,20,8] | 512+1032；mode cost负dot/√8 |
| pair gate | concat128+128+14+12+4=286→128/SiLU→64/SiLU→1/Sigmoid | [E,20,20] | 45057；连续，不detach |
| dormant direct scorer | 282→128/SiLU→128/SiLU→4 | 不用于lowrank forward | 53252仍注册 |
| dormant edge gate | 274→128/SiLU→1/Sigmoid | 不用于pair gate forward | 35329仍注册 |
| synchronization | 4轮消息C_ij P_j及Cᵀ_ij P_i，weighted-degree归一化；αmsg1,αprev.1,αidentity.1；8 log-Sinkhorn，τ.2；anchor固定I | P[N,20candidate,20slot] | 0；soft可微，有限Sinkhorn非精确DS |
| confidence | gain,entropy,gate,log(size),log(dispersion),density,cycle error共7→32/SiLU→1/Sigmoid | component scalar | 289；末weight0/bias logit(.1) |
| hard | 每agent CPU Hungarian投影；component confidence<.6 identity | perm[N,slot]=candidate；gather Y | 0；无梯度；旧实现SciPy缺失可显式greedy fallback，正式比较须拒绝fallback |

V4注册390579；active301998，剩88581为两备用头，不因optimizer含它们便称有梯度。frozen GDTS7826588；Stage0 variant不需要旧social/relation参数参与。全部路径无dropout；relation/gate小std1e−3输出初始化，relation embedding std=128⁻½。C正式设计FP32 coupling；基座维持认证dtype。既有历史运行dtype从其resolved args决定，不冒称本轮核验所有V4 checkpoint。

geometry12逐项：[end distance,min_t distance,mean_t distance,formation change x/y,net displacement cosine,mean velocity cosine,mean relative velocity norm, (argmin+1)/12,final heading cosine,mean closing rate,end distance−observed distance]。位置m、速度m/s、time无量纲；formation2反向取负，其余swap不变；无距离就一定有害的硬标签。

训练沿soft P计算候选ADE+FDE的expected slot cost，非坐标插值。损失1alignment+.5pair-score+.5pair-assignment+2no-harm+.02entropy+.02relation-prior KL；gate-reg0。GT pair target来自个体误差+relative path+relative endpoint，detach后softmax负cost。硬permutation不给梯度；soft-hard差不受这套loss严格控制。packed版本按scene平衡，不同于对整个packed E直接平均。

## Proposed A / B / C 实施契约

| 路线 | 逐层继承与唯一变化 | 参数/初始化/load | pipeline与边界 |
|---|---|---|---|
| A | 上表B全部层、上表A全部frozen层、C corrector frozen零；无层删除/新增；loss mean-energy改S4 MC conditional | 注册train331845/active331781；新增0；old state strict load可行，optimizer/目标元数据不得假装同义resume | X→h→u/relation/E→两套S4局部CE+原KL训练；eval完全同canonical；仍非exact Gibbs |
| B | A的现有网络表，但**loss先用旧mean-energy**；dynamic第一Linear6→32替换4→32，其余32→16/M4/rank8不变 | +64=331909注册train；active331845；旧strict load失败，显式迁移拷贝4旧列、新2列zero，新commit/schema，随后重训 | geometry原4+直线closest distance/6+time/4.8；其余图、teacher、sampler、diffusion不改 |
| C | 上表D的V4逐层配置，Stage0 frozen base；不用A goal scorer/corrector/scene latent | active301998，注册390579；相对已有V4新增0；strict-load对应V4可行，不能strict-load JDV2 A | frozen fullY[N,20,12,2]→path GRU→relation/energy/gate→softsync/hardperm→gather；精确保有限marginal，不能生新坐标 |

B新增两维严格定义：H=12×.4=4.8s，r=x_j−x_i，v=((g_j−x_j)−(g_i−x_i))/H；a=‖v‖²；若a≤1e−8(m/s)²取t*=0，否则clip(−r·v/a,0,H)；d*=‖r+t*v‖；追加[d*/6,t*/H]。交换方向r,v同取负，d*,t*不变。有限检查先行；非finite拒绝，不填NaN。不是实际diffusion路径最近距离，缺失candidate沿mask排除。geometry由已有向量计算，新增的是给relation分支的归纳偏置，不是全模型新增不可恢复信息。

三路线默认宽度/时间/图预算固定，不搜索M/R/hidden/round。A/B fresh StageA从同源冻GDTS、同初始化种子训练；不能为了MC效果选更好的旧checkpoint。C同源fullbank缓存，相对raw逐样本认证。旧bank用于历史回归/配对，不能替代新clean训练输出。

## 复杂度与单GPU内存计划（未测）

A社会时间O(N Tobs dh²)，graph消息O(E dh²)；energy factors O(E M Kg R)，full pair O(E M Kg² R)；full relation有E Kg² M。edge_chunk256时一个FP32 relation张量1,806,336 bytes≈1.72MiB；同时存在logits/prob/E/autograd时是数倍，不能只报一份。MC增加[S,N,Kg]及chunk[S,E,Kg]，S4、不复制fullpair4次；批N128、E512时[S,N,Kg]43,008bytes。每个chunk保留autograd到backward，edge chunk不自动保证峰值随chunk而非总E固定。

A推断两轮O(E P Kg M R)+最多2Nactive个CPU矩形assignment，算术复杂度约O(N P² Kg)，任意精度整数位长影响常数；不枚举Kg^N。所需CPU/GPU传输每agent P×Kg。tree190次有效denoiser，跨P共享trunk；base attention约O(N Tpred² d)而非N²。

C path O(N Ky Tpred dh²)，pair MLP O(E Ky² dh²)，sync四轮O(E Ky³)+8次O(N Ky²)Sinkhorn；CPU Hungarian O(N Ky³)。E256/Ky20/hidden128单个pair activation52,428,800bytes=50MiB，双向多层autograd可能数百MiB到数GiB；不保证现有V4天然edge-chunk。C bank自身N128×20×12×2×4=245760bytes，远小于pair hidden。

设计A/B为N≤128/E≤512/B=1、累积8窗（C≤4/N64/E256），不截断一个scene来凑预算；超限scene单独whole-scene batch，若仍OOM就停止并记录未覆盖，不静默丢agent/edge。预计A/B候选缓存训练<8GiB、C<12GiB只属保守规划假设，不是测量；16GiB GPU预留4GiB，未来获授权后先做单batch峰值gate。基座训练预算另列，绝不能把缓存上便宜的A训练等同完整clean协议成本。

E0/N1：A保留unary/topP，refinement identity；不等于independentGDTS。C强制identity，不调用无边学习bias改变坐标。mixed孤立agent不参与correction/重排；disconnected图没有跨component消息，但整个scene的共同slot编号会影响scene JADE，数学图分布因子化不等于固定RNG/有限slot严格独立。agent relabeling要同步graph、stable RNG keys与payload；不能从网络共享权重推导任意重新编号后的逐位输出相同。
