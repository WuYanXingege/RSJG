# 训练/推理伪代码与最小实施接口

以下是**设计文档，不是可执行生产补丁**。现状以79ea3fa为准；A保持模型层与部署不变，只替换mean-energy的两项CE。已有函数复用均给出具体来源，新增函数在此展开，不用“joint_model()”掩盖损失和推断。

## 1. 候选与共同forward（现状/推荐共用）

正式JDV2必须加载显式冻结cache；没有cache时报错，不能悄悄改成在线随机候选。下面build只描述既有离线来源，不是本轮授权build。model.py:789；sampling_2D_map.py:228：

~~~python
# 离线构建，基座eval/no_grad；以后固定candidate顺序及seed。
heatmap_logits = goal_UNet(cat(scene_raster_6, history_maps_8)) # [N,12,H,W]
endpoint_prob = sigmoid(heatmap_logits[:, -1:])               # [N,1,H,W]
# 原GDTS TTST：先多点采样后20簇代表+argmax，共21；不可换排序/实现。
G_map, candidate_prob = generate_goal_candidates(
    endpoint_prob, num_candidates=21, device=device, use_ttst=True)
G_world = scene.make_world_coord_torch(G_map * down_factor_8)
log_prior = log(candidate_prob.float().clamp_min(1e-8))
edges, edge14, weights = interaction_graph(X_world, scene_index)
# 缓存仅上述冻候选/观测几何/训练descriptor；不存h/unary/关系网络输出。
# manifest: source commit, base SHA, split SHA/raw files, K/dt/units/order/dtype.
~~~

训练现状还会在encode中重新算Goal U-Net logits、GT endpoint的history context，虽然候选取cache且A loss不使用diffusion。这些冗余不在A目标补丁中顺便删除，以免引入第二变量。teacher descriptor允许来自训练cache；部署loader不开放未来监督字段。

~~~python
def deployed_layers(model, X, G, log_prior, edges, edge14, weights, scene):
    # [N,8,2], [N,K,2], [N,K], [2,E], [E,14], [E], [N]
    with autocast(device_type=X.device.type, enabled=False):
        h = model.social_encoder(X.float(), edges, edge14.float(),
                           weights.float(), scene_index=scene) # [N,128]
        _, b = model.relation_inference(h, edges, edge14.float(),
                                  hard=False, return_logits=True) # [E,4]
    # Unary implementation: goal4->64->64/LN;
    # agent128->64/LN; cat(context,goal,product)192->64->1.
    u = model.jdv2_unary(h, G, X[:, -1], log_prior)['score'] # [N,K]
    u = u.float() # existing loss/solver FP32 boundary
    return h, b, u

def cost_chunk(model, h, b, G, last, edge, e14, q_relation=None):
    # Actual modules from dynamic_relation.py and joint_energy.py.
    logp = model.jdv2_dynamic_relation.full_pair_relation(b, G, last, edge,
                                      mode_enabled=False)['log_prob'] # [Ec,K,K,4]
    f = model.jdv2_joint_energy.factors(h, G, last, edge, e14, mode_enabled=False)
    # L/R:[Ec,4,K,8]. Candidate features are goal relative to BOTH agents.
    with autocast(device_type=G.device.type, enabled=False):
        Em = -einsum('emkr,emlr->eklm', f['left_factor'].float(),
                      f['right_factor'].float()) / sqrt(8) # [Ec,K,K,4]
        Cprior = -logsumexp(logp.float() - Em, dim=-1)  # [Ec,K,K]
        Cpost = None if q_relation is None else -logsumexp(
            q_relation.float()[:, None, None, :] - Em, dim=-1)
    return Cprior, Cpost, logp
~~~

这里q_relation参数是teacher **log probability**，不是endpoint q，名字不意味着detach。所有energy mixing在disabled autocast。mask全真是canonical实际情况；泛化接口用valid mask，仅对valid矩阵做loss/solver，非法support拒绝。

## 2. 当前Stage-A training_step

~~~python
def scene_mean(per_agent, scene):
    return stack([per_agent[scene == b].mean()
                  for b in unique(scene)]).mean()

def CE(q, logits, scene):
    logp = log_softmax(logits.float(), dim=-1)  # candidate轴
    per_i = -where(q > 0, q * logp, zeros_like(q)).sum(-1)
    return scene_mean(per_i, scene)

def current_train(batch, model, optimizer, completed_steps, total_steps):
    G, logp0, edges, edge14, weights = checked_frozen_cache(batch)
    # checked_frozen_cache only validates manifest and returns the named tensors
    # in section1; no trainable cached h, no rebuilt candidates on mismatch.
    N, K = G.shape[:2]
    E = edges.shape[1]
    h, b, u = deployed_layers(model, batch.X, G, logp0, edges,
                             edge14, weights, batch.scene)
    q = softmax(-squared_norm(G - batch.Ystar[:, -1, None]) / (2*1.0**2),
                dim=-1).detach() # [N,K]; mask before softmax
    desc = future_pair_descriptor(batch.Ystar, batch.X[:, -1], edges)
    teacher_log = model.jdv2_future_teacher.relation_posterior(desc)['log_prob'] # [E,4]
    # No teacher_log.detach(). Descriptor has no trainable graph.
    loc_pr = zeros_like(u, dtype=float32)
    loc_po = zeros_like(u, dtype=float32)
    KLsum = u.float().sum() * 0
    for start, stop in edge_chunks(E, 256):
        ec = edges[:, start:stop]
        i, j = ec
        Cpr, Cpo, logpr = cost_chunk(model, h, b[start:stop], G, batch.X[:, -1],
                                      ec, edge14[start:stop], teacher_log[start:stop])
        for C, local in [(Cpr, loc_pr), (Cpo, loc_po)]:
            local.index_add_(0, i, einsum('ekl,el->ek', C, q[j]))
            local.index_add_(0, j, einsum('ekl,ek->el', C, q[i]))
        lq = log_softmax(teacher_log[start:stop].float(), -1)
        lp = log_softmax(logpr.float(), -1)
        kl = (lq[:,None,None,:].exp() * (lq[:,None,None,:]-lp)).sum(-1)
        KLsum = KLsum + einsum('ek,el,ekl->e', q[i], q[j], kl).sum()
    Lr = KLsum / E if E else KLsum
    beta = .1 * min((completed_steps / total_steps) / .2, 1)
    loss = .5*CE(q, u-loc_po, batch.scene) + .5*CE(q, u-loc_pr, batch.scene) + beta*Lr
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                    # no hard sampler / diffusion in this loss
    clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return loss
~~~

既有teacher/noisy训练图不涉及pseudo gradients穿过assignment。训练中的mask、dtype和scene reduction不是可以留给实现者猜的细节。

## 3. 推荐A的模块定义、forward与training_step

网络逐层见架构B表，直接持有相同Social/Unary/Dynamic/Energy/Teacher模块，state_dict不变。新增的是纯loss对象，无Parameter、无buffer。算法采样q是固定label分布，使用独立训练generator，避免污染部署RNG；目标样本按source/window/epoch/训练seed记录。

~~~python
class ExpectedConditionalComposite:
    # Proposed pure loss, no nn.Parameters, no learned embedding.
    def __init__(self, neighbor_draws=4):
        assert neighbor_draws == 4
        self.S = 4

    def draw_neighbors(self, q, generator):
        # One draw vector per sample, all nodes. Reuse across both objectives.
        # Sampling current model predictions would be a DIFFERENT objective.
        return multinomial(q.detach(), num_samples=self.S, replacement=True,
                           generator=generator).T.contiguous() # [S,N]

    def accumulate(self, C, edge, z, local):
        # C[Ec,K,K]; local[S,N,K]; z[S,N]; no copying C S times.
        K = C.shape[-1]
        i, j = edge
        for s in range(self.S):
            # source alternative k conditioned on destination's z_j.
            src_local = C.gather(2, z[s,j,None,None].expand(-1,K,1)).squeeze(2)
            # destination alternative l conditioned on source's z_i.
            dst_local = C.gather(1, z[s,i,None,None].expand(-1,1,K)).squeeze(1)
            local[s].index_add_(0, i, src_local)
            local[s].index_add_(0, j, dst_local)
        return local

    def forward(self, u, q, local, scene):
        # Candidate normalizer INSIDE Monte Carlo mean.
        return stack([CE(q, u-local[s], scene) for s in range(self.S)]).mean()

def proposed_A_train(batch, model, optimizer, completed_steps, total_steps,
                     train_generator):
    criterion = ExpectedConditionalComposite(neighbor_draws=4)
    G, logp0, edges, edge14, weights = checked_frozen_cache(batch)
    N, K = G.shape[:2]
    E = edges.shape[1]
    scene = batch.scene
    h, b, u = deployed_layers(model, batch.X, G, logp0, edges,
                              edge14, weights, scene)
    with autocast(device_type=G.device.type, enabled=False):
        q = softmax(-squared_norm(G.float()-batch.Ystar[:, -1, None].float())
                    / (2*1.0**2), dim=-1).detach()
    desc = future_pair_descriptor(batch.Ystar, batch.X[:, -1], edges)
    teacher_log = model.jdv2_future_teacher.relation_posterior(desc)['log_prob']
    z = criterion.draw_neighbors(q, train_generator)            # [4,N]
    loc_pr = zeros((4,N,K), device=u.device, dtype=float32)
    loc_po = zeros_like(loc_pr)
    KLsum = u.sum()*0
    for start, stop in edge_chunks(E, 256):
        ec = edges[:, start:stop]
        i, j = ec
        Cpr, Cpo, logpr = cost_chunk(model, h, b[start:stop], G,
            batch.X[:, -1], ec, edge14[start:stop], teacher_log[start:stop])
        with autocast(device_type=G.device.type, enabled=False):
            loc_pr = criterion.accumulate(Cpr, ec, z, loc_pr)
            loc_po = criterion.accumulate(Cpo, ec, z, loc_po)
            lq = log_softmax(teacher_log[start:stop].float(), -1)
            lp = log_softmax(logpr.float(), -1)
            kl = (lq[:,None,None,:].exp() * (lq[:,None,None,:]-lp)).sum(-1)
            KLsum = KLsum + einsum('ek,el,ekl->e', q[i], q[j], kl).sum()
    Lr = KLsum/E if E else KLsum
    beta = .1 * min((completed_steps/total_steps)/.2, 1)
    # After ALL chunks, not a separate softmax per chunk:
    loss = .5*criterion.forward(u,q,loc_po,scene) \
         + .5*criterion.forward(u,q,loc_pr,scene) + beta*Lr
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return {'loss': loss, 'relation_kl': Lr, 'neighbor_ids': z.detach()}
~~~

此wrapper复用第1–2节已展开函数；squared_norm表示最后坐标轴平方求和，edge_chunks表示range(0,E,256)及min(start+256,E)，不封装新网络。canonical cache返回全有效候选；若接口扩展mask，则q的softmax前无效项设−∞，CE logits同mask，support不足20拒绝。BF16外层autocast、成功step计数、8窗梯度累积和scheduler由既有trainer管理：上面展示一个optimizer update的数学图，累积实现需各micro-loss除8、只在末个micro-batch clip/step，不能每窗step再称等价。若q one-hot，任意S都应等于原CE；E0两项均退化unary CE。MC条件q fixed，无score-function或straight-through estimator；teacher继续通过Cpost/KL更新。此目标不是“采样部署sampler”的近似。

## 4. 当前及A的部署（完全相同）

~~~python
@no_grad()
def infer(batch_obs, model, sampler_generator, evaluation_seed, window_index):
    # Future fields unavailable. Read checked deployment cache (section1).
    G, logp0, edges, edge14, weights = checked_frozen_cache(batch_obs)
    X, scene = batch_obs.X, batch_obs.scene
    N, K = G.shape[:2]
    valid = ones((N,K), dtype=bool, device=G.device) # actual canonical cache
    h, b, u = deployed_layers(model, X, G, logp0, edges, edge14, weights, scene)
    expanded_mask = valid[:,None,:].expand(N,20,K)
    # Actual joint_sampler.py helper preserves finite masks and RNG arithmetic.
    # Mathematically topP(u - log(-log(U))); production helper is authoritative.
    ids = weighted_gumbel_top_p(u[:,None,:].expand(N,20,K), expanded_mask,
                                1.0, sampler_generator)
    src, dst = edges.long()
    degree = zeros(N, dtype=long, device=u.device)
    degree.index_add_(0, src, ones_like(src))
    degree.index_add_(0, dst, ones_like(dst))
    # Canonical temp=1. Persistent priorities generated once with ORIGINAL
    # eval seed/window/agent namespace; don't introduce fresh round Gumbel.
    for round_id in range(2 if edges.shape[1] else 0):
        old = ids
        local = zeros((N,20,K), dtype=float32, device=u.device)
        dynamic, energy = model.jdv2_dynamic_relation, model.jdv2_joint_energy
        r_src = dynamic.selected_neighbor_relation(b, G, X[:,-1], edges,
            old, None, 'source', dynamic_enabled=True, mode_enabled=False)
        r_dst = dynamic.selected_neighbor_relation(b, G, X[:,-1], edges,
            old, None, 'destination', dynamic_enabled=True, mode_enabled=False)
        c_src = energy.selected_effective_energy(h, G, X[:,-1], edges,
            edge14, old, None, r_src['log_prob'], 'source', mode_enabled=False)
        c_dst = energy.selected_effective_energy(h, G, X[:,-1], edges,
            edge14, old, None, r_dst['log_prob'], 'destination', mode_enabled=False)
        local.index_add_(0, src, c_src.float()) # C_ij(k,old[j,p])
        local.index_add_(0, dst, c_dst.float()) # C_ij(old[i,p],l)
        score = u[:,None,:] - local
        # Existing function loops active agents only; ONE rectangular arbitrary-
        # integer CPU Hungarian per active agent, preserving degree-zero ids.
        new = exact_persistent_refinement(score, expanded_mask, old, G, degree,
                         evaluation_seed=evaluation_seed, window_index=window_index)
        ids = new    # synchronous: no agent reads new neighbor within round
    selected = G.gather(1, ids[...,None].expand(-1,-1,2))
    # Append frozen heatmap argmax trunk endpoint, not a 21st evaluation output.
    goals_map_21 = world_to_map(concat(selected, trunk_goal[:,None], axis=1))
    contexts = []
    for p in range(21):
        # [goal-relative absolute map pos2, last-observation-relative pos2,
        #  map velocity2, map acceleration2] = 8 channels
        inp = cat(X_map - goals_map_21[:,p,None,:],
                  X_map - X_map[:,-1,None,:], V_map, A_map, dim=-1)
        contexts.append(frozen_LSTM_last_hidden(inp)) # [N,256], dropout_keep=1
    return literal_base_tree(contexts, scene, last_map)
~~~

工程必须复用现有selected函数，不切换full计算顺序来“优化”。exact_persistent_refinement定义于exact_lexicographic_assignment.py:358，被joint_sampler.py导入使用；内部调用同文件的solve_exact_persistent_tie；FP32分数精确整数化后按sum score、stay count、负goal位移平方、persistent power-of-two priority之和求lex最优。不退回浮点SciPy/greedy。无效/不足support报错；masked edges不参与；persistent keys沿用seed/window/agent_index，新增source标识属于外层manifest而非擅改原key。X_map/V_map/A_map/last_map由第0节架构的prepare_inputs得到，trunk_goal为冻结heatmap argmax（来自对应cache或原frozen encode）；world_to_map与LSTM接口复用model._jdv2_encode。伪代码中的literal_base_tree在下一块逐步展开，noise来源/形状不另改。

树采样方程展开（model.py:3778）：

~~~python
x = torch.randn([N,12,2]).to(device)  # actual initial CPU draw; fixed common payload
for t in range(100,30,-1):            # 70 calls, context=trunk
    eps = frozen_diffnet(x, beta=betas[t], context=contexts[20])
    x = (x - (1-alpha[t])/sqrt(1-abar[t])*eps)/sqrt(alpha[t])
middle = x
for p in range(20):
    x = middle
    for t, t_prev in [(30,25),(25,20),(20,15),(15,10),(10,5),(5,0)]:
        eps = frozen_diffnet(x, beta=betas[t], context=contexts[p])
        var = (1-abar[t_prev])/(1-abar[t])*(1-abar[t]/abar[t_prev]) # ETH eta=1
        xi = randn_like(x)  # still consumed at final step; no extra draw
        x = sqrt(abar[t_prev]/abar[t])*x \
          + (sqrt(1-abar[t_prev]-var)
             -sqrt(abar[t_prev]*(1-abar[t])/abar[t]))*eps \
          + sqrt(1-abar[t]/abar[t_prev])*xi  # actual simple_var ETH path
    Ymap[p] = last_map[:,None,:] + cumsum(x, dim=time)
Yworld = scene.make_world_coord_torch(Ymap * 8) # return future [N,P,12,2]
~~~

ε而非x0/v参数化；这里epsilon变量是预测噪声，velocity tensor才是被去噪对象。非ETH的simple_var/eta规则沿用源码，不能用这个ETH表达式改SDD。Stage-A严格零corrector直接调用literal base arithmetic；绝不加FP32零，训练不绕过zero head。

## 5. B、C的接口与合同

B仅替换dynamic geometry模块输入，公式/参数见架构；用6→32第一层新两列零初始化，shared geometry同时用于full/selected。migration必须保存新schema，旧config失败而非strict=False全局吞missing keys。

C沿用RelationAwareMultiwayCoupler.forward(Y,last,edges,edge14,weights,q0,scene,hard)。默认stage0 q0=uniform4、Ky20。soft P[candidate,slot]，初始化I；每轮：
message_i=Σ_j strength_ij (compat_ij @ P_j)/Σ_j strength_ij；
P_i=Sinkhorn((message_i+.1log P_i+.1I)/.2)，anchor I，4rounds；每次8交替行列logsumexp归一化。train confidence c使P_eff=cP+(1−c)I，eval按c<.6整component identity，不对soft mixture求projection。raw P CPU Hungarian得到perm[slot]=candidate，hard gather Y[:,perm]；无soft coordinate averaging。

GT target cost c_i(k)=ADE_i(k)+FDE_i(k)；edge oracle为(c_i(k)+c_j(l))/2加relative path/end误差（具体既有lambda均1，scene_balanced wrapper）；p_target=softmax_(k,l)(−cost/.5).detach()。Lalignment=normalized softmin_.5 of mean_iΣ_kP_i(k,p)c_i(k)；noharm=relu(softmin_aligned−softmin_identity+margin)，默认margin=0。pair-score/assignment KL=.5/.5，assignment distribution来自P_i P_jᵀ并归一化；entropy .02、relation KL .02。C保持这些既有函数及方向，不声称它们是normalized joint likelihood。

C proposed推理top-k=0，全图same scene；类默认topk4不可误用。硬合同用validate_bijection + inverse-gather字节等同保证多重集合，不能只用“集合中存在相等轨迹”处理重复候选。无边/singleton直接I；正常soft-hard gap仍必须报告。

## 6. Proposed YAML（不是生产配置；三路线无自动启动）

I=INHERITED_VERIFIED，D=DESIGN_DEFAULT，T=TUNE_ON_INTERNAL_VALIDATION。没有标T的值不搜索。

~~~yaml
# D: new nested-source protocol; historical 139 ETH windows regression only
outer_holdout_source_family: hotel_biwi_hotel_all_aliases
inner_validation_raw_source: uni_examples.txt
train_source_policy: exclude_outer_and_inner_entire_files_before_windowing
reject_duplicate_raw_content: true
historical_eth139_is_final_test: false
base_checkpoint: null # must hash a freshly clean-trained GDTS; old SHA regression only
base_train_cap_epochs: {goal: 150, joint: 250} # D; fixed caps, not promise to finish
base_selection: {goal: masked_BCE, joint: minADE} # D internal source only
base_optimizer: {type: Adam, lr: 0.001, betas: [0.9, 0.999], weight_decay: 0} # D
base_scheduler_gamma: {goal: 0.99, joint: 0.995} # D
base_clip: 1.0 # D
base_early_stopping: {patience_epochs: 10, min_delta_m: 0.001} # D joint
obs_length: 8 # I
pred_length: 12 # I
trajectory_dt: 0.4 # I; metres only in graph/metrics
graph: {type: radius_ttc, radius: 6.0, ttc_threshold: 8.0, adaptive: false} # I
Kg: 21 # I
Ky: 20 # I for C
P: 20 # I
relation_modes: 4 # I
rank: 8 # I
social: {width: 128, gru_layers: 1, message_layers: 1, dropout: 0.0} # I
unary: {width: 64, zero_init_last: true} # I, fresh only
relation: {geometry: 4, widths: [32,16]} # I; B explicitly geometry6
scene_latent: false # I
refinement: {rounds: 2, policy: exact_lexicographic_persistent_tie} # I
diffusion: {ddpm: 100, ddim: 20, trunk_end: 30, branch_index: 14} # I
trainable: [social, base_relation, unary, dynamic_relation, energy, future_teacher] # I A/B
frozen: [goal_UNet, history_LSTM, GDTS_diffnet, corrector] # I A/B
objective: expected_conditional_mc # D A only; control mean_energy unchanged
neighbor_draws: 4 # D; no search
goal_target_sigma_m: 1.0 # I
loss_weights: {post: 0.5, prior: 0.5, relation_kl_max: 0.1} # I
relation_warmup_stage_fraction: 0.2 # I
optimizer: {type: Adam, lr: 0.0001, betas: [0.9,0.999], weight_decay: 0.0} # D
gradient_clip_norm: 1.0 # D
scheduler: {type: ExponentialLR, gamma: 0.99, lr_warmup_epochs: 1} # D
epoch_cap: 40 # D
early_stop: {patience: 5, validation_every_epochs: 1, min_delta_JADE_m: 0.001} # D
selection: internal_source_mean_JADE # D; JFDE then earlier epoch tie break
batch: {scene_budget: 1, agent_budget: 128, edge_budget: 512, accumulation: 8} # D A/B
edge_chunk: 256 # I
amp: {enabled: true, dtype: bf16} # I; preserve existing FP32 islands
training_seeds: [3101,3102,3103] # D, distinct optimizations with shared clean base
inference_seeds: [2035,2036,2037,2038,2039] # I explicit separated RNG namespace
cache_version: innovation-design-v1 # D metadata wrapper; do not overwrite jdv2-cache-v1
cache_payload_schema: jdv2-cache-v1 # I tensors; all new split/base/source hashes required
validation_hyperparameter_search: none # D; no tuning on ETH139
final_test: once_after_selection_and_analysis_freeze # D
C_overrides: # D except fixed existing V4 widths/loss forms
  batch: {scene_budget: 4, agent_budget: 64, edge_budget: 256, accumulation: 1}
  amp_coupling: false
  trajectory_encoder: {step: [7,128,128], gru_hidden: 128, gru_layers: 1}
  sync: {rounds: 4, sinkhorn: 8, temperature: 0.2, topk: 0, threshold: 0.6}
  weights: {alignment: 1, pair_score: 0.5, pair_assignment: 0.5,
            no_harm: 2, entropy: 0.02, relation_prior: 0.02, gate: 0}
~~~

A/B不新建packed执行模式：scene_budget1沿用原variable-N场景loader，积累8窗等权；共同baseline同预算。不要仅更改parser的use_multi_scene_packing，它主要为既有V4路径定义。超预算大scene不拆成伪独立agent batch；停止并申请新预算。base也必须训练源不含inner validation，否则冻generator已泄漏；base尚未生成所以身份null是有意阻塞clean评估，不填旧SHA冒充。

## 7. 下一步最小实现（另行授权）

只允许以下默认off的目标实现；不顺带优化缓存/网络/精度/RNG。

| 拟议文件/接口 | 改动 | CPU验收 |
|---|---|---|
| src/joint_goal_loss.py: expected_conditional_composite(unary,q,scene,edge_cost_chunks,edge_index,neighbor_ids)→scalar | 纯FP32损失，post/prior共用邻居IDs，无Parameter | onehot与旧loss等价；小K枚举期望vsMC统计；analytic gradient；E0/N1 |
| src/models/model.py: _jdv2_no_z_goal_losses | 默认原分支；新flag取MC支；不改eval | 教师/post/KL梯度全保留；基座/unused embedding无新梯度；跨scene mask |
| src/parser.py + trainer.py | 显式目标版本/S4，独立训练generator state入checkpoint/原子保存，默认旧语义 | reload后draw序列一致，不污染global/eval RNG；未知目标拒绝 |
| tests/test_jdv2_expected_conditional.py | 小CPU张量合成测试，不实例化完整A | chunked/dense误差阈值FP64 1e−10/FP32 1e−6；所有q/cost有限检查 |

最小授权预算：CPU≤1小时，训练0、GPU0；先新增loss及测试，训练接线若涉及resume工程可拆第二PR，不自动启动。生产实现需完整兼容测试后再申请GPU门槛；这里“测试计划”不冒称本轮pytest已通过。
