# L1 bounded update / stop

仅显式 p2_mode=l1，Stage-A joint_goal/strict_no_z，seed3101；固定K21/P20/M4/rank8/两轮sampler，原网络及两种目标定义不变。A0不创建MC owner；A1保留S4 CPU target Generator和既有CUDA FP32/BF16门禁。本轮仅CPU tiny fixture验收，不认证真实CUDA运行。

生产 max_update_attempts≤500、reference_total_steps=3484、num_workers0、accumulation1、skip_validation/no_automatic_resume。每步 t 前 progress=(t−1)/3484，beta=.1 min(progress/.2,1)，第500次0.07161308840413318，500次成功后500/3484；不能以H替代native epoch。Adam1e−4/clip1/ExponentialLR gamma.995不变。

检查顺序：cap/resource/pair-stop → next(loader) →身份认证→loss→backward→原optimizer分支→记录outcome。fetchattempt与实际fetch/loss/backward/optimizerattempt/success/skip/failure/unknown独立。loss按实际计入样本数平均，另列success分母。identity失败和异常不能取新窗补齐。原子pair-stop marker保留首个失败原因，另一臂在fetch前停止。

共享pair monotonic origin须当前进程可解释的同boot时钟，非有限/未来值拒绝；两臂需在认证的新pair ledger下运行。阶段2700s、每臂900s、reserved10GiB，长算子只能在返回边界被发现超限；CPU用fake probes，生产未来CUDA探针本轮未执行。故障不回滚已发生参数/RNG消费，不清MC pending。

| stop | scheduler | validation/best/ordinary-last | snapshot |
|---|---|---|---|
| 部分epoch正常cap | 0 | 0 | 独立weights-only |
| 恰好完整native epoch | 1 | 0 | 独立weights-only |
| skip/exception/nonfinite/identity/resource | 不当作完整epoch | 0 | 审计ledger；不伪造可恢复权重 |

snapshot名 bounded_stop_weights_only.pt、artifact_role同名、resumable=false，已存在拒绝覆盖；原子写入复用既有helper。旧resume入口拒绝；P2 base也不认证training resume。ledger含cursor/exposurehash/progress、首末loss/components、N/E、gradient/update norm、冻结参数hash及MC draw/agentdraw/pending。旧format1 CPU exact resume与旧format2 metadata回归不等于新L1部分epoch或真实CUDA续跑认证。

shared initial严格新artifact role、合格base/data binding、完整state hash、fresh_heads_zero_corrector声明、optimizer_updates0；本轮真实state为null。未来资格认证仍须核验实际初始化组件与冻结base，不可仅由填写policy字符串跳过审核。L2选择适配未实现。
