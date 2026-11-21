"""Chinese engineering report from measured zero-update receipts."""
from common import *


def main():
    ready=read(PUBLIC/'READINESS.json');dev=read(PUBLIC/'INITIAL_DEV.json');asset=read(PUBLIC/'ASSET_RECEIPT.json')
    all_,dense=dev['groups'];cpu=read(PUBLIC/'CPU_RECEIPT.json');real=read(PUBLIC/'REAL_ACCEPTANCE.json');recover=read(PUBLIC/'RECOVERY_RECEIPT.json')
    rr=list(csv.DictReader((PUBLIC/'epoch_000_windows.csv').open()));multi=[r for r in rr if int(r['n_agents'])>1]
    entropy=sum(float(r['sinkhorn_normalized_entropy']) for r in multi)/len(multi)
    table='\n'.join(f"| {r['group']} | {r['scene_count']} | {r['JADE']:.9f} | {r['G_JFDE']:.9f} | {r['I_JFDE']:.9f} | {r['H_JFDE']:.9f} | {r['feedback_gain']:.9f} |" for r in dev['groups'])
    text=f'''# SSEC Energy-only ZARA2 Stage2 正式入口准备与零更新验收

**结论：{ready['status']}。** 当前没有未解决的工程阻塞，可以等待另一次明确的训练授权；**本轮没有启动训练，真实 optimizer updates=0，合成 optimizer updates=0，完成训练epoch=0。**

基点：`{BASE}`。本次新增正式入口、冻结配置、监测、零更新验收和唯一受控命令；原SSEC Energy-only网络及v2/v3正式入口均未修改。初态验证不是训练结果，不据此选出权重或宣布新网络有效。

- [正式入口](../../../../bcep_scene_energy/formal.py)、[数据流/监测/恢复说明](INTERFACE_ZH.md)
- [READINESS](READINESS.json)、[冻结配置](FROZEN_CONFIG.json)、[配置SHA](CONFIG_SHA256.txt)、[源码SHA](SOURCE_SHA.json)
- [唯一受控训练命令](CONTROLLED_COMMAND.txt)，**仅准备、未执行**；[零更新复现](REPRODUCE.md)

## 1. 固定资产、初始化和授权

Provider保持density epoch19/head epoch9，四层Flow、M256→KMeans20及既有有界修正均未重训。原seed2035 train/dev refined bank逐行核验；官方训练2681窗、开发590窗，其中127窗N≥9。Stage2 seed3201、K20、20步DDIM、六层条件桥、final_coordinate、FP32、AdamW、微批与分组归约、H-JADE选模规则保持原样，不使用可选CSEE。

模型共{real['parameters']:,}个参数。初始参数指纹为：

`{real['initial_model_sha256']}`

实际数值环境见 [RUNTIME.json](RUNTIME.json)，包括GPU、PyTorch/CUDA版本与FP32确定性开关。

它与固定基点已归档的SSEC随机初态逐字节一致。没有用训练后的旧v2或v3权重初始化新模型，Provider不在Stage2优化器中。

[INPUT_ROWS.csv](INPUT_ROWS.csv)包含3271个train/dev窗口的实际数据SHA、候选文件SHA及成员SHA；[REAL_INPUTS.json](REAL_INPUTS.json)给出Provider、bank和划分身份。

正式运行同时要求`RUN_ONE_ZARA2_SSEC=true`和冻结配置**文件SHA256**：

`{ready['frozen_file_sha256']}`

源码清单和匹配的READY收据也必须一致。缺授权/缺SHA/错误SHA时，测试证明入口在模型、数据和优化器构造前拒绝执行。所有参数、fold、seed及输出目录固定；不存在自动测试、扩折或旧队列启动路径。未来授权写入新的运行授权文件，不覆写本轮冻结配置中的零更新事实。

## 2. 全590窗初态开发集结果

实际运行全部590窗、原20步DDIM，耗时约{dev['seconds']:.2f}秒。全部模型参数在验证前后哈希一致。以下均为米；JFDE与反馈收益按窗口等权。

| 分组 | 窗数 | H-JADE | G-JFDE | I-JFDE | H-JFDE | I−H |
|---|---:|---:|---:|---:|---:|---:|
{table}

全体边缘minFDE保持agent-micro **{dev['metrics']['minFDE']:.12f} m**、scene-macro **{dev['metrics']['scene_weighted_minFDE']:.12f} m**，与冻结bank归档值一致。原路径统计JFDE与直接端点JFDE均保留；两种FP32归约的差异在约1e−9 m量级，不影响H-JADE选择器。

当前随机初态的全体反馈收益为负：最终JFDE比初配增加约{-all_['feedback_gain']*1000:.3f} mm；N≥9改善约{dense['feedback_gain']*100:.3f} cm。**公共能量下降没有保证全体真实JFDE下降。** 这些是初态监测基线，不是已训练效果，也没有据此改变损失、选模或超参数。

[INITIAL_DEV.json](INITIAL_DEV.json)、[逐窗CSV](epoch_000_windows.csv)、[分组统计](epoch_000_groups.json)、[人数成本](epoch_000_costs.csv)保留未舍入数值。文件中的epoch000只表示初态诊断，没有创建“完成第0轮”的训练权重或Top5。

## 3. SSEC专属监测已接入

初态all组有{all_['attempted']}次多人协调提案，其中接受{all_['accepted']}、拒绝{all_['rejected']}、严格并列{all_['strict_ties']}、可靠下降候选{all_['reliable_decreases']}、舍入不确定{all_['roundoff_uncertain']}；N≥9接受{dense['accepted']}/{dense['attempted']}。每次被接受更新均通过原唯一性与FP64可靠下降检查。

all的接受能量变化总和为{all_['accepted_energy_delta_sum']:.9f}，N≥9为{dense['accepted_energy_delta_sum']:.9f}，它们是神经公共能量单位，不能当作米单位的FDE收益。

最终提案有效评分RMS的窗口均值：all {all_['final_proposal_effective_score_RMS']:.9f}，N≥9 {dense['final_proposal_effective_score_RMS']:.9f}。对应proposal匹配gap均值为{all_['proposal_matching_gap_mean']:.9g} / {dense['proposal_matching_gap_mean']:.9g}。多人501窗的归一化Sinkhorn熵均值仍为 **{entropy:.9f}**，接近均匀；全590窗平均会被89个单人identity旁路的零熵拉低，不能据此宣称软硬偏差已解决。

正式每轮将保存all/N≥9的I/H-JFDE及I−H、全部协调决策计数、FP64能量变化、φ/ρ梯度与实际参数差、评分/gap/软熵和人数成本。参数差由原优化器post hook在真实更新后测量，不用梯度乘学习率代替。当前真实更新为0，因此没有伪造非零参数更新记录。

## 4. 实际反向与零更新证据

使用真实训练窗口N=1、2、9、57执行原完整反向，并额外执行包含这些窗口的16窗原生逻辑组。多人φ/ρ和共享编码器梯度均有限、非零；不根据梯度范数偏小判失败。单人按既有协议旁路联合配对，因此φ/ρ无梯度是预期行为，共享编码器仍有梯度。

| N | φ梯度L2 | ρ梯度L2 | 共享编码器梯度L2 |
|---|---:|---:|---:|
| 1 | 0（旁路） | 0（旁路） | 0.084424903 |
| 2 | 0.023985922 | 0.014226493 | 0.097887641 |
| 9 | 0.009412827 | 0.002496247 | 0.022239772 |
| 57 | 0.015317534 | 0.003230488 | 0.017557721 |

真实57人反向allocated峰值约2240MiB，单次审计约2.23秒，不作为稳定训练吞吐预测。原分组回调与加入身份/监测的回调在相同随机输入下，总损失和全部参数梯度逐值一致。所有反向前后模型哈希均未变。

四个真实规模都捕获了**实际DDIM sampler**输入：配对一次、20步DDIM、固定终点精确gather、原bank槽位keyed noise逐值一致。[REAL_BACKWARD.csv](REAL_BACKWARD.csv)、[16窗逻辑组](LOGICAL_GROUP_BACKWARD.json)、[真实验收收据](REAL_ACCEPTANCE.json)可核验。

## 5. 完整状态恢复、Top5与早停

恢复测试实际调用原`run_stage(..., resume=True)`，恢复到另一份初始参数不同的同结构实例，然后在第一组反向/更新前按暂停请求退出。严格模型加载1次，恢复后指纹精确等于本次seed3201初态。

模型、326组AdamW非空零步槽位、param groups、调度、CPU/CUDA/Python/NumPy/order/view RNG、数据顺序/cursor、history和各项epoch统计均严格一致；下一批及下一次随机抽样一致。改变recipe会拒绝恢复。AdamW零动量是显式零更新测试夹具，**不是通过step产生**，不冒充训练后的动量恢复。

原规则仍为5轮warmup、100轮调度、每轮168更新、minimum30、patience15、max100。原checkpoint writer的实际函数体完成原子保存及Top5排序测试；原停止规则的实际表达式覆盖minimum/patience/上限与严格同分取早轮。测试用的epoch标签只属于明确标注的writer fixture，不算训练epoch。

本轮没有通过真正AdamW更新验证连续训练与中断训练的轨迹等价；该限制来自本轮零step要求。原控制器、优化器和恢复实现未修改，所有完整epoch、best、last及Top5不同epoch的保留策略沿用并写入冻结配置。[RECOVERY_RECEIPT.json](RECOVERY_RECEIPT.json)和[CPU测试收据](CPU_RECEIPT.json)记录边界。

## 6. 最终准入与保护结果

CPU共 **{cpu['tests']}项测试通过**，包括已有网络/身份测试、未授权拒绝、实际writer/停止规则、监测有限性、post hook注册和恢复日志汇总。所有验收进程的optimizer.step拦截计数为0。

历史受保护资产共 **{asset['protected_files']:,}个文件、{asset['protected_bytes']:,}字节**，逐文件哈希一致。v2五折仍PAUSED_RESUMABLE，v3仍COMPLETE，candidate_cost仍SESSION_BUDGET_PAUSED，sharedworld仍COMPLETE。新正式运行目录只有准备资料和锁，未生成正式`joint/last.pt`、训练授权记录或完成epoch。

工程结论为 **{ready['status']}**。没有要求先解决的新增工程阻塞。仍保留此前明确的限制：固定行人成员顺序，SSEC坐标更新顺序依赖，Sinkhorn近均匀与proposal-ST偏差，以及尚未验证训练后的预测质量。

等待独立训练授权；本轮结束，不启动训练、其他seed、模块消融、其他折、测试集或旧队列。
'''
    (PUBLIC/'README.md').write_text(text)
    print('Chinese report generated from receipts',flush=True)
if __name__=='__main__':main()
