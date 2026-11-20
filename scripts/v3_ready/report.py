"""Publish readiness only after all zero-real-update acceptance receipts pass."""
import json,shutil,sys,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write
from bcep_joint.checks import equal
from bcep_v3.formal_inputs import FormalInputs,paths,BASE
from bcep_v3.formal import freeze,model_hash
from bcep_v3.model import build
from readiness import protect
PUBLIC=ROOT/'docs/research/bcep_v3/stage2_ready_20261120'


def main():
    inputs=FormalInputs();roots=paths();out=roots['audit'];run=roots['run'];PUBLIC.mkdir(parents=True,exist_ok=True)
    f,_=freeze(inputs);assert f==read(run/'FROZEN_CONFIG.json');h=canonical_hash(f)
    before=read(out/'PROTECTED_BEFORE.json');assert protect(roots)==before
    assert not (run/'joint').exists() and not (run/'AUTHORIZATION.json').exists()
    assert read(run/'STATUS.json')['optimizer_updates']==0
    receipt={n:read(out/n) for n in ['REAL_FRESH.json','SYNTHETIC_RECOVERY.json','REAL_ACTIVATED.json','INITIAL_DEV.json']}
    assert all(v['status']=='PASS' for v in receipt.values())
    assert receipt['SYNTHETIC_RECOVERY.json']['physical_optimizer_updates']==8
    equal(torch.load(out/'synthetic/continuous/last.pt',map_location='cpu',weights_only=False),torch.load(out/'synthetic/resumed/last.pt',map_location='cpu',weights_only=False))
    receipt['SYNTHETIC_RECOVERY.json']['entire_checkpoint_dict_exact']=True
    xml=ET.parse(out/'CPU.xml').getroot();suites=list(xml.iter('testsuite'));count=sum(int(x.attrib['tests']) for x in suites)
    assert count==21 and not any(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    provider={}
    for d in f['provider']:
        p=roots['formal']/'zara2'/d['stage']/d['file'];assert digest(p)==d['sha256'];provider[d['stage']]=dict(sha256=d['sha256'],epoch=d['epoch'],unchanged=True)
    model,_=build(inputs.config)
    assert {n.split('.')[0] for n,p in model.named_parameters()}=={'bridge','coupling'}
    batch=inputs.data.batch([inputs.split['train_indices'][0]],'train');g=inputs.banks[0].batch([inputs.split['train_indices'][0]],batch);assert not g.requires_grad
    protection={name:dict(status=d['status']['status'],protected_files=len(d['files']),all_sizes_mtimes_unchanged=True) for name,d in before.items()}
    write(PUBLIC/'PROTECTION.json',dict(roots=protection,provider=provider,provider_absent_from_stage2_optimizer=True,candidates_require_grad=False,no_queue_started=True))
    for n,v in receipt.items():write(PUBLIC/n,v)
    write(PUBLIC/'FROZEN_CONFIG.json',f)
    shutil.copy2(run/'INITIALIZATION_ORIGIN.json',PUBLIC/'INITIALIZATION_ORIGIN.json')
    shutil.copy2(out/'dev/epoch_000_groups.json',PUBLIC/'INITIAL_DEV_GROUPS.json')
    shutil.copy2(out/'dev/epoch_000_windows.csv',PUBLIC/'INITIAL_DEV_WINDOWS.csv')
    (PUBLIC/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
    status=dict(status='PASS',frozen_sha256=h,base_source=BASE,ready_for_one_authorized_ZARA2_run=True,
                engineering_blockers=[],training_authorized=False,real_optimizer_updates=0,synthetic_optimizer_updates=8,cpu_tests=count,
                old_v2_initial_equivalence=True,protected_assets_unchanged=True,no_test_access=True,no_auto_expand=True)
    write(out/'READINESS.json',status);write(PUBLIC/'READINESS.json',status)
    command=f'RUN_ONE_ZARA2_V3=true OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ../.conda/rsjg/bin/python -m bcep_v3.formal train --frozen-sha {h}'
    (PUBLIC/'CONTROLLED_COMMAND.txt').write_text(command+'\n')
    fresh=receipt['REAL_FRESH.json'];activated=receipt['REAL_ACTIVATED.json'];synth=receipt['SYNTHETIC_RECOVERY.json'];groups=read(PUBLIC/'INITIAL_DEV_GROUPS.json')
    grads=[]
    for prefix,label in [('coupling.csee.','CSEE'),('coupling.hcra.','HCRA'),('bridge.denoiser.grg_ctf.','GRG-CTF')]:
        a=fresh['backward'][0]['gradient'][prefix]['gradient_l2'];b=activated['gradient'][prefix]['gradient_l2'];c=synth['telemetry'][3]['modules'][prefix]
        grads.append(f"| {label} | {a:.8g} | {b:.8g} | {c['gradient_l2']:.8g} | {c['parameter_delta_l2']:.8g} | 0 |")
    table=[]
    for g in groups:table.append(f"| {g['group']} | {g['scene_count']} | {g['JADE']:.9f} | {g['G_JFDE']:.9f} | {g['I_JFDE']:.9f} | {g['H_JFDE']:.9f} |")
    report=f'''# BCEP–EBJD v3 正式 ZARA2 Stage2 训练准备

**入口及必要工程验收已完成，具备在另行明确授权后启动一次 ZARA2 Stage2 的条件；当前没有必须先解决的工程阻塞。本任务实际真实 optimizer updates=0，没有启动训练。** 合成验收独立执行8次更新。固定基础 `{BASE}`，独立分支 `prepare/zara2-bcep-v3-stage2-20261120`。

| 用户要求的判断 | 结果与证据 |
|---|---|
| 是否具备真实训练条件 | 是；正式 train/dev loader、bank、原 grouped backward、完整H-JADE验证及原 run_stage 已接通；全590窗完成零更新评测 |
| 是否存在工程阻塞 | 当前已测范围无必须先解决的阻塞；训练授权尚未给出，入口保持未启动 |
| 初始模型能否退化为旧v2 | 是；同seed3201旧参数指纹相同，真实1/2/9/57人窗口完整20步 G/H轨迹最大差0，pi一致 |
| 是否保护历史资产 | 是；v2、candidate_cost、sharedworld状态、权重、bank清单未变化；Provider所选权重SHA再次核验 |
| 实际真实更新是否仍为0 | 是；真实数据只前向/反向不step，参数哈希前后不变；正式运行目录没有joint checkpoint或AUTHORIZATION.json |

## 入口改动与结构不变量

新增 `bcep_v3/formal_inputs.py`、`formal_validation.py`、`formal_telemetry.py`、`formal.py`；stage2.py保留原合成入口，另增加只接受已核验FormalInputs的正式回调。原 `bcep_v2.stages.joint_callbacks` 和 `bcep_v2.state.run_stage` 直接复用；只包装记录梯度/更新以及验证分组指标，没有重写优化器更新、数据迭代或早停算法。

启动前逐文件与固定基础比较：CSEE、HCRA、GRG-CTF、模型初始化与张量接口完全不变；bcep/ebjd/bcep_v2/bcep_joint既有Python源码不变。四层Flow、六层3+CTF+3桥、final_coordinate旧有效损失、K20、20步DDIM、两次Hungarian、终点双射与keyed noise对应、原损失权重全部保持。Provider不注册到Stage2模型/optimizer，候选tensor不带梯度，固定bank是唯一候选来源。模型只含bridge和coupling可训练参数。

正式阶段仍采用随机初态seed3201，不加载旧epoch26微调；新增模块使用已验收的局部初始化。`INITIALIZATION_ORIGIN.json`记录旧/新张量来源与SHA，冻结初始完整指纹为 `{f['initial_model_sha256']}`。原结构中的支持池统计缺失、近并列Hungarian换解等已知限制继续存在，本轮未通过改网络或换损失规避它们。

## 数据身份与候选来源

官方ZARA2 train **2681窗**（最大57人），selection_dev **590窗**（最大42人，其中N≥9为127窗）。核验split原始SHA、规范化split身份、每个train/dev数据记录SHA、bank成员/窗口/候选顺序与逐行npy哈希。加载器在原FormalData之上补齐显式单位/坐标/帧映射元数据，数值与成员顺序不做变换；单位米、dt0.4秒、观测8/预测12。增强仍调用原联合变换，obs/maps/candidates同步。

候选seed2035；固定Provider density epoch19、候选修正头epoch9。train bank `{f['bank_sha256']['train']}`；dev bank `{f['bank_sha256']['selection_dev']}`。不重新导出bank、训练Provider或改变终点坐标。只核对train/dev文件；heldout/test访问在适配器边界拒绝，不加载测试窗口、不用测试选模。

## 真实入口前反向与模块更新验收

真实输入验证不是训练：直接调用正式回调进行原始逻辑16窗分组backward，包含最大57人窗口；上下文内禁止AdamW.step，模型完整参数哈希不变。该检查峰值allocated **{fresh['backward'][0]['peak_bytes']/2**30:.3f} GiB**，前反向耗时 **{fresh['backward'][0]['seconds']:.3f}s**，仅为一次容量观测，不是正式训练吞吐保证。

另将完全独立的合成数据更新4步模型用于只读真实57人梯度诊断，随后丢弃；它不进入正式初始化/恢复目录。三个新增模块激活后都获得有限梯度，真实参数变化仍为0。合成组通过原AdamW真实更新，记录了每次参数差值而非只看日志损失。

| 模块 | 真实初态梯度L2 | 合成激活后真实57人梯度L2 | 合成第4步梯度L2 | 合成第4步参数变化L2 | 真实参数变化 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(grads)}

CSEE初态梯度0来自下游ScoreDelta零出口，是已验收的初始化行为；激活后约1e-7量级不被直接判为失败。判据是图路径、有限性、跨步参数实际变化与恢复一致性，不设凭空的梯度范数下限。未来正式训练在每个实际step后写 `MODULE_UPDATES.jsonl`，记录三个模块裁剪前梯度范数及真实参数增量；范数仅供机制观察，不能证明模型有效。

## 全开发集零更新验证

同一未训练初态完成590窗的原20步H-DDIM，按原MetricAccumulator得到完整H-JADE。初配pi从同一assign返回；G/I/H终点JFDE由同一固定候选计算，H-JADE来自完整轨迹，不增加DDIM步数或重新配对。

| 分组 | 窗数 | H-JADE m | G-JFDE m | I-JFDE m | H-JFDE m |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

上述是**初始化管线检查，不是训练结果或精度改进证据**，没有据此选权重或改超参。全体agent-micro minFDE保持0.3520422351797007 m，逐窗等权oracle保持0.33942601943420153 m；逐窗G/H边缘minFDE在1e-6容差内一致。匿名明细见[INITIAL_DEV_WINDOWS.csv](INITIAL_DEV_WINDOWS.csv)。正式每轮同样记录all/N≥9的G/I/H-JFDE、反馈收益和H-JADE；唯一selector仍是全体dev H-JADE，完全同分取早轮。

## 恢复、早停、Top5与保护

GPU合成验收：连续4步与2+2恢复，共8次物理合成更新。中断发生在首轮未完成、cursor=4处，不能被标记为完整epoch。恢复后model、AdamW、scheduler、CPU/CUDA/Python/NumPy/RNG、数据order/cursor、stale/history及统计分子分母逐项一致。[SYNTHETIC_RECOVERY.json](SYNTHETIC_RECOVERY.json)

CPU **21项测试通过**。用原run_stage与正式minimum30/patience15/max100规则核验：验证值并列且不改进时于第30轮原生early_stop；Top5为最早五个不同epoch，保留全部30个完整快照、best和last，状态包含完整恢复字段；recipe不匹配拒绝恢复。实际训练沿用原AdamW、5轮warmup/100轮调度、FP32/math-SDPA、4窗微批×4累积、group_max_slots32、每轮168更新。所有完整epoch都保留，至少最佳五个不同epoch，选模不看训练loss或test。

入口同时持有本次run锁及v2/candidate_cost/sharedworld控制锁，旧运行状态仍分别为PAUSED_RESUMABLE / SESSION_BUDGET_PAUSED / COMPLETE。SIGTERM由原controller请求在安全边界保存；继续使用本次last恢复，身份或配置不符则拒绝，不从旧实验恢复、不自动拉起任何其他队列。技术中断恢复最近原子保存点，尚未落盘的更新可能按保存的RNG顺序重放；逐step日志保留实际执行记录。磁盘预检覆盖剩余所有epoch快照与原子写入余量，运行期仍保留原磁盘检查。[PROTECTION.json](PROTECTION.json)

## 冻结配置与唯一受控命令

冻结配置SHA（规范化JSON）为 **`{h}`**。`allow_real_training=false`记录本次准备状态，配置不被偷偷改成true；将来只有用户另行明确授权并主动执行以下命令，才会写单独AUTHORIZATION.json并进入真实run_stage。该命令本轮**没有执行**。无环境授权时，train在加载数据、构建模型前直接拒绝；还必须同时匹配冻结SHA、全部源码哈希和PASS验收收据。

从本次独立worktree根目录执行，Python为同级已验证环境：

```bash
{command}
```

相同命令只从本次运行目录的last恢复；已COMPLETE则返回，不新建第二次训练。不提供换fold/seed/超参选项，不启动Stage1、测试集评测或后续队列。唯一正式输出名 `{f['experiment']}`，准备验收输出名 `zara2_bcep_v3_stage2_readiness_20261120`；两者分离，合成权重不会被误当正式last。

[冻结配置](FROZEN_CONFIG.json) · [验收收据](READINESS.json) · [真实初态验收](REAL_FRESH.json) · [激活后真实梯度](REAL_ACTIVATED.json) · [初始开发集](INITIAL_DEV_GROUPS.json) · [唯一命令](CONTROLLED_COMMAND.txt)。权重、候选库和内部路径留在主机，GitHub只提交代码、脱敏配置/统计及报告。

结论：可以进入用户审阅后的单折受控训练授权阶段；工程PASS不等于精度提升，也不保证长期训练没有数值或资源故障。本任务到准备与验收结束，真实optimizer updates仍为0。
'''
    (PUBLIC/'REPORT_ZH.md').write_text(report)
    write(PUBLIC/'ARTIFACT_SHA256.json',{p.name:digest(p) for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print('READY',h,'real updates 0; synthetic updates 8; CPU',count)
if __name__=='__main__':main()
