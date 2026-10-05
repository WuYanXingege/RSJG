# 命令、执行顺序与计数

工作目录：RSJG_JDV2_clean；解释器：../.conda/rsjg/bin/python -B。
实际输入 HEAD c9dbfecf64eb95504664c2901d3c2bb26f41aea5，branch research/joint-dependency-v2-clean。
起始 tracked worktree clean，已有 untracked data/outputs 全保留。未发现适用 AGENTS.md，未启动新子代理。

1. 只读 git status/branch/HEAD、源码/历史报告/配置/manifest/provenance 检查、版本与官方文档核对。sandbox nvidia-smi 看不到 GPU；经授权 escalated read-only 查询成功。未 kill/暂停任何进程。
2. 必需后端补丁、6 项 CPU 合同测试。
3. `CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_c9dbfec_cuda_amp_loss_smoke/run_cpu_audit.py`：只执行一遍，172 passed in 9.79s，runner wall 10.029252s。无真实模型 forward/CUDA init，CPU thread 保持10。历史独立 resume child 沿用自己原有线程设置（未改 GPU 策略）。stdout 中间行被工具截断；保留原始截断文本和完整尾部，不声称完整 per-phase raw ledger。
4. `.../prepare_real.py`：一次 train sorted selection；4112 source batch deserializations、22 deployment candidate cache、2 teacher cache；checkpoint 仅 hash、不 deserialize。之前探索只读 train/000000.pt 与 train_batch_0001.pkl 各一次。
5. `.../audit_gpu.py --plan`：第一次因文件转义语法错误未执行；第二次完成前两例 4 个 CPU loss/backward，第三个 singleton draw 后 empty cost max 失败；修正 CPU preparation 后第三次完成16例、48个 CPU scalar forward/backward、16次 draw。没有 CUDA 初始化，没有重跑 CPU 回归套件。固定输入/z/start/end bytes 保存为 SYNTHETIC_INPUTS.pt，SHA/shape/stride 为 SYNTHETIC_PLAN.json。
6. AST syntax 与 git diff --check。AUDIT_PLAN/PREFLIGHT/SYNTHETIC_PLAN 在 GPU 前已落盘，阈值未放宽。
7. 仅一次 escalated `../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_c9dbfec_cuda_amp_loss_smoke/audit_gpu.py --run`。资源检查后，CPU 再验证 manifest/selected hashes、各载入两个 source/deployment/teacher 窗口，随后首次 CUDA init 起 900s watchdog。无 warmup、重试、替换窗口。输出保存在 gpu_run/RESULTS.json 与20个原生small .pt。结束 status=CUDA_FP32_BF16_LOSS_SMOKE_CERTIFIED_NO_TRAINING，exit0，GPU 3.334183s。
8. `.../source_scope.py` 与 `.../postcheck_cpu.py`：只读 AST/原生小证据分析，无新 loss/model/GPU 调用。JSON 保存为 SOURCE_SCOPE/POSTCHECK。
9. 显式 stage 本轮路径、普通 commit/push；完整commit远端 fetch_file 回读 bytes/SHA256，receipt 放 /tmp（不再生成自引用提交）。最终交付消息报告真实发布状态，不提前把计划推送当成功。

## 账本解释

GPU有效合成 top-level=16，每例一次total backward；两例 mixture 各包含2个post/prior CE，故纯CE primitive=18。非法 attempts=8，其中6个 owner.draw predraw拒绝、2个纯loss拒绝；无非法 backward。真实 get_loss/encode/backward attempted=completed=4，failed=unknown=0。一个 GDTS constructor、一个 canonical checkpoint strict load；真实 sampler/diffusion/corrector/full trajectory=0。

GPU owner draw=16合成+3真实，agent-draw=288+28，toy successful updates=2；合成其余14次是skip，真实3次是skip，无pending。真实 weights/buffers全不变。

CPU回归 runner counts 为父进程：Adam step109、owner draws124、agent draws2268；独立 resume child另有更新6、draw6、agent draws120。fixture counters 是其子集，不能与父计数相加。纯函数测试中的 grad API等不属于真实训练。CPU计划另有3（部分失败run）+16（完成run）draw，与CPU回归分开。

Native module pre-hooks实测完整键在gpu_run/RESULTS；例如goal U-Net4、social/GRU4、message-layer4、history LSTM4、unary4、dynamic geometry3。teacher/relation某些子MLP每有边窗执行双方向，计6，不能把父方法调用与子模块次数相加当模型forward。未使用内部算子observer。真实 artifact读取与审计 evidence .pt 重读分别说明，未将后者伪装真实checkpoint resume。

警告：一次 requires_grad scalar转float提示（记录已完成backward的值，未改变图），一次Transformer nested-tensor提示。没有 NaN/OOM/数值gate失败。GPU预算已使用至case上限；严禁删除 gpu_run guard 后重跑来补证据。
