# 本轮失败、修正与测量边界

- 首次公开 release 直连 range GET 超时，接收0字节。仅在本条命令范围移除格式错误的小写 proxy 变量后，沿用已有有效大写 HTTPS proxy 完成 range GET 和实际下载；没有修改全局代理。含 cookie/签名重定向的原始响应 headers 仅在本地 outputs，禁止提交 GitHub。
- 发行包首次绑定已实际通过60个引用。随后发现 download script 的 fallback 实际路径为仓库根目录 `download_data.sh`，修正归档 URL 后重新完整绑定，未修改本地源数据。
- 第一轮 pytest：143 passed / 6 failed。六项是旧 observation suite 依赖的专用 CPU guard 未安装：full-pickle 拦截、4个 CUDA Python 入口与1个 native 查询拦截。不是六个数值错误。未安装相应 query hook 的首次运行确实可能调用 CUDA/NVML 查询，不声称 CUDA 枚举0；新 harness 的 `_lazy_init` 仍禁止初始化。失败记录和原始日志 SHA 保留。
- 修正为两个独立进程：新协议/loader 测试使用可计数的 synthetic optimizer 插件；原 observation suite 使用其原始专用 guard 并显式 `CUDA_VISIBLE_DEVICES=''`。149项重新通过（99+50），不删除、不跳过失败测试。
- 旧 guarded harness 本轮输出写入新审查目录，没有修改旧审查记录。unit fixture 优化器仅用于 tiny synthetic 参数，和真实 train-data smoke 的累计24次预算分开。
- 全量 exporter 运行期间只扩展了 CLI 的后续 verify/receipt 行为，没有修改其侧车绑定的数值 producer、旧 observation producer 或原 Gaussian 实现。全量 verify 将以最终 CLI 实走；运行中旧 CLI 未包含的补充检查不能倒算为第一次导出已执行。
- 首批 bootstrap/只读发现命令没有完整 PID/RSS 仪表，不补造零记录。之后 supervisor 记录实际 PID namespace、PID、start ticks、进程树、HWM和采样 aggregate RSS；计算任务顺序执行。名义采样1秒，实际间隔由 RESOURCE_SUMMARY 报告；采样峰值不等于连续测量峰值。
- 原始 import-before-hook/native CUDA 覆盖未知、历史首差因果链 OPEN、seed2036 非逐位续跑、旧五份违规权重和旧配对结果均保持原结论。本轮短 replay 不消除这些边界。
- 最终合同复核补充了 selection wall-time 边界、将可补充的来源证明与不可变data identity解耦，以及显式预检查整个预选集合的 `PREFLIGHT_SMOKE` 用途。首轮真实8次更新已通过且保留；首轮读者确实检查train角色/gradient-preparation，但没有单独调用smoke用途检查，不把后加护栏倒算为首轮已执行。最终源版本据同一预登记窗口重跑有界矩阵、独立reload和fresh初态，所有重复更新计入同一个24次预算。最终次数以账本为准，不仅报告末轮。

后续真实模型/文件验收若失败，以 append-only PROCESS/OPTIMIZER/GPU 账本及不可变 attempt JSON 为准；最终 REVIEW 应解释新增失败，不能只展示最后一次 PASS。
