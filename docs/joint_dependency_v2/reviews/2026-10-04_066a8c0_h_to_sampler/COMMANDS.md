# 执行与复核记录

## 源码版本

GPU实际源码为066a8c007c887f913805ae6d3ce719fac1ae0acb，在CPU验收后、GPU初始化前提交。

- [输入合同/注入](../../../../tools/jdv2_h_sampler_contract.py)
- [诊断/观察器](../../../../tools/jdv2_h_sampler_probe.py)
- [CPU stub](../../../../tools/jdv2_h_sampler_cpu.py)
- [全调用分析器](../../../../tools/jdv2_h_sampler_analyze.py)
- [运行后CPU复核](../../../../tools/jdv2_h_sampler_postcheck.py)：GPU后新增，仅比较已保存payload，无模型重放/新求解。

## 实际命令

仓库工作目录RSJG_JDV2_clean。独立来源认证一次，CPU stub一次，真实model/sampler forward均0；CPU_CHECKS记录四个GPU前工具的SHA256。

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_h_sampler_cpu.py
git add tools/jdv2_h_sampler_contract.py tools/jdv2_h_sampler_probe.py tools/jdv2_h_sampler_cpu.py tools/jdv2_h_sampler_analyze.py
git commit -m 'audit: add bounded fixed-h sampler propagation diagnostic'
../.conda/rsjg/bin/python -B tools/jdv2_h_sampler_probe.py --process P0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_h_sampler_probe.py --process P1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_h_sampler_analyze.py --output docs/joint_dependency_v2/reviews/2026-10-04_066a8c0_h_to_sampler
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_h_sampler_postcheck.py
```

P0完成后才启动P1，各固定4次；末次compute查询exit0且输出为空。中间只读查看P0已保存的三组IDs对照，未改变P1顺序/预算。

Postcheck完整输出保存POST_CHECKS.json，并通过patch原样加入RESULTS.post_run_checks；原分析器未修改。该复核证明28对非h入口/mask/goals一致、跨进程模型/全部buffers/设置一致、全局RNG不变、实际generator与P2匹配、固定h storage和version1–4，并补齐逐轮score摘要。不触发CPU exact replay，额外求解0。

原sampler内部144次CPU exact求解属于8个原片段。原evaluator初始化包含缓存只读校验及中间legacy初始化，最终A strict load；没有从loader枚举评估窗口或运行完整encode/forward。

## 发布

只提交本轮工具、新审查目录及总索引，普通非force push到research/joint-dependency-v2-clean。固定归档commit的REVIEW.md/RESULTS.json经GitHub文件接口回读完整UTF-8字节，与本地逐字节和SHA256核对。实际发布commit与回读状态以最终交付为准，不以本说明代替成功证据。
