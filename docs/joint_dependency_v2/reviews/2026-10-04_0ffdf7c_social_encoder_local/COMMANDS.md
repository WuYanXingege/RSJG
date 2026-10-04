# 执行命令

均在RSJG_JDV2_clean仓库根目录执行。授权prompt SHA256：`7561e086e78f3622c05a865e80194ebeb93c40b6ccfe9ed673c51cf22d6e7572`。

GPU前CPU验收及来源检查（没有真实encoder forward）：

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_social_local_cpu.py
```

随后仅提交本次独立工具，实际probe源码为 `0ffdf7c35624e2ecc9e446bc0ee0cd9f3ac1e25a`。按如下顺序分别启动、等待退出再启动下一个；没有重复启动或自动重试：

```bash
../.conda/rsjg/bin/python -B tools/jdv2_social_local.py --process U0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_social_local.py --process U1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_social_local.py --process T0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_social_local.py --process T1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
```

共16次原encoder调用，含全部冷启动和执行前缀；完整A/encode/sampler/diffusion为0。脚本固定唯一新目录/四进程名额，不接受更多次数。所有进程均保持同一初始化、P2输入、原FP32区段、输入/model/RNG批次外认证及四份h延迟转存；仅T0/T1第4次注册有限模块hook。

完成后CPU比较：

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_social_local_analyze.py --output docs/joint_dependency_v2/reviews/2026-10-04_0ffdf7c_social_encoder_local
```

已有输出受no-clobber保护，重做分析须使用新的输出目录。归档阶段给CPU分析器补充了RESULTS中的source、per-call、settings和batch-validation字段，实际在 `/tmp/rsjg-social-cpu-analysis.ZsP0yK` 重做纯CPU比较，再以新JSON更新本轮尚未提交的报告；原比较字段不变，其余生成文件逐字节相同。GPU probe实际运行源码仍为上列0ffdf7c，未重跑GPU。最终CPU分析器版本随归档commit提交。

本轮另有只读JSON统计汇总、文件hash/JSON语法/工作区检查，以及GPU owner查询；没有额外forward或子模块重放。
