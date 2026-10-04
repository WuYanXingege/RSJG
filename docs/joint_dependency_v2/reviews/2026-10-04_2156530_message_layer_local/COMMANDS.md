# 执行与复核记录

GPU 工具在提交 `2156530d2d7bc15e4b16c53d7e1eb4120ad2dd38` 下运行，之后未修改工具。运行目录为仓库 RSJG_JDV2_clean；Python 为外层 `../.conda/rsjg/bin/python`。

## 独立源码

- [observer](../../../../tools/jdv2_message_observer.py)：原 forward frame 与父模块有限观察。
- [diagnostic runner](../../../../tools/jdv2_message_diagnostic.py)：认证、四槽账本、T1 条件原语诊断。
- [CPU fixture checks](../../../../tools/jdv2_message_cpu.py)：8 类验收，无真实模块 forward。
- [CPU analyzer](../../../../tools/jdv2_message_analyze.py)：完整输出与归档汇总。

## 实际命令

CPU 验收执行两次（首次通过后补充初始化 failed/unknown 账本，再次通过）。最终 CPU_CHECKS 记录匹配 GPU 使用的 observer/runner/CPU 源码 SHA256。未在 CPU 验收中执行真实 message forward。

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_message_cpu.py
git add tools/jdv2_message_observer.py tools/jdv2_message_diagnostic.py tools/jdv2_message_cpu.py tools/jdv2_message_analyze.py
git commit -m 'audit: add bounded original message-layer diagnostic tools'
```

下面四条实际顺序执行，各仅一次；每条终态 COMPLETE 后才执行下一条：

```bash
../.conda/rsjg/bin/python -B tools/jdv2_message_diagnostic.py --process U0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_message_diagnostic.py --process U1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_message_diagnostic.py --process T0 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
../.conda/rsjg/bin/python -B tools/jdv2_message_diagnostic.py --process T1 --gpu-uuid GPU-078a9326-675c-cb94-af4a-ac46eb77c484
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_message_analyze.py --output docs/joint_dependency_v2/reviews/2026-10-04_2156530_message_layer_local
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader
```

末条只读查询 exit 0、输出为空。另做 CPU 聚合断言：4 个 COMPLETE；设置与模型状态完全一致；所有验证字段通过；全部 native roundtrip/有效快照通过；两次额外调用入口记录完全一致；source/target 共享 edge_index storage、offset 0/36；累加终态版本 2；88 个本地 artifact 摘要一致；src/configs 相对 a13eae9 无差异。全部通过。

## 发布

只提交本次四个工具、此新归档目录及审查总索引，普通非 force push 到 research/joint-dependency-v2-clean。固定归档 commit 的 REVIEW.md 和 RESULTS.json 通过 GitHub 文件接口取回完整 UTF-8 字节，与本地逐字节比较并核对 SHA256；具体发布 commit 与回读结果在最终交付提供。此发布说明不替代实际远端回读。
