# 命令账本

工作目录：RSJG_JDV2_clean；Python：../.conda/rsjg/bin/python。所有本轮脚本CPU-only；无训练/模型/CUDA入口。

## 已执行（不要为归档重复checkpoint读取）

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_clean_objective_preflight.py --output docs/joint_dependency_v2/reviews/2026-10-05_4f11f44_clean_base_split_objective_preflight
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B -m pytest -q -p no:cacheprovider tests/test_jdv2_clean_objective_preflight.py
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B -m pytest -q -p no:cacheprovider tests/test_jdv2_clean_objective_preflight.py -k cross_recording
```

主审计一次成功，5checkpoint各一次；第一次测试28pass，新增两项后仅运行cross_recording得到2pass/28deselected。未重跑172旧测试。
supplement_metadata.py、cross_recording_metadata.py分别在本目录运行一次；stdout JSON经apply_patch保存为SUPPLEMENT_METADATA/CROSS_RECORDING_IDENTIFIERS。它们只读metadata/source/log，不加载torch/模型/checkpoint。跨recording检查是首轮审计后的补充，manifest增加independent_recordings=UNKNOWN注释；不是重新执行checkpoint审计。
另有git status/show/diff、rg/sed、文档读取和只读哈希检查；主EXECUTION不声称覆盖全部交互式read。旧training.log一次shell输出截断，补充脚本完整提取。

归档收尾执行 validate_archive.py 一次，校验JSON/YAML/CSV、4249窗口分片SHA、共同顺序SHA、文档链接及全部生产src/config/main与输入commit字节一致，结果保存SOURCE_SCOPE.json。--hashes模式随后仅计算归档/工具/测试/索引SHA，输出EVIDENCE_MANIFEST.json（自排除），不重跑审计。SOURCE_EXPOSURE_MATRIX的selection字段在文档整理时按具体ETH/UNIV/P1来源细化；其他来源标NO_DIRECT_SELECTION_FOUND/HYPERPARAMETER_HISTORY_UNKNOWN，未新增标签读取。

## 现在可复制的安全拒绝检查

```bash
CUDA_VISIBLE_DEVICES='' ../.conda/rsjg/bin/python -B tools/jdv2_clean_objective_preflight.py --check-plan docs/joint_dependency_v2/reviews/2026-10-05_4f11f44_clean_base_split_objective_preflight/PILOT_PLAN.json
```

预期exit2，TEMPLATE_BLOCKED；只读，不导入torch、不启动训练。pytest已验证此子进程，不需再跑。
工具audit --output只允许新目录且要求CUDA_VISIBLE_DEVICES为空；未来重复audit属于新计数/授权轮次，本轮不会重复读取五份文件。
静态check-plan不是安全的训练launcher，也不做实时GPU检查/完整manifest签名验证；改JSON字段不能据此获得授权。

## 未来命令：NOT_IMPLEMENTED / NOT_EXECUTABLE

当前main/parser实际支持phase=goal_pretrain或train、goal_model_type=independent或joint_dependency_v2、training_stage=joint_goal、run_name、device等；无 --config。
但P2 role loader、保护source预处理、L1 attempt cap/no-validation、L2选择adapter尚未实现。因此这里**不发布假可执行训练命令**，不虚构--max_steps或--config。待上述patch的CPU验收与base/cache/初态SHA齐备，在新run目录生成正式config.yaml并由真实parser校验后，再归档该具体版本的命令。
授权顺序见RESOURCE_AND_DEPENDENCY_PLAN；L1授权不含L2/outer/归因。不要复制A0/A1 blocked YAML到main自动读取的run目录。

## 归档

仅stage本次新工具、测试、审计目录与reviews/README；普通commit，无force。按完整commit经GitHub逐文件base64回读，与git show bytes/SHA256/Git blob SHA一致后在/tmp保留receipt。commit哈希由提交结果给出，不在提交内自引用伪造receipt。
