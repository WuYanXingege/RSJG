# 本轮命令与记录

工作目录RSJG_JDV2_clean，解释器../.conda/rsjg/bin/python -B；输入HEAD28728bf，分支research/joint-dependency-v2-clean。先检查AGENTS/HEAD/status；不reset/clean，保留既有未跟踪outputs/data。

## 已执行

1. 源码/旧metadata只读：git status/diff/show、rg、sed；原windows JSONL/hash解析，无原始pickle/旧checkpoint反序列化。
2. metadata生成：`../.conda/rsjg/bin/python -B tools/p2_metadata_plan.py --output docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu`。首个相对路径生成尝试失败见GENERATION_ATTEMPT_1；修复output.resolve后metadata生成成功，不覆盖旧归档。
3. 首轮CPU命令见CPU_RUN_FIRST；第二次记录失败见CPU_RUN_SECOND_RECORDING_FAILURE及RECOVERED。后续全部精确test argv逐次保存在CPU_RUN_CAP_REPAIR/REGRESSIONS/ADDITIONAL/FINAL_CONTROL/LAST_GUARDS/THIRD.json。环境均CUDA_VISIBLE_DEVICES空；父进程runner的真实网络/CUDA调用护栏，CPU Adam健康检查单独处理。
4. 旧相关测试仅一次范围回归：test_jdv2_clean_split_protocol、test_jdv2_mc_wiring、test_sdd_goal_pretrain_protocol、test_sdd_lightweight_scene、test_jdv2_stage_a_freeze_contract；filter排除all_320_manifest、real_logical、clean_stage_b_scientific、checkpoint_hash_and_strict、global_default_and_alternative、degree_zero_identity。这些涉及真实payload/checkpoint/真实神经模块构造，不符合本轮范围。SDD stub失败只定向复测。
5. pytest --collect-only仅清单；源/loss AST对照及旧archive所有hash复核；`finalize_evidence.py`汇总tmp JSON、记录丢失恢复边界，并实际运行四个main --p2_launch_check。它不执行pytest测试、训练或模型；只收集用例。
6. 来源检索：UCY页面失败；CMU/Google/NVlabs固定commit+file由GitHub MCP读取。短证据/访问范围/失败在RECORDING_AND_MAP_PROVENANCE；未下载annotation/视频。
7. git diff --check；显式stage本轮源码/tests/tools/archive与reviews README，普通commit/push（无force）。推送后以完整SHA逐文件GitHub readback，核对bytes/SHA256/Git blob，receipt另存本地/tmp（避免自包含commit循环）。

## 重要执行例外

旧MC子进程第一次范围回归只拦截_lazy_init，未拦截CPU Adam内部的is_available。推定6次availability调用，底层device枚举/驱动初始化UNKNOWN，不能写0。随后强化子进程CUDA guard与checked CPU health no-op，定向跨进程resume复测成功。首轮无更新；第2/3轮各6次父进程查询被guard拦截。所有历史失败保留，不称全轮严格零查询。

## 未执行

真实模型/GPU smoke、真实更新/训练/outer指标、protected target读、obs真实分离、新真实cache/bank、五checkpoint重复清点、SDD停止/恢复、Skills/Apps修改均未执行。未来训练须新授权；配置null不能填旧违规base顶替。
