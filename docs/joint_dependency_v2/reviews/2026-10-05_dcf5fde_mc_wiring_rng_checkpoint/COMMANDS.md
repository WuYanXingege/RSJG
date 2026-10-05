# 实际命令与重跑记录

工作目录 RSJG_JDV2_clean；Python ../.conda/rsjg/bin/python -B。测试环境 CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1，torch threads=1。工具返回 stdout 后用 apply_patch 归档，不通过 shell 重定向写源码/报告。仅测试运行时在 pytest /tmp 目录生成 toy checkpoint。

1. read-only：git status/rev-parse、rg AGENTS.md、读取 prompt、必读审查/纯函数/测试及 model/parser/trainer；无真实 artifact 读取。
2. apply_patch 预登记 AUDIT_PLAN.json，随后新增接线/helper/测试。
3. 原 92 项仅运行一次：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_cb5160f_expected_conditional_cpu/run_cpu_audit.py
```

输出 PURE_LOSS_RUN.json，92 passed。

4. 新 runner 无参数运行默认 selectors（新测试 + stopping-state 两项 + parser seed + cross-stage restart）。首轮59项，33 pass/26 fail；输出超出工具上限，保留 CPU_RUN_1_TRUNCATED.txt 与明确 derived 的 SUMMARY。不能当完整原始 JSON。
5. 修正 runner Path guard 和 parser fixture 后，仅重跑：
test_parser_and_eval_reject_before_draw、
test_checkpoint_matrix_transactional、
test_objective_roles_eval_and_legacy_matrix、
test_atomic_failure_policy[directory_fsync]、
test_jdv2_resume_stopping_state.py::test_interrupted_resume_matches_uninterrupted_early_stop。
完整 selectors/命令参数保存在 CPU_RUN_2.json，26 passed。
6. 加强语义校验并补充 all-active、flush、fresh/deferred 恢复等后：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_dcf5fde_mc_wiring_rng_checkpoint/run_cpu_audit.py tests/test_jdv2_mc_wiring.py
```

CPU_RUN_3.json：当时68项全通过。独立 child 由 test_separate_process_exact_resume 启动同一 Python，运行 tests/mc_wiring_fixture.py <pytest tempdir>；不是实际 train 命令。
7. 最后只运行新增 test_post_draw_failure_never_reseeds_or_counts_update，CPU_RUN_4.json，两项全通过。
8. verify_scope.py 进行 AST/逐字源码对照，输出 SOURCE_SCOPE.json；git diff --check；JSON/哈希只读核验。无其他数据集/模型回归。
9. 显式 git add 本轮6个源码/测试文件、本归档和 reviews/README.md；普通 commit/push，不 force。固定完整 SHA 远端回读全部本轮文件，比较 UTF-8 bytes、Git blob SHA、SHA256。最终 SHA 和远端核验结果由交付回复提供；若推送失败则明确报告，不能由本计划文字推断成功。

没有执行 GPU、训练、clean-base、geometry6、V4、SDD 或 Skills 操作。历史结果保留。
