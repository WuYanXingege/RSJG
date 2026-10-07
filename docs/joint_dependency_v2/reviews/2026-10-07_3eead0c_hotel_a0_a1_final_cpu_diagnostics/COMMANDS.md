# Reproduction commands

以下命令只读现有产物并执行 CPU 分析。它们不导入模型，不运行 dataloader、sampler、
diffusion 或 GPU forward。

```bash
rsjg_root="/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG"
rsjg_repo="$rsjg_root/RSJG_JDV2_clean"
rsjg_run="$rsjg_root/RSJG_JDV2_hotel_a0_a1_5b6e4e4"
rsjg_review="$rsjg_repo/docs/joint_dependency_v2/reviews/2026-10-07_3eead0c_hotel_a0_a1_final_cpu_diagnostics"
rsjg_py="$rsjg_root/.conda/rsjg/bin/python"

cd "$rsjg_review" || exit 1
ulimit -v 8388608
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$rsjg_py" audit_cpu.py \
    --main-repo "$rsjg_repo" \
    --run-worktree "$rsjg_run" \
    --review-dir "$rsjg_review"

"$rsjg_py" -m unittest -v test_cpu_math.py
```

队列完成后原样再次运行上面的 analyzer；它会按 `QUEUE_MANIFEST.json` 的真实 result
路径读取 formal artifacts，不改变 finalizer，也不会补跑缺失的推理。只有当所有
`E_joint` 行均非 `PENDING` 时才能应用 A1 gate。

绘图与 QA：

```bash
export MPLCONFIGDIR="/tmp/rsjg_mplconfig"
export PYTHONPATH="/home/ee615/.codex/skills/nature-figure/scripts"
"$rsjg_py" plot_training_dynamics.py \
  --input TRAINING_CURVES.csv --output-dir figures

python3 /home/ee615/.codex/skills/nature-figure/scripts/validate_figure.py \
  plot_training_dynamics.py --backend python --json --strict
python3 /home/ee615/.codex/skills/nature-figure/scripts/audit_pdf_text.py \
  figures/training_dynamics.pdf --min-pt 5 --json
"$rsjg_py" /home/ee615/.codex/skills/nature-figure/scripts/audit_figure_collisions.py \
  figures/training_dynamics.pdf \
  --json-out figures/training_dynamics.collision-audit.json \
  --overlay-pdf figures/training_dynamics.collision-overlay.pdf --strict
```

本轮没有训练命令或 GPU 恢复命令；`NEXT_DESIGN.md` 仅是条件性提案。
