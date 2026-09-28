# SDD Baseline Commands

本文整理 RSJG 仓库中原始 GDTS SDD baseline 的数据预处理、训练、恢复和推理命令。
以下命令均从仓库根目录执行：

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean
```

固定环境和输出位置：

```bash
export PYTHON_BIN=/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python
export RUN_NAME=gdts_baseline_sdd_seed2035
export CACHE_ROOT=outputs/joint_dependency_v2/cache/baseline_source_batches
```

实验采用 SDD 30/17 `validate_on_test` 协议、像素坐标、seed 2035、K=20、
300 epochs、Adam 1e-4、ExponentialLR、batch size 64 和 FP32。由于 validation
与 test 使用相同的 17 个场景，best-checkpoint 结果不是独立 held-out model
selection。

## 1. 推荐入口

### 1.1 仅构建完整压缩 cache

```bash
bash tools/run_sdd_baseline.sh pre-process
```

cache 位置：

```text
outputs/joint_dependency_v2/cache/baseline_source_batches/
  sdd/sdd/data_batches_zstd_v1/
```

完整 cache 必须同时存在：

```text
finished_train_batches.txt
finished_valid_batches.txt
finished_test_batches.txt
cache_manifest.json
```

### 1.2 已有完整 cache，从头训练

```bash
bash tools/run_sdd_baseline.sh train
```

### 1.3 一次执行预处理、训练和五次最终推理

```bash
bash tools/run_sdd_baseline.sh all
```

`all` 依次执行 `pre-process` 和 `train_test`。若完整 cache 已存在，预处理器会
只读复用，不会重新构建。

### 1.4 从编号 checkpoint 恢复训练

例如从 epoch 10 恢复：

```bash
bash tools/run_sdd_baseline.sh resume 10
```

恢复前必须确认文件存在：

```text
output/sdd/runs/gdts_baseline_sdd_seed2035/saved_models/epoch_010.pt
```

CLI 当前只接受 `best` 或正整数 epoch。虽然每个完整 epoch 都会更新
`last_model.pt`，但不要执行 `--load_checkpoint last`；该字符串不是受支持的
CLI checkpoint 标识。编号 checkpoint 默认每 10 epochs 保存一次。

### 1.5 使用最佳 checkpoint 做五次推理

```bash
bash tools/run_sdd_baseline.sh test
```

该命令等价于 `phase=test`、`load_checkpoint=best`、`num_test_runs=5`。

## 2. 低资源后台运行

SDD cache 较大。桌面工作站建议限制 CPU affinity，并降低 CPU 和 I/O 调度
优先级：

```bash
mkdir -p output/sdd/runs/gdts_baseline_sdd_seed2035

nohup setsid nice -n 10 ionice -c2 -n7 taskset -c 0-3 \
  env PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      OPENBLAS_NUM_THREADS=4 \
      NUMEXPR_NUM_THREADS=4 \
      PYTHON_BIN="$PYTHON_BIN" \
      RUN_NAME="$RUN_NAME" \
      CACHE_ROOT="$CACHE_ROOT" \
  bash tools/run_sdd_baseline.sh all \
  > output/sdd/runs/gdts_baseline_sdd_seed2035/run.log 2>&1 \
  < /dev/null &

echo $! > output/sdd/runs/gdts_baseline_sdd_seed2035/run.pid
```

只进行最佳 checkpoint 推理时，将上述命令中的 `all` 改成 `test`。

不要同时启动两个使用相同 `RUN_NAME` 或 `CACHE_ROOT` 的任务，否则它们可能
同时写入相同 cache、config 或 checkpoint。

## 3. 原始完整命令

### 3.1 数据预处理

```bash
"$PYTHON_BIN" main.py \
  --phase pre-process \
  --dataset sdd \
  --test_set sdd \
  --goal_model_type independent \
  --training_stage baseline \
  --run_name "$RUN_NAME" \
  --reproducibility True \
  --seed 2035 \
  --validation_seed 2035 \
  --num_epochs 300 \
  --batch_size 64 \
  --start_validation 5 \
  --validate_every 20 \
  --learning_rate 0.0001 \
  --optimizer Adam \
  --scheduler ExponentialLR \
  --skip_ts_window 1 \
  --down_factor 8 \
  --num_workers 1 \
  --data_augmentation True \
  --num_samples 20 \
  --num_test_runs 5 \
  --amp_enabled False \
  --amp_dtype fp32 \
  --compress_batch_cache True \
  --batch_cache_root "$CACHE_ROOT" \
  --use_wandb False
```

只有在确认需要删除并完整重建现有 cache 时，才额外添加：

```bash
--force_reprocess True
```

它会删除目标 cache 后从头生成，不应用于普通恢复。

### 3.2 从头训练

将上面命令中的 phase 改为：

```bash
--phase train
```

训练输出位置：

```text
output/sdd/runs/gdts_baseline_sdd_seed2035/
```

### 3.3 从编号 checkpoint 恢复

使用与预处理完全相同的参数，并增加：

```bash
--phase train --load_checkpoint 10
```

### 3.4 最佳 checkpoint 推理

使用与预处理完全相同的参数，并增加：

```bash
--phase test --load_checkpoint best --num_test_runs 5
```

## 4. 状态检查

实时查看日志：

```bash
tail -f output/sdd/runs/gdts_baseline_sdd_seed2035/run.log
```

如果本次任务使用 `resume.log`，则执行：

```bash
tail -f output/sdd/runs/gdts_baseline_sdd_seed2035/resume.log
```

查看进程树：

```bash
pgid=$(cat output/sdd/runs/gdts_baseline_sdd_seed2035/run.pid)
ps -eo pid,ppid,pgid,sid,etime,pcpu,pmem,stat,cmd | \
  awk -v pgid="$pgid" '$3 == pgid || $1 == pgid'
```

查看 GPU：

```bash
nvidia-smi
```

检查 cache 完成标记：

```bash
cache="$CACHE_ROOT/sdd/sdd/data_batches_zstd_v1"
ls -l "$cache"/finished_*_batches.txt "$cache"/cache_manifest.json
```

## 5. 指标边界

该 baseline 的原始 cache 按独立轨迹组成 batch，适用于上游 SDD pixel
ADE/FDE。它不保存严格同步的多人 scene windows，因此不能从该训练任务直接
报告严格的 JADE/JFDE、CRmean 或 CRJADE；这些指标需要另建同步评估协议。
