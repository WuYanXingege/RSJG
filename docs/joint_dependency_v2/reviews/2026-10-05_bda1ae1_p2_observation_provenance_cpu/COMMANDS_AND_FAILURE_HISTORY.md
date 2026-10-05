# 执行命令与失败轨迹

工作目录 RSJG_JDV2_clean；解释器 ../.conda/rsjg/bin/python -B。所有 tensor/metadata 计算脚本以 CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1 启动；guard还将NUMEXPR限制为1并设置Torch intra/inter-op线程各1，原静态图转换将OpenCV设为1。无nvidia-smi，无SDD操作。

数据四阶段实际峰值RSS由resource.getrusage保留；早期测试/metadata脚本未保留各自RSS，不能声称测得全进程同时峰值。RESULTS单列数据进程实测峰值与并行额外2 GiB保守估计，后者不是实测；工作进程调度最多2、各1计算线程。预算在导出操作之间检查，不是全进程cgroup硬隔离。

工具实际四阶段按 plan→固定canary→export→verify执行；相同参数：

```text
tools/p2_observation_artifacts_cpu.py <phase>
 --enable-observation-preparation --device cpu
 --manifest docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu/P2_RUNTIME_MANIFEST.json
 --manifest-hash a768b1765d4345b17e450df04d81035f1407ee3338079dfbad1730b7a1f751b5
 --archive docs/joint_dependency_v2/reviews/2026-10-05_bda1ae1_p2_observation_provenance_cpu
 --output outputs/joint_dependency_v2/p2_observation_bda1ae1_20261005
```

每阶段完整执行计数、进程护栏、起点/耗时、RSS及错误保存在 RUN_<phase>_<time_ns>.json；固定8canary的集合/hash在生成前写出。export复用8个认证canary，生成其余4241；verify不生成、不查询坐标，仅重新回读4249份独立obs。

## 测试失败与定向重跑（不抹平）

1. CPU_TESTS_1_SETUP_FAILURE.json：pytest插件hook参数写成r而非report，在注册时失败，执行测试体0；该原进程guard快照未保留，不能补写成零查询。
2. CPU_TESTS_2.json：33通过、5失败。PyTorch拒绝以隐藏临时路径字符串序列化；失败全部为合成artifact测试，真实canary尚未执行。
3. 修复为打开二进制file handle传给torch.save，保留flush/fsync/atomic replace，并检查异常清理。CPU_TESTS_3.json：6通过（5个原失败项+原子失败清理）。RESOURCE_PLAN保存的是修复前producer hash，真实canary及后续artifact绑定修复后hash，不回填历史plan。
4. CPU_TESTS_4.json：6通过（raster外原公式行为+4个Python CUDA/1个native查询故意拒绝）。
5. CPU_TESTS_5.json：6通过（真实UNKNOWN outer-provider阻断、manifest/重复成员、3类重新hash的错误wrapper、资产绑定）。
6. CPU_TESTS_6.json：最终完整50/50通过。本次单独完整运行不替代前五次历史。故意CUDA查询在第4和第6次各5次，合计10次，全部拦截；guard覆盖内转发0。

run_tests.py记录精确pytest argv/node IDs及失败栈；最后唯一覆盖集在TEST_AND_PROCESS_LEDGER.json。没有重跑旧optimizer/MC完整套件，不把上一轮157项合并进本轮50项。

prepare_evidence.py为CPU metadata/hash提案脚本；补入网络HEAD重试记录后再次运行以刷新引用hash。第一次同内容提案的独立guard快照已被本轮该metadata文件覆盖，不计成另一份独立保留进程记录。两个执行均没有坐标/payload读取；本轮最终账本对未保留的第一次metadata快照明确保留覆盖例外。

finalize_results.py在export与verify完成后只读取metadata，实际调用现有parser的launch-check分支；不调用模型、设备或权重。归档后仅暂存本轮src/tool/test与审查文本metadata，正常commit/push，不force；远端按完整commit回读每个变更文件的bytes/SHA256并核对Git blob SHA1，回读收据保留本地，避免自引用commit/hash。
