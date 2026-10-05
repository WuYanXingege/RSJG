# 实际CPU命令与环境

日期2026-10-05（Asia/Shanghai）；工作目录RSJG_JDV2_clean。使用项目既有../.conda/rsjg/bin/python，没有安装PyTorch/CUDA或其他依赖。Python3.10.21，PyTorch2.8.0+cu128，pytest9.1.1；线程1。CUDA build=12.8只是wheel元数据；未查询/启动GPU运行。

## 已执行

开始只读：git branch --show-current；git rev-parse HEAD HEAD:src；git status --short；查找适用AGENTS.md（工作路径各父级及repo内未发现）；完整读设计五份Markdown，并解析完整RESULTS.json（SHA256=8f3e033066e87e5668f115705757f41c3cb2e71f818574dcfd8cac79574a5499）。阅读loss和相关测试，不加载bank/checkpoint。

以下同一命令实际运行两次，记录完整pytest phases、输出、梯度误差及MC各分量：

~~~bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_cb5160f_expected_conditional_cpu/run_cpu_audit.py
~~~

第一次：88 passed, 1 warning in 3.08s（runner主体3.320835s）；第二次：92 passed in 2.68s（runner主体2.835318s）。第一次warning是测试断言转scalar，已用detach修正；没有验收失败、重抽seed或放宽阈值。第二次额外测试degree8、多mask两dtype和不隐式归一化。

runner内部仅选新test文件，以及旧test_joint_dependency_v2.py的以下纯loss/日程节点：test_pseudo_likelihood_no_edges_reduces_to_unary_scene_balanced、test_scene_mode_score_and_mixture_match_manual_enumeration、test_mixture_and_posterior_distillation_gradient_ownership、test_relation_kl_and_scene_kl_are_finite_with_empty_edge、test_relation_kl_detaches_scene_responsibility_weight、test_kl_warmup（4参数）、test_teacher_curriculum（5参数），共14项。该旧文件导入模型类但所选节点不实例化/forward它们；runner额外禁止所有nn.Module forward。未跑整个模型集成/GPU suite。

~~~bash
../.conda/rsjg/bin/python -B docs/joint_dependency_v2/reviews/2026-10-05_cb5160f_expected_conditional_cpu/verify_scope.py
git diff --check
git diff -- src/joint_goal_loss.py
~~~

静态核验：原23个定义逐字不变，新src调用者0；保护路径无diff。纯函数和脚本AST/JSON及文档链接检查不调用模型。检查black/ruff是否已装，两者均未装；未安装或格式化旧源码。

## 输出及统计口径

run_cpu_audit.py仅向stdout输出JSON；CPU_RUN_1/2.json通过apply_patch归档原结果，没有脚本写到data/outputs。AUDIT_PLAN.json先于首次运行固定seed527015、1024重复、S4、5SE+1e−10阈值及FP32/64数值容差。

forward_rejected是非法输入负测试命中预期异常，不是test failure。grad_attempted统计测试显式autograd.grad（包括reference），backward_hook_visits统计新loss输出VJP访问（包括gradcheck），不等同模型backward或optimizer steps。

正式发布只stage新增loss、test、本轮archive与reviews索引，普通commit/push；随后以实际full SHA读取每个发布文件，核对字节与SHA256。不会force push或加入既有未跟踪data/outputs。
