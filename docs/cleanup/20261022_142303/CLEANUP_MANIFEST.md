# RSJG 清理清单（执行前冻结）

- 时间戳：`20261022_142303`
- 项目根：`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG`
- 原则：不确定即 `KEEP`；不删除旧工作区、研究资产或历史证据。
- 磁盘执行前：已用 `336,588,001,280` B，可用 `163,830,874,112` B。

## DELETE_CACHE

共 81 个目录、`8,363,033` B。逐项绝对路径、类型、执行前体积和恢复方式见
[`CACHE_DELETE_ALLOWLIST.tsv`](CACHE_DELETE_ALLOWLIST.tsv)。它们仅为
`__pycache__` 或 `.pytest_cache`，不含 checkpoint、数据、日志、结果或 Git
对象；没有进程打开这些路径。拟执行动作是仅按清单中的精确路径逐项删除。

以下逐项元数据对 allowlist 的每一行均成立：条目本身不是 Git worktree（部分
父目录是已登记 worktree）；条目不是软链接且内部没有软链接；不含未提交源码、
研究资产或独有证据；删除后通过重新导入模块或重跑 pytest 恢复。因此没有把
父 worktree 的保护状态错误地扩展为“缓存不可删除”，也没有触及父目录的其他内容。

## QUARANTINE

本轮无项目满足隔离条件。原因不是“没有旧目录”，而是所有候选至少命中
以下一项保护条件：已登记 Git worktree、未提交内容、未推送提交、共享 Git
公共目录、独有 checkpoint/预测 bank/正式日志，或用途无法完全确认。

因此不会创建新的隔离目录；既有 `.cleanup_quarantine/20261013` 保持原状。

## KEEP：顶层目录

| 绝对路径 | 执行前体积 | Git worktree | 软链接/修改/研究资产 | 保留理由与恢复策略 |
|---|---:|---|---|---|
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_Paper` | 1.1 MB | 是 | 三个资产链接；清理报告将形成修改 | 受保护的唯一论文工作区；不得隔离 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_assets` | 452 MB | 否 | 数据、manifest、checkpoint、参考结果 | 论文运行的独立持久资产；不得隔离 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean` | 171 GB | 是 | 55 个未跟踪项、38,026 个 checkpoint、审计和输出 | Git common-dir 位于其 `.git`；移动会破坏全部 worktree |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_clean` | 12 GB | 是 | 分支领先远端 1 commit；277 checkpoints、30,684 arrays | 未推送提交及训练/评测资产 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_diagnostic_assets` | 6.0 GB | 否 | 139 checkpoints、2,018 arrays、77 logs | 预测 bank、失败实验与正式评测证据，尚未完整迁移 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/GDTS` | 97 GB | 否 | 根 `data` 链接及多个 worktree依赖；9,131 checkpoints | 原始数据、官方资源与历史模型依赖 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/GDTS_official_297d508_HOTEL` | 21 GB | 否 | 两个链接指向 `GDTS`; 26 checkpoints | matched baseline 与复现证据 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda` | 7.4 GB | 否 | 正在使用的 `rsjg` 环境 | 明确保护的项目环境；不清理其缓存或链接 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_coverage_repair` | 132 MB | 是 | 本地分支无 upstream | 未推送研究分支与复现代码 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_goal_proposal_triage` | 133 MB | 是 | 本地分支无 upstream | 未推送研究分支与复现代码 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_native_goal_pilot` | 144 MB | 是 | 本地分支无 upstream | 未推送研究分支与历史变体证据 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_rollout_boundary` | 141 MB | 是 | 2 个未跟踪报告目录；无 upstream | 未提交证据与未推送分支 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_targeted_diagnostic` | 140 MB | 是 | 本地分支无 upstream | 未推送诊断分支 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_eth_worktree` | 140 MB | 是 | 本地分支无 upstream | ETH 正式训练历史分支 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_no_token_path_pilot` | 141 MB | 是 | 已登记 worktree | 失败实验必要复现证据 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_univ_no_token_transfer` | 142 MB | 是 | 已登记 worktree | HALF/迁移历史复现证据 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_univ_worktree` | 139 MB | 是 | 本地分支无 upstream | UNIV 正式训练历史分支 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_v2_integrated` | 141 MB | 是 | 已登记 worktree | 失败迁移和兼容性证据 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_hotel_a0_a1_5b6e4e4` | 130 MB | 是（detached） | 2 个未跟踪链接/目录 | 未提交历史结果；且链接指向 JDV2/GDTS |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.cleanup_quarantine` | 476 KB | 否 | 既有隔离内容 | 用户尚未批准永久删除，保持不动 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/data` | 链接 | 否 | 指向 `GDTS/data` | 旧入口仍有工作树引用，保持 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/{.agents,.codex,.aws,.vscode,.nature-figure*}` | < 32 KB | 否 | 工具/IDE状态，用途未完全确认 | 不确定项默认保留 |
| `/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/{*.md,*.json,*.pdf}` | 约 14 MB | 否 | prompts、报告、论文原文 | 历史决策与复现依据，不因体积小或日期旧而清理 |

`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG` 是上述公共 Git
目录与保护目录的父目录，本身禁止移动或删除。

## Git 与运行依赖边界

- `RSJG_EBJD_Paper/.git` 实际为
  `RSJG_JDV2_clean/.git/worktrees/RSJG_EBJD_Paper`。
- Git common-dir 为 `RSJG_JDV2_clean/.git`。
- 13 个登记 worktree 全部保留；没有使用 `git worktree move`。
- 论文分支执行前为 `c6ed61b7c31c6388f2f6a52b3ca8e2243fea6181`，与远端一致。
- `research/ebjd-clean-v1` 领先远端 1 commit；多个研究分支无 upstream；这些均阻止隔离。
- 扫描时未发现训练、评测或服务进程引用 RSJG 文件；仅当前 Codex 会话 cwd
  位于项目根。
- 用户 crontab 因权限不足无法读取，user systemd bus 不可用；系统 cron 可读范围
  内未发现 RSJG 引用。无法排除的计划任务风险按 `KEEP` 处理。

## 软链接边界

- 论文三条资产链接均解析到 `RSJG_EBJD_assets/{data,checkpoints,references}`，
  不依赖旧工作区。
- 项目内无失效软链接。
- `GDTS_official_297d508_HOTEL`、JDV2 worktree 和根 `data` 仍引用
  `GDTS`/`RSJG_JDV2_clean`，因此相关旧目录继续保留。

