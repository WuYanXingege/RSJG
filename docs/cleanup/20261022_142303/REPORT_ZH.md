# RSJG 项目第一阶段清理报告

状态：**COMPLETE_WITH_CONSERVATIVE_KEEP**。

本轮完成了依赖检查、可再生缓存清理、旧目录隔离判定和清理后验证。没有旧
开发目录达到安全隔离门槛，因此没有为了“看起来更干净”而移动任何含研究证据
或 Git 状态的目录。

## 1. 已永久删除

仅删除清单中 81 个 Python/pytest 可再生缓存目录：

- `__pycache__`
- `.pytest_cache`

逐项路径与执行前大小见
[`CACHE_DELETE_ALLOWLIST.tsv`](CACHE_DELETE_ALLOWLIST.tsv)。删除缓存实际占用
`10,084,352` B；计入新增报告后，文件系统可用空间净增 `10,047,488` B。

## 2. 已移入隔离区

无。所有旧目录至少含一个保护条件：Git worktree、未提交内容、未推送提交、
共享 Git 元数据、checkpoint/数据/预测 bank/正式日志，或用途仍不确定。

既有 `.cleanup_quarantine/20261013` 未修改，当前占用 `448,445` B 表观空间。

## 3. 关键保留项

- `RSJG_EBJD_Paper`：唯一论文工作区。
- `RSJG_EBJD_assets`：论文真实数据、manifest、checkpoint 与参考资产。
- `RSJG_JDV2_clean`：存放所有 worktree 的 Git common-dir，同时有大量未跟踪
  研究输出，绝不能整体移动或删除。
- `RSJG_EBJD_clean`：分支领先远端 1 commit，并保存大量训练资产。
- ETH/UNIV/V2/JDV2 各 worktree：存在无 upstream 分支、未提交报告或历史实验
  复现价值。
- `RSJG_EBJD_diagnostic_assets`：完整预测 bank、checkpoint、正式评测与失败实验
  证据。
- `GDTS` 与 `GDTS_official_297d508_HOTEL`：旧入口和 worktree 仍通过软链接引用，
  且含原数据与 matched baseline 证据。
- `.conda/rsjg`：当前验证环境，未清理。

完整分类与原因见 [`CLEANUP_MANIFEST.md`](CLEANUP_MANIFEST.md)。

## 4. 清理后状态

- 论文代码可从新工作区独立导入。
- 配置和三条资产入口可访问，无新增失效链接。
- 三份登记参考 checkpoint 哈希和 strict-load 全部通过。
- Git 分支 `paper/ebjd-final-v1`、HEAD、common-dir 与 13 个 worktree 正常。
- 核心代码和配置没有被清理动作修改。
- 无 GPU 工作、无训练、无评测重跑、无进程干预。

## 5. 待审核的永久删除清单

**当前为空。** 本轮没有新隔离项，因此没有可安全建议立即永久删除的旧目录。
既有隔离区也未在本轮重新认证为可删除。

若希望继续释放大空间，下一阶段必须先完成以下前置工作，再逐个提交审核：

1. 推送或归档无 upstream/领先远端的研究分支；
2. 将独有 checkpoint、预测 bank 和正式日志做哈希化资产迁移；
3. 处理 `RSJG_JDV2_clean` 中 55 个未跟踪项和其他 dirty worktree；
4. 另行迁移 Git common-dir 后，才能讨论 JDV2 主工作区；
5. 单独审计既有 `.cleanup_quarantine/20261013`。

在完成这些条件前，本报告建议所有旧目录继续 `KEEP`。

