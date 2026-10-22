# 清理执行记录

## 动作

1. 冻结清单后重新检查 81 个路径：全部位于 RSJG 根内，basename 仅为
   `__pycache__` 或 `.pytest_cache`，均为普通目录，分类均为
   `DELETE_CACHE`，且执行时体积与清单一致。
2. 按 `CACHE_DELETE_ALLOWLIST.tsv` 的精确绝对路径逐项删除；没有使用跨目录
   通配符，没有执行 `git clean`、`git gc`、`reset` 或 `stash`。
3. 删除后 81 个路径全部不存在。
4. 未移动任何目录；未创建新隔离区；未修改既有 `.cleanup_quarantine`。

## 空间计量

- 删除缓存表观体积：`8,363,033` B。
- 删除缓存实际分配块：`10,084,352` B。
- 文件系统可用空间：`163,830,874,112` B → `163,840,921,600` B。
- 可用空间净增：`10,047,488` B。
- 差额来自新增清理报告本身和文件系统计量粒度；隔离移动为 0 B，未混入
  释放空间。
- 既有隔离区保持 `448,445` B 表观 / `487,424` B 分配块。

## 验证

- `ebjd` 实际导入位置：论文工作区的 `ebjd/__init__.py`。
- architecture/training/HOTEL/UNIV YAML 均可解析。
- `assets/data`、`assets/checkpoints`、`assets/references` 均解析到持久资产，
  且目标存在。
- 项目内（排除受保护 conda 环境）失效软链接：0。
- HOTEL epoch45、UNIV epoch73、UNIV Control epoch81 checkpoint 均重新计算
  预期 SHA256，并以 `strict=True` 加载；307 个张量逐值一致，状态 PASS。
- Git common-dir、分支、HEAD 及全部 13 个登记 worktree 正常。
- 清理前已有核心代码、配置、README、环境和结果文件没有差异；新增内容仅为
  `docs/cleanup/20261022_142303/`。
- 验证使用 `PYTHONDONTWRITEBYTECODE=1` 和 Python `-B`，没有重新生成已删缓存。
- GPU 使用：0；未启动训练或正式评测。

## 未执行

- 未永久删除旧工作区、日志、checkpoint、数据、预测 bank 或隔离区。
- 未移动 Git worktree。
- 未停止、重启或杀死进程。
- 未清理 `.conda`、用户主目录、系统缓存或其他项目。
- 用户 crontab 与 user systemd bus 因当前权限不可读；因此相关不确定目录继续
  `KEEP`，没有据此扩大清理。

