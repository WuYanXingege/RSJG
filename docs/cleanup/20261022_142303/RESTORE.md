# 恢复说明

## 可再生缓存

本轮删除的缓存没有需要恢复的研究数据。运行对应模块或测试时，Python 会重新
创建 `__pycache__`，pytest 会重新创建 `.pytest_cache`。论文工作区可用：

```bash
cd /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_EBJD_Paper
/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.conda/rsjg/bin/python -m pytest -q
```

## 隔离项

本次没有新增隔离项，因此没有待恢复路径。既有
`/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/.cleanup_quarantine/20261013`
未被本轮修改。

未来若经审核隔离 Git worktree，恢复必须使用：

```bash
git -C /media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG/RSJG_JDV2_clean \
  worktree move <隔离区中的worktree> <原绝对路径>
```

恢复前必须确认目标路径不存在；不得使用 `--force` 覆盖现有目录。
