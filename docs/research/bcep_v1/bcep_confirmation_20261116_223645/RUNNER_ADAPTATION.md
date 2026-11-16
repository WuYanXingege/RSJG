# 三用途确认的最小适配

不修改 bcep/、bcep_control/、selector_control/。以固定提交074d54b的 bcep/pilot_runner.py 为基础，在 bcep_confirmation/runner.py 中仅作：

- 模型工厂选择既有 BCEP / MDNProvider；新初始化哈希记录。
- 独立 schema/identity；dev_indices 显式改名 selection_dev_indices。
- dataset[index] 改为用途守卫 get(index, train/selection_dev)；确认录制在读取 NPZ 之前拒绝访问。
- 异常时保留完整状态并退出；不自动重试数值失败。原 optimizer、scheduler、场景游标、RNG、预算、patience 和选模公式保持原逻辑。

数据适配器只读取当前用途许可的 NPZ 字段。原 SceneManifestDataset 会一次物化所有字段，不能用于要求“导出阶段不读确认 future”的接口；新适配没有更改坐标、地图、样本、损失或增强。

原连续/中断恢复测试 fixture 用于两个工厂，逐项比较模型、optimizer、scheduler、RNG、顺序、游标、epoch、best、stale 与累积量；确认 truth 替换不改变训练/选模结果。CPU预检结果见CPU_PREFLIGHT.json。

录制身份依据：源manifest七个规范注释序列、原导出器直读规则及源文件SHA。UCY序列/地图区别参照[OpenTraj原始数据说明](https://github.com/crowdbotp/OpenTraj/blob/master/datasets/UCY/README.md)和[TrajNet转换入口](https://github.com/vita-epfl/trajnetplusplusdataset/blob/master/trajnetdataset/convert.py)。未另下载原视频做取证；本轮仅称“留出录制序列确认”，不称未见地点或未经历史暴露的盲测。Zara三个录制共享静态语义地图，并不据此合并为一个视频。

默认训练开关仍关闭。唯一执行入口必须同时给出已锁定PLAN和 --execute-authorized-four-runs；run claim和文件锁拒绝重复启动。最多四个矩阵run，结束后仅进行固定一次确认，无后续自动队列。
