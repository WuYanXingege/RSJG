# 六层基线恢复收据

八层正式进程、监督器与4个worker已定向停止，epoch81/update20761；最近完整epoch80/update20560，201个未保存更新不声称可恢复。未发现相关用户服务、定时器或cron；at/tmux未安装。

实际删除8项已核实废弃运行的25份权重/optimizer文件，实际分配空间3544657920字节（3.30GiB）。哈希只作身份记录，不是恢复备份。保留小型日志、配置、报告及其他归属未核实资产（BLOCKED_AMBIGUOUS_OWNERSHIP）；未删除共享数据、旧provider、旧五折权重、bank或既有归档。

主工作区已实际切至 restore/ebjd-sixlayer-provider-20261116，核心基于3b5228e036eb94d5f89b5461c557130370edf913。原八层及SDD等后续代码仍在原Git分支/历史中，不重写历史。两个cleanup目录及无关未跟踪文件保留。

五折CPU strict-load通过，GPU有界20-step推理通过；ETH seed2035完整253场景/364行人G/H四指标相对原记录差值均为0，预设atol=rtol=2e-5。除UNIV未发现单人窗口外，每折覆盖N=1及N>=2；UNIV实际smoke含57人。通过真实gather索引验证每人候选排列及固定endpoint，不仅比较minFDE。

入口只读：scripts/replay_restored_baseline.py。旧原始训练入口仍为scripts/run_zara1_unified_coupling_bridge.py，逐折配置/身份由RESTORED_BASELINE_LOCK.json绑定，本次不启动训练。历史validation/test镜像，不能称独立测试。

初次ZARA1定位因链接别名/相对cwd失败，已按登记路径与原论文cwd修正后通过；未改权重、配置或数值容差。
