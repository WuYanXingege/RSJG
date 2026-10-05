# 后续 fresh goal → joint：尚未执行

状态 TEMPLATE_BLOCKED。下面只列本轮真正执行过的只读 launch-check 形状；不可移除 --p2_launch_check 直接训练。精确 argv、真实候选manifest hash与结果在 LAUNCH_CHECKS.json：

```text
CUDA_VISIBLE_DEVICES='' <python> -B main.py
  --p2_config <prior archive>/CLEAN_GOAL_P2.yaml
  --p2_manifest <this archive>/P2_RUNTIME_MANIFEST.json
  --p2_manifest_hash <exact recorded candidate hash>
  --device cpu --p2_launch_check
```

joint对应CLEAN_JOINT_P2.yaml；cache在A0_BLOCKED.yaml上显式改p2_mode=cache、phase=build-jdv2-cache；L1分别A0/A1。实际运行由guarded脚本调用同一parser分支，未调用timed_main。它们都是来源阻断，没有weight/provider读取或设备查询转发。

## 未来操作依赖，不是可执行训练命令

1. 先明确保留现行P2并补齐recording/member/converter/map证据，或批准独立grouped-UNIV新协议并完成其证据/实现验收。不能只改recording_id。
2. 另行授权CPU用途分离targets：只train3484的gradient、inner320的metric；若选新协议则重新固定成员/用途。真实target exporter CLI目前未实现；现有split_window_payload仅是需要先获得full-window授权的纯函数，本轮只用合成夹具，不能当作实际raw target exporter。
3. 新streaming target工具需先按window_id+t=8..19+registered agents检查purpose、source/address/hash，再解析对应x/y，独立原子保存并回读；outer始终拒绝，inner不得给prediction/teacher/gradient。先合成反例测试，再对指定成员执行。无teacher/cache生成授权自动继承。
4. 来源/地图与targets真正齐备后，另行申请fresh goal有限GPU预算、原生epoch/selection边界与失败策略；注册准确config/hash/输出路径，不继承违规权重。
5. 审核合格fresh goal后再单独申请joint，goal-only初始化；history/diffusion fresh。审核合格完整base后才能申请新cache及train-only teacher；shared初态与两臂预算齐备后才考虑L1。

本轮没有打印可直接开训的命令，没有恢复SDD、启动GPU或训练。
