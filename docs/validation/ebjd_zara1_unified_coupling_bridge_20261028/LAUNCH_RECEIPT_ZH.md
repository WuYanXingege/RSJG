# ZARA1 统一配对条件桥启动收据

状态：`TRAINING`。

- 冻结源码提交：`9bf9e71b0da74e69f3bca9add6833c39b51f1642`
- 后台服务：`ebjd-zara1-unified-coupling-20261028.service`
- 实验目录：`RSJG_EBJD_assets/experiments/zara1_unified_coupling_bridge_v1_20261028_214026/`
- 固定设置：ZARA1、训练种子 3101、30 epochs、192/4/768、6 FJAB、FULL、CTF、Raw、20 worlds、20 DDIM steps。
- rollout 与 paired actual-step projection 均关闭。
- 冻结原生 proposal provider 不在 optimizer 中；除同折已认证地图 U-Net 初始化外，MSCE、coupling head 与条件桥均随机初始化。

首个正式 optimizer update：总损失 `1.7873766422`，bridge `1.1133211851`，pair `0.6740555167`；bridge 梯度范数 `1.1815886497`，coupling 梯度范数 `0.0002859483`，均有限且非零。该 update 的 rollout/projection 标志均为 0。

ZARA2 旧流水线已在完整 epoch 15 后暂停，服务为 inactive；恢复点 SHA256 为 `a3dc355fe7ca61c9b8fa586fb1701f8ac831a4ce5eeb671c8e0a2ff792b68f29`。本实验不会自动恢复该流水线。

本文件仅为启动收据；模型结论需等待完整训练、固定 selector 与五推理种子 G/H 评测。
