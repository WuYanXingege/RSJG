# 修改后网络的逻辑评审与干净目录交付

**已完成。核心统一方案的逻辑与联合预测证据成立；八层门控主干继续作为研究候选，
当前不能宣称它已优于原六层或形成新的理论贡献。**

完整评审见 [ARCHITECTURE_REVIEW_ZH.md](../../../network_release/ebjd_gated_bridge_v1/ARCHITECTURE_REVIEW_ZH.md)，
方法公式及梯度路径见 [METHOD.md](../../../network_release/ebjd_gated_bridge_v1/METHOD.md)。

## 交付目录

`network_release/ebjd_gated_bridge_v1/`：35个常规文件，无软链接、数据、权重、bank和历史大日志。
包含独立模型API、原生provider API、统一训练/评测入口、配置准备、
legacy6/gated6/gated8三种配置、CPU检查、许可证、依赖声明和逐文件来源哈希。
论文工作区仍是唯一开发源；该目录是冻结快照，不新增worktree。

模型/损失/采样文件逐字复制；loader和训练折尺度拟合按原函数提取。
统一训练函数通过AST对照，除来源身份新增release哈希外计算逻辑一致。
剔除原无条件trainer、投影优化器和旧pipeline依赖，规范化有效AdamW/30轮调度的配置描述，
并修复JSON科学计数法被YAML解析为字符串的问题。

## 主要判断

- 论文中心宜放在“保持逐人终点候选集合的学习联合配对＋固定终点条件交互桥”。
- 门控、Q/K归一化和加深具有实现依据，但组件本身不是首创；8层精度、成本仍需对照。
- 桥路径损失当前没有通过H反传到coupler；配对主要由终点代理目标监督。
- 训练GT终点与推理provider终点存在条件分布差异，现有证据不能证明其已经解决。
- 保持终点集合意味着G/H minFDE相同；新路径主干不能直接修复同bank端点覆盖不足。
- 原六层已显示联合和CR收益，但HOTEL路径精度、ETH端点差距和UNIV绝对CR仍是限制。

唯一后续优先事项：匹配共同初始化条件，做legacy6/gated6/gated8主干消融，
决定是否采用8层。本次没有启动该实验。

## 本次实际检查

- 19项定向测试通过：14项既有门控主干测试、5项新目录/配置/来源测试。
  JSON解析修正后重新通过全部5项目录测试。
- 搬到临时独立目录、Python isolated模式导入成功；没有借用论文根目录的ebjd包。
- 完整192/4/768八层模型：4次合成小batch临时更新，关键模块梯度非零并更新，
  模型/optimizer恢复正确，错误架构恢复被拒绝。
- 20-world/20-step G/H采样：有限，固定终点与双射保持。合成检查地图32px，不作为性能数据。
- 真实SDD输入：bookstore_0首个训练窗口，13人、6×256×256地图；
  同折U-Net strict初始化、bank身份、配置启动闸门、完整loss/backward均通过。
  该输入检查没有optimizer更新，也不代替完整训练折尺度拟合或正式精度评测。
- 真实30轮SDD研究配置已准备到资产目录，保留原数据/初始化/baseline绑定；训练未启动。
- GPU成本0；正式optimizer updates=0。既有正式源码、checkpoint、结果和两个cleanup目录未改动。

机器收据见 [VALIDATION.json](VALIDATION.json)。完整运行检查与prepared config位于：
`RSJG_EBJD_assets/analyses/ebjd_network_review_20261113_195834/`。
以 `prepared_sdd_v2/` 为最终配置，早期prepare收据保留用于追踪解析修复，不能用于启动。

上述检查证明目录可运行、训练接口接通和来源可核对；不证明8层的准确率、GPU提速或最优层数。
