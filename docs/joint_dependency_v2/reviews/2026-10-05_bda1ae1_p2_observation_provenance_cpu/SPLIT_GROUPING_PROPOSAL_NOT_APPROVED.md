# UNIV 保守分组提案：未批准、未执行

状态 PROPOSAL_NOT_APPROVED。正面证据未能证明 uni_examples 与 students001/003 的 recording 独立性；这不等于已经证明它们来自同一 recording，也没有证明轨迹重叠。既有 ID 碰撞仅是身份疑点，不比较受保护 future 来消除此疑点。

从固定 metadata 逐项导出 947 个移动 window_id（students001 425 + students003 522），清单与摘要在 PROPOSAL_MOVED_MEMBERS.json。family/schema 草案和 source-role 映射见 SPLIT_GROUPING_PROPOSAL.json；不是生产 Registry 接受的新协议。

| role | 现行 P2 | 提案 |
|---|---:|---:|
| train | 3484 | 2537 |
| inner_valid | 320 | 1267 |
| outer HOTEL | 445 | 445 |

提案将全部 UNIV sources 放 inner，HOTEL 不变。metadata-only 原 base agent64 packing 重算：train175 packs/11122 agent-window exposures；inner391 packs/24955 exposures。若沿用 epoch cap，goal最多26250、joint43750更新，约为原556 packs/epoch名义更新量的31.47%；没有测量训练吞吐、收敛或真实 wall 成本。

训练窗口减少27.18%，去掉全部 UNIV train 暴露；验证场景分布与选择准则发生变化。不能与旧 P2 当同一实验、不能按大量重叠窗口声称独立统计样本，实际统计效能 UNKNOWN。scene-family grouping 只降低该已知跨角色疑点，不自动解决 zara 来源/recording、静态 map 权限或所有独立性问题。

需用户批准后另行：冻结新 source-role rows、独立 family/schema/expected_role、metadata/order/grant、source/map资格证据、生产 loader与边界测试、全套新预算与预登记。L1 reference_total_steps 的3484不能静默保留或改成2537；需显式新预登记。现 production roles、REGISTERED_ROWS、seed3101 train_order、旧 source/cache 全部不改，也未通过改 recording_id 绕过冲突检测。

947窗已有观察文件在数学内容上可重用，但仍需新协议授权与 wrapper/source-identity 兼容性审核，不能将当前manifest直接改角色开训。
