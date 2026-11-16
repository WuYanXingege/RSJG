# 预注册原型pilot计划（历史版本）

2026-11-16勘误：以下早期“GPU秒/epoch边界”描述已由 prepilot_fixes_20261116 的累计流程墙钟、场景边界检查和37900更新上限取代；不构成追加预算或自动启动授权。本次单run授权另见对应GPU gate/pilot计划，不修改历史默认开关。

协议：ETH外层训练manifest内按完整原始sequence隔离，字典序最后约20%的sequence作为internal dev（当前uni_examples），其余train；同sequence不跨split，因此重叠滑窗不跨边界。当前train3790窗口、dev320窗口；尺度仅用train的36285次行人出现。原ETH validation/test不参与选择。内部dev被旧matched模型训练暴露，旧provider仅作历史工程参考，不能用于该dev公平胜负判断。

默认真实训练关闭。未来需另行授权并冻结配置后，最多10epoch/3600 GPU秒，patience=3次dev NLL，Adam 1e-4，gamma1，seed3101；预算在完整epoch边界检查，不启动第二轮探索。报告实际成本，不承诺一小时可完成。配置中的所有数值是原型预注册设置。

同一密度checkpoint比较：IID K20；M256贪心覆盖K20；同M256的kmeans/medoid对照；M512明确预算消融。后两者与无多horizon、无运动归一化属于后续计划，本轮未实现或运行，不伪报已完成。独立小GMM训练控制尚待准备，不能以旧matched在dev的成绩替代。

pilot只判断代码/候选是否值得进一步验收：B相对同密度IID的dev minFDE至少改善0.01m，Recall@0.5不下降、P90不退化；分布NLL与合法IID能量评分单列，不能用无权代表点评估概率校准。若不满足则停止，不扫描lambda/M/划分。真正替换旧provider还需公平独立dev控制、预先固定非劣界与尾部/校准检查，以及3个训练seed，再随机初始化训练六层coupling/bridge。

计划记录minFDE、固定半径Recall、P50/P90/P95、NLL、IID能量评分、重复率、N=1/N>=2、速度/转弯与尾部、总延迟及内存。现有endpoint指标和IID能量函数已提供；校准分组/地图可走性完整评估未实现。无地图不造可走性指标。旧0.5811m不是此新dev的合法绝对门槛。

本轮不自动启动任何训练、队列或三seed实验。新provider换入旧桥仅能标为diagnostic_provider_swap_inherited_weights，不能声称完整系统无GDTS训练依赖。
