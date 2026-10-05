# 架构与真实接口静态复核

来源SUPPLEMENT_METADATA的11模块AST对比与BASE_PROVENANCE的实际checkpoint keys/shapes；未实例化任何网络。

| 模块 | 固定结构 |
|---|---|
| Social history | 4→128/ReLU，单层单向GRU128 |
| 一层message | 270→128/ReLU→128；update256→128/ReLU→128，残差+LN128 |
| base relation | 270→128/ReLU→128/ReLU→4 |
| unary | candidate4→64/SiLU→64/LN，agent128→64/LN，concat192→64/SiLU→1 |
| dynamic | geometry4→32/SiLU→16；M4 query，已有embedding16 |
| pair | candidate4→64/SiLU→64；pair270→128/SiLU→64/LN，fusion64→rank8、M4 |
| future teacher | descriptor6→64/SiLU→64/SiLU→4；仅训练 |
| frozen base | goal U-Net、单层LSTM256、2层time Transformer d512/head4/FF1024 |
| inference | K21/P20，两轮exact persistent-tie refinement，原tree/diffusion |

Stage-A注册331845；其中dynamic relation_embedding64在endpoint loss无路径，331781是结构路径计数，不保证所有参数逐步非零梯度。
base注册7826588（goal613772+history272384+diffusion参数6940432）；forward有效5723804，不用的deepcopy template2102784；buffer12793=positional12288+schedule505单列。corrector30851冻结且末层严格零的canonical literal base bypass保持。新增层/参数/buffer为0。
radius6m、TTC8s、dt.4、Tobs8/Tpred12、edge_chunk256、no scene z，继承配置。

## 流程与缺口

训练：validated logical train→frozen G/p0/obs graph→Social h→unary/base+dynamic relation/energy→prior和future-teacher posterior cost→A0旧soft-neighbor CE或A1 S4 hard-neighbor MC conditional CE→.5post+.5prior+同beta KL→相同Adam/clip/scheduler。
部署：obs+map→K21/h/u/relation/energy→P20原sampler→goal-relative history/tree diffusion→Y[N,20,12,2]；无teacher/GT q/训练MC draw。

真实生产入口是main.py的phase/task flags，parser从run_name推导run目录config.yaml；**没有 --config**。RESOLVED_CONFIGS是parser literal defaults+旧P1科学配置+blocked overrides的静态视图，未调用parser/device guard，不声称生产已resolve成功。
batch_size64为遗留字段，Stage-A loader实际scene1；use_multi_scene_packing=false。空base/cache/init字段不能填旧SHA凑“ready”。
L1必须新增attempt/H停止和跳过evaluate的控制分支，当前没有可运行flag；L2选择adapter也未实现。A0 joint_diagnostics额外teacher loss在prediction之后，A1 MC不做此分支；未来共同禁用/隔离，不把这种不对称带入公平评估。
