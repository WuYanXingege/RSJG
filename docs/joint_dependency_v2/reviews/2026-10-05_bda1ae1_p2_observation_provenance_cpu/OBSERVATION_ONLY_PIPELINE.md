# obs8 纯 CPU 流程

源坐标单位沿用现有 ETH/UCY scene 合同的 world metres；原始 release 的独立单位/recording/map 证据仍待核，不能把代码约定当成来源认证。

| 步骤 | 输入/允许信息 | 输出与原始合同 |
|---|---|---|
| 注册/gate | 固定 metadata、grant、source SHA | 原 window/frame/agent 顺序；无目标坐标 |
| streaming route | 原始 bytes、ID；获准的 (window,t<8,agent) | float64 world [8,N,2]；缺失/重复/非有限拒绝 |
| 几何 | 该 scene H.txt 与 xy8 | data_utils.world2pixel_numpy(H_inv)；ETH/HOTEL 交换 x/y；原生 pixel float64 |
| 静态图 | 原 RGB、grayscale semantic 0..5 | create_tensor_image：NEAREST/down8；RGB uint8→float32/255，semantic float32 onehot，不再 /255 |
| 观察轨迹图 | 原生 pixel.float()/8 | 原 create_CNN_inputs_loop，[N,8,h,w]；std=min(h,w)/64，逐图 peak normalization，不是 sum normalization |
| 保存/验收 | 以下 typed payload | 原子保存、SHA、FileProviders weights_only CPU 回读、内容 SHA、数值校验 |

| 字段 | 形状 / dtype |
|---|---|
| abs_pixel_coord | [8,N,2] / float64，native pixel，不预除 8 |
| seq_list | [8,N] / float64，全 1，保留注册 cohort |
| frame_ids | [8,N] / int64，保持前 8 帧顺序 |
| input_traj_maps | [N,8,h,w] / float32 |
| tensor_image | [6,h,w] / float32，unlabeled/pavement/road/structure/terrain/tree |
| tensor_map | [3,h,w] / float32 RGB |
| scene_index、scene_ptr | [N] 全0、[2]=[0,N] / int64 |
| batch_format_version | scalar / int64，2 |

h/w 从静态图片实际尺寸及原 resize 推导：本次 ETH 60×80，其余 72×90。各窗 N/shape/文件大小/内容 SHA/raster 外槽位在 catalog shards，source-level 汇总在 RESULTS。RGB/semantic 尺寸、H 可逆、通道合法性、CPU、dtype、finite、onehot、RGB范围、Gaussian峰值均检查。

不读取 Dataset/Experiment/Trajectory_Data_Pre_Process 构造器，不使用 bbox/future 归一化。纯函数引用：src/data_src/data_utils.py；src/models/model_utils/cnn_big_images_utils.py；sampling_2D_map.py。producer_hash 精确绑定这些数值路径及 src/p2_observation.py。其余执行代码另列完整 SHA。

## 输入准备与合成测试

observation_inputs 生成 [8,N,8]，仅由观察位置、相对位置和观察段 derivative_of 得到。原 GDTS.prepare_inputs 本来就对 observed/future 分段求导；没有修复不存在的跨界导数问题。用 fake scene + 合成 full20 调用无绑定 prepare_inputs 纯准备方法，对照选择输入逐位一致，改变合成未来不影响观察特征；不构造模型。真实 full20 parity 为 NOT_RUN（未授权）。

本轮只有合成 target 接口测试；真实 target exporter CLI 未实现、未运行。后续如单独授权，应新建用途受限的 streaming target exporter：train3484 的 t=8..19 用于 gradient，inner320 仅 metric，outer445 一律拒绝；先验证 gate，再按相同 ID/cohort 路由并写独立 target wrapper。不得用现工具的 obs grant 扩权，也不得先读 full20 再切片。
