_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# 没有改模型结构的话，直接加载融合预训练，在此基础上继续训练（无需像官方一样先训 20e lidar 再训 6e 融合）
load_from = 'checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

# 只改了点云分支：不 load_from 点云；图像预训练照常。
#   --cfg-options load_from=None model.img_backbone.init_cfg.checkpoint=checkpoints/swin_tiny_patch4_window7_224.pth

# 只改了图像分支：不加载图像 init_cfg；点云/融合用融合预训练，strict=False。
#   load_from=checkpoints/bevfusion_lidar-cam_xxx.pth（且需 load 时 strict=False）

# 只改了融合模块：用 configs/lidar_cam_change_fusion.py（点云+图像预训练，融合随机初始化）
#   bash tools/dist_train.sh projects/BEVFusion/configs/lidar_cam_change_fusion.py 8

# 只改了检测头：融合预训练 + strict=False，backbone/融合加载，head 随机初始化。
#   load_from=checkpoints/bevfusion_lidar-cam_xxx.pth，并在加载逻辑里 strict=False
# 改了多处：未改的模块对应预训练可加载（点云/图像/融合 ckpt 按上面组合），改的部分不加载。
