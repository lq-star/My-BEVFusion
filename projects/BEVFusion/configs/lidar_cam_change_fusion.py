# 替换 BEV 融合模块（如用扩散模型替代 ConvFuser）时的训练配置
# 策略：加载点云预训练 + 图像预训练，融合模块随机初始化，再训 6 epoch
_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# 不加载完整融合 ckpt，改为两路预训练
load_from = 'checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'
# 图像 backbone 用预训练（load_from 只加载与当前模型匹配的 key，其余靠 init_cfg）
# 融合模块（下面替换成的 diffusion）无预训练，随机初始化
model = dict(
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_tiny_patch4_window7_224.pth')),
    # 用你自己的扩散融合模块替换官方 ConvFuser
    # 接口需一致：forward(inputs: List[Tensor]) -> Tensor
    # 输入: [img_bev_feat, pts_bev_feat]，通道 [80, 256]；输出: 256 通道 BEV
    fusion_layer=dict(
        type='ConvFuser',  # 改成你的模块，如 type='DiffusionFusion', in_channels=[80, 256], out_channels=256
        in_channels=[80, 256],
        out_channels=256),
)

# 若 load_from 点云 ckpt 时报 state_dict key 不匹配，需对 load_checkpoint 使用 strict=False
# 可在 tools/train.py 里对 cfg 增加 load_strict=False，并在 Runner 中传给 load_checkpoint（若框架支持）
