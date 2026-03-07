_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

data_root = 'data/nuscenes/'

# 只改训练参数：epoch数、验证间隔
train_cfg = dict(
    by_epoch=True,
    max_epochs=2,
    val_interval=1
)

# mini数据少，降低学习率防止 loss 爆炸/不稳定
optim_wrapper = dict(
    optimizer=dict(
        lr=0.0001,   # 原版可能是 0.0002~0.0004，这里减半
        # 其他参数继承
    )
)

# 覆盖 backbone 的预训练路径（只改 checkpoint 字段）
model = dict(
    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_tiny_patch4_window7_224.pth'  # ← 改成这个本地相对路径
        )
    )
)


# load_from = 'checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth'