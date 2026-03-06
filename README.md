# 一、项目文件说明

---
- configs
  基础配置文件，其他的继承它。
  
- data
  数据集存放在这。
  
- mmdet3d
  提供通用 3D 检测/分割模型、数据集接口和工具函数
  
- projects
  本项目核心代码文件。

- requirements
  `pip install -e .`时，先找setup.py，setup.py说去找requirements.txt，requirements.txt说去找requirements文件夹里的文件，然后根据这些文件安装依赖。

- tool
  训练、测试、数据转换与日志分析等脚本工具。
  
- requirements.txt
  Python 依赖列表，用于创建虚拟环境或一次性安装本项目所需的第三方库。
  
- setup.cfg
  项目打包和工具配置文件。
  
- setup.py
   `pip install -e .` 以开发模式安装本项目，就是把mmdet3d文件夹注册成当前虚拟环境的一个包并生成mmdet3d.egg-info文件夹，就可以在写代码时import mmdet3d,改mmdet3d里的源码虚拟环境包里的也会同步。
# 二、配置环境过程

---
参考链接：https://blog.csdn.net/h904798869/article/details/132210022

对于一台新电脑需要安装好:
- 显卡驱动
- ubuntu20.04
- python-3.8
- torch-1.10.0
  安装Anaconda来创建虚拟环境，每个环境可以配置不同的python和torch版本。python和torch都是在虚拟环境中安装的，不是系统级的。
- cuda-11.3
  BEVFusion项目需要安装系统级cuda，因为有自定义的算子。
  在Anaconda的虚拟环境中安装的cuda没有nvcc编译器，都是预先编译好的。对于不需要自定义算子的模型可以使用。
  BEVFusion项目因为需要安装系统级cuda，因此，如果用自己电脑可以安装多个系统级cuda，设置路径指向需要的cuda版本。本文使用Auto DL。
- cudnn-8.6
  它也分为系统级和虚拟环境中的，系统级的cudnn跟系统级的cuda在同一个文件夹下。

在Auto DL上选好上面的配置之后：
```python
# 1 创建虚拟环境
conda create -n bevfusion python=3.8
# 2 激活虚拟环境
conda activate bevfusion
# 安装torch
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# 安装系统依赖
sudo apt-get update
sudo apt-get install wget libgl1-mesa-glx libglib2.0-0 openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y
# 安装基础 Python 依赖
pip install Pillow==8.4.0 tqdm torchpack nuscenes-devkit mpi4py==3.0.3 numba==0.56.4 setuptools==56.1.0 ninja==1.11.1 numpy==1.23.4 opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 yapf==0.40.1

# 本项目的mmdetion3d版本是1.4，因此安装mim来自动匹配mmengine，mmcv，mmdet版本
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"

# 拉取我的仓库源码 
git clone https://github.com/lq-star/My-BEVFusion

# 在项目根目录下，把当前项目（My-BEVFusion）以可编辑方式安装进当前环境，主要是把mmdet3d文件夹变成可以import的包。装完后可以 import mmdet3d，并且直接改仓库里的代码就会生效。这一步跟安装mmengine，mmcv，mmdet一样，相当于安装mmdet3d。

# 这一步完成如果没有自定义算子就算安装完了，可以训练了。bevfusion项目由于有自定义算子，还需要继续安装。
pip install -v -e .

# 编译 BEVFusion 的自定义算子，生成bev_pool.egg-info和build文件夹
python projects/BEVFusion/setup.py develop

# 至此，环境配置完成
# 检查torch，cuda
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
# 检查mmcv, mmengine, mmdet, mmdet3d
python -c "import mmcv, mmengine, mmdet, mmdet3d; \
print('mmcv', mmcv.__version__); \
print('mmengine', mmengine.__version__); \
print('mmdet', mmdet.__version__); \
print('mmdet3d', mmdet3d.__version__)"
# 检查bevfusion自定义算子
python -c "from projects.BEVFusion.bevfusion.ops import bev_pool, voxel; print('ops import ok')"
```

# 三、数据集准备

---
预处理data/nuscenes数据集，得到可训练的数据。
```python
# 用mini数据集生成pkl文件
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini

# 用完整数据集生成pkl文件
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

# 四、下载预训练模型

```python
# 创建一个放预训练模型的文件夹
mkdir -p checkpoints
cd checkpoints
# 下载 lidar-only
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
# 下载 lidar+cam
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
```

# 五、训练，测试，推理

---
- ## 官方命令
```python
# demo
python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_FILE} --cam-type all --score-thr 0.2 --show

# 训练分两步，先训练激光雷达
bash tools/dist_train.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8
# 下载图像预训练模型，再训练完整bevfusion模型
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py 8 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}

# 测试
bash tools/dist_test.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ${CHECKPOINT_PATH} 8

```

- ## 单卡lidar-only
```python
# 1. lidar-only 训练（从头训练，不加载预训练权重）
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# 2. lidar-only 训练（加载预训练权重）
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e  _nus-3d.py \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth

# 3. lidar-only 测试（用官方预训练模型），要用自己训练出来的模型的话，就把第三行换成自己训练出来的模型
python tools/test.py \
  projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  --eval bbox

```

- ## 单卡lidar+cam
```python
# 1.idar+cam训练（从头训练，不加载预训练权重）
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# 2. idar+cam训练（加载预训练权重）
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

# 3. idar+cam测试（用官方预训练模型），要用自己训练出来的模型的话，就把第三行换成自己训练出来的模型
python tools/test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --eval bbox
  
# 4.论文里的训练方式是加载点云预训练模型和图像预训练模型
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  model.img_backbone.init_cfg.checkpoint=checkpoints/swint-nuimages-pretrained.pth
```
- ## 单机多卡lidar-only
```python
# 1.lidar-only 多卡训练（从头训练，不加载预训练权重）
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  4 \
  --amp
  
# 2.lidar-only 多卡训练（加载预训练权重）
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  4 \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  --amp
  
# 3. lidar-only 多卡测试（用官方预训练模型），要用自己训练出来的模型的话，就把第三行换成自己训练出来的模型
bash tools/dist_test.sh \
  projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  4 \
  --eval bbox
```
- ## 单机多卡lidar+cam
```python
# 1. lidar+cam 多卡训练（从头训练，不加载预训练权重）
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  4 \
  --amp

# 2. lidar+cam 多卡训练（加载官方 lidar+cam 预训练权重，微调）
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  4 \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  --amp

# 3. lidar+cam 多卡测试（用官方预训练模型），要用自己训练出来的模型就把第三行换成自己的 ckpt
bash tools/dist_test.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
  4 \
  --eval bbox

# 4. 论文方式的多卡训练：加载点云预训练 + 图像预训练
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  4 \
  --cfg-options \
  load_from=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth \
  model.img_backbone.init_cfg.checkpoint=checkpoints/swint-nuimages-pretrained.pth \
  --amp
```