# 机器学习大作业——目标检测(不均衡问题)
> 检测图片中的带芯和不带电芯充电宝，参考项目:https://github.com/open-mmlab/mmdetection

## 安装

### 依赖
+ Linux
+ Python 3.5+
+ PyTorch 1.1 or higher
+ CUDA 9.0 or higher
+ NCCL 2
+ GCC 4.9 or higher
+ mmcv


### 安装步骤
a. 创建虚拟环境并进入
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. 安装PyTorch 和 torchvision
```shell
conda install pytorch torchvision -c pytorch
```
c. 下载项目
```shell
git clone https://github.com/gjt9274/ML-homework.git
cd mmdetection
```
d. 安装mmdetection(其他依赖会自动安装)
```shell
pip install mmcv
python setup.py develop
```

## 目录介绍

> ├── Anno_test  助教给的测试标注
> ├── Image_test 助教给的测试图片
> ├── calculate_map_test.py  助教给的评估脚本
> ├── core_coreless_test.txt 助教给的测试文件
> ├── mmdetection
> │   ├── mmdet
> │   ├── tools
> │   ├── configs
> │   ├── data
> │   ├── VOCdevkit
> │   │   ├── VOC2007


## 训练（以faster_rcnn_r50）为例
```shell
cd mmdetection
python tools/train.py configs/faster_rcnn_r50_fpn_1x.py --gpus 1 --work_dir work_dirs
```

## 测试
```shell
cd mmdetection
python tools/test.py  configs/faster_rcnn_r50_fpn_1x.py work_dirs/faster_rcnn_r50_fpn_1x/latest.pth --out=eval/result.pkl
```
> 会在根目录下生成一个predicted_file文件夹
> ├── predicted_file
> │   ├── det_test_不带电芯充电宝.txt
> │   ├── det_test_带电芯充电宝.txt


## 用助教给的文件计算map
> 在根目录下运行
```shell
python calculate_map_test.py
```
