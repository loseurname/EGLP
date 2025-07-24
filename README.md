# EGLP:高效利用有限的提示图像指导夜间场景语义分割
#### 作者：liu xi yuan
# 概述：

<img width="650" height="300" alt="image" src="https://github.com/user-attachments/assets/25170f0a-ef6e-4c70-9fa1-867d10afa885" />

夜间场景解析旨在从夜景图像中提取像素级语义信息，这对于理解场景中的物体分布至关重要。然而，由于标注夜间图像数据集的稀缺性，**无监督领域适应（UDA）** 成为研究夜间场景的主要手段。传统的UDA方法依赖于成对的日间和夜间图像对来指导领域适应，但这种方式不仅增加了数据集构建难度，还限制了模型对不同数据集夜间场景的泛化能力。此外，专注于网络架构和训练策略的UDA难以处理域相似性较低的类别。其对动态及小型物体（如车辆和信号灯）的忽视，往往导致预测精度不尽人意。  

我们利用**EGLP方法**即高效利用有限的提示图像来指导夜间场景的语义分割。针对夜间出现极少的稀有类别，我们提出了**稀有类别记忆库模块（RCM）**，旨在通过在标签和图像层面增强动态和小型物体类别，以提升夜间语义分割的性能。提升网络对于动态小目标类别的预测。为了生成高质量的伪标签，我们提出了**针对图像相似性引导的伪标签融合方法（FIS**）。对于图像相似性较低的类别，由专门在夜间图像上训练的NFNet进行预测；而对于领域相似性较高的类别，则由具有丰富标注语义的UDA进行预测。此外，针对少提示图像我们提出了一种伪标签混合策略：**动态混合策略（AMS）**，旨在通过动态混合伪标签来缓解过拟合问题。  

我们在四个夜间数据集上进行了广泛的实验：NightCity、NightCity+、Dark Zurich和ACDC。实验结果表明，使用EGLP可以提升UDA的解析准确性。

<img width="700" height="450" alt="image" src="https://github.com/user-attachments/assets/95918d64-34a1-46a2-b69c-a0d5b5dc0d25" />

# 设置
本项目的 EGLP (HRDA) 代码基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 和 [HRDA](https://github.com/lhoyer/HRDA)构建。我们强烈建议您在探索此代码之前先熟悉 MMSegmentation 和 HRDA。此外，在运行代码时遇到问题是很正常的，参考基准代码问题可以帮助您更快速、更有效地解决问题。


我们使用 Python 3.8.16，并建议设置一个新的虚拟环境：请运行 
```bash
conda create -n pig python=3.8.16
conda activate pig
```
在该环境中，可以使用以下命令安装要求：
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full==1.3.7
```
MiT_b5 的数据集下载及处理请参考 HRDA。NightCity 的数据集下载请参考FDLNet。最终文件夹结构应如下所示：
```bash
EGLP
├── ...
├── pretrained
│   ├── mit_b5.pth
├── data
│   ├── acdc
│   │   ├── gt
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── dark_zurich
│   │   ├── gt
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── nightcity
│   │   ├── gt
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── img
│   │   │   ├── train
│   │   │   ├── val
│   ├── prompt
│   │   ├── gt
│   │   ├── img
```
请注意，代码会根据您选择的提示图像进行适当调整。
# 训练与测试
我们以Cityscapes --> NightCity为例，运行进行训练：
```bash
python run_experiments.py --config configs/pig/pig_city2nightcity.py
```
Cityscapes --> ACDC 和 Cityscapes --> DarkZurich 的结果在目标数据集的测试集上报告。要生成测试集的预测，请运行：
```bash
python tools/test.py path/to/config_file path/to/checkpoint_file --test-set --format-only --eval-option imgfile_prefix=labelTrainIds to_label_id=False
```
预测可以提交到相应数据集的公共评估服务器以获得测试分数。

# Checkpoint
