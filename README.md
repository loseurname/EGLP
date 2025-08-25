# EGLP:Efficient Guidance with Limited Prompt Images for Nighttime Scene Semantic Segmentation
#### by Xiyuan Liu, Jinlong Shi, Caisheng Liu, Suqin Bai, Qiang Qian, Ao Zhang
# Overview：

<img width="650" height="300" alt="pipeline" src="https://github.com/user-attachments/assets/ddc3d2ae-6c09-418c-8ea2-db6ae833cd3d" />

Nighttime scene parsing faces the challenge of limited labeled night image datasets, making unsupervised domain adaptation (UDA) the predominant method in this field. Traditional UDA methods rely on paired day-night images to guide domain adaptation, which increases the difficulty of data acquisition and limits adaptation generalization across different nighttime datasets. UDA methods assisted by a limited number of prompt images can alleviate this issue. However, the limited prompt images often lead to highly redundant pseudo-labels. Moreover, UDA methods that focus on network architecture and training strategies struggle to handle classes with low domain similarity and tend to neglect dynamic and small-scale objects, such as vehicles and traffic lights, resulting in suboptimal parsing accuracy. To address these challenges, we propose \textbf{Efficient Guidance with Limited Prompt Images for nighttime Scene semantic segmentation (EGLP)}. We design a Rare Class Memory (RCM) module that enhances rare classes at both the label and image levels to improve segmentation performance. Furthermore, we propose the Fusion of Pseudo-labels via Image Similarity (FIS) method, which employs a Day-Night Image Similarity (DNIS) function to compute class-wise similarity between source and target domains. Classes with lower similarity are predicted by a Night-Focused Network (NFNet) trained specifically on night images, while classes with higher similarity are predicted by UDA with rich semantic supervision. A Dynamic Mixing Strategy (DMS) further integrates predictions based on DNIS scores and utilizes limited prompt images to generate high-quality pseudo-labels. We conduct extensive experiments on four nighttime datasets: NightCity, NightCity+, Dark Zurich, and ACDC. The results indicate that EGLP can improve the parsing accuracy of UDA.

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
| Method       | Adaptation                 | mIoU     |Checkpoint|
|:-------------|:--------------------------:|---------:|---------:|
| EGLP(HRADA)  | Cityscapes --> ACDC        | 61.54    |  |

