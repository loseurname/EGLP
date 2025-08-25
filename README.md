# EGLP:Efficient Guidance with Limited Prompt Images for Nighttime Scene Semantic Segmentation
#### by Xiyuan Liu, Jinlong Shi, Caisheng Liu, Suqin Bai, Qiang Qian, Ao Zhang
# Overview：

<img width="650" height="300" alt="pipeline" src="https://github.com/user-attachments/assets/ddc3d2ae-6c09-418c-8ea2-db6ae833cd3d" />

Nighttime scene parsing faces the challenge of limited labeled night image datasets, making unsupervised domain adaptation (UDA) the predominant method in this field. Traditional UDA methods rely on paired day-night images to guide domain adaptation, which increases the difficulty of data acquisition and limits adaptation generalization across different nighttime datasets. UDA methods assisted by a limited number of prompt images can alleviate this issue. However, the limited prompt images often lead to highly redundant pseudo-labels. Moreover, UDA methods that focus on network architecture and training strategies struggle to handle classes with low domain similarity and tend to neglect dynamic and small-scale objects, such as vehicles and traffic lights, resulting in suboptimal parsing accuracy. To address these challenges, we propose \textbf{Efficient Guidance with Limited Prompt Images for nighttime Scene semantic segmentation (EGLP)}. We design a Rare Class Memory (RCM) module that enhances rare classes at both the label and image levels to improve segmentation performance. Furthermore, we propose the Fusion of Pseudo-labels via Image Similarity (FIS) method, which employs a Day-Night Image Similarity (DNIS) function to compute class-wise similarity between source and target domains. Classes with lower similarity are predicted by a Night-Focused Network (NFNet) trained specifically on night images, while classes with higher similarity are predicted by UDA with rich semantic supervision. A Dynamic Mixing Strategy (DMS) further integrates predictions based on DNIS scores and utilizes limited prompt images to generate high-quality pseudo-labels. We conduct extensive experiments on four nighttime datasets: NightCity, NightCity+, Dark Zurich, and ACDC. The results indicate that EGLP can improve the parsing accuracy of UDA.

<img width="700" height="450" alt="image" src="https://github.com/user-attachments/assets/95918d64-34a1-46a2-b69c-a0d5b5dc0d25" />

# Setup
This EGLP (HRDA) code builds upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [HRDA](https://github.com/lhoyer/HRDA)构建。We highly recommend that you familiarize yourself with MMSegmentation and HRDA before exploring this code.Additionally, when running the code, it is natural to encounter issues, and referring to the baseline code issues may assist you in resolving them more expeditiously and effectively.

We use Python 3.8.16 and recommend setting up a new virtual environment:
```bash
conda create -n pig python=3.8.16
conda activate pig
```
In that environment, the requirements can be installed with:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full==1.3.7
Please refer to HRDA for MiT_b5 and data set download and data set processing. Please refer to [FDLNet]([https://github.com/open-mmlab/mmsegmentation](https://github.com/wangsen99/FDLNet)) for NightCity. The final folder structure should look like this:
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
Please note that the code is properly adjusted according to the prompt images you choose.
# Training & Testing
Let's take Cityscapes --> NightCity as an example and run it for training:
```bash
python run_experiments.py --config configs/pig/pig_city2nightcity.py
```
The results for Cityscapes --> ACDC and Cityscapes --> DarkZurich are reported on the test split of the target dataset. To generate the predictions for the test set, please run:
```bash
python tools/test.py path/to/config_file path/to/checkpoint_file --test-set --format-only --eval-option imgfile_prefix=labelTrainIds to_label_id=False
```
The predictions can be submitted to the public evaluation server of the respective dataset to obtain the test score.

# Checkpoint
| Method       | Adaptation                 | mIoU     |Checkpoint|
|:-------------|:--------------------------:|---------:|---------:|
| EGLP(HRADA)  | Cityscapes --> ACDC        | 61.54    |  |

