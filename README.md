# RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions [TCSVT 2025](https://arxiv.org/pdf/2509.19165)

## 📖 Introduction
This repository hosts the official resources for the paper *“RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions”* (published in **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**).

RoSe addresses the challenge of **stereo matching under adverse weather conditions** such as rain, fog, and snow, by introducing a **robust self-supervised learning framework** that eliminates the dependency on dense ground-truth disparity annotations.

# Motivation
Self-supervised methods for vision tasks perform well in normal conditions but suffer significant performance degradation under adverse lighting and weather (night, rain, fog). This is because these conditions introduce noise and reduce visibility, which break the assumption of reliable photometric consistency across different views. As a result, supervision signals become unreliable, weakening the model’s learning process.

<img width="900" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/assumption.png"/></div>

# Superior performance against adverse weather
<img width="900" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/teaser1.png"/></div>

---

## 🚀 Key Contributions
- **Robust Self-supervised Framework**: Learns disparity estimation without requiring ground-truth labels, leveraging image reconstruction and disparity consistency losses.  
- **Multi-scale Guidance**: Mitigates the degradation effects of adverse weather on stereo matching.  
- **Cross-domain Generalization**: Achieves superior robustness and generalization across diverse datasets and weather scenarios.  
- **Balanced Design**: Delivers competitive accuracy while maintaining efficiency for practical deployment.
  
# Overview

<img width="900" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/framework_v2.png"/></div>

An overview of our RoSe. 
  (a) denotes the self-supervised scene correspondence learning. Both branches share weights except for the feature extractors.
  (b) In Step 2, the frozen stereo model from the first step acts as the teacher model, generating high-quality pseudo labels on clear samples and guiding the student model (trainable) with mixed clear and adverse inputs. Both the teacher and student models share the same architecture.
  
---

## 📊 Highlights from the Paper
- Extensive experiments on **KITTI 2015**, **DrivingStereo Weather Subset**,  **MS2 dataset**, and synthetic adverse weather datasets demonstrate RoSe’s superior performance.  
- Particularly strong results in **foggy and rainy** environments compared to both supervised and self-supervised baselines.  
- See *Tables 2–5* and *Figures 6–8* in the paper for detailed benchmarks.

---

# Model Zoo

All pretrained models are available in the [Google Driver:ADStereo](https://drive.google.com/drive/folders/1jdx4-gU8WuytiolZbGDLI-NSUHlQWuH4) and [Google Driver:ADStereo_fast](https://drive.google.com/drive/folders/1WcGgA7OS1lf5JJ3ajbXw-hMtz8cXrQ7k?dmr=1&ec=wgc-drive-globalnav-goto)

We assume the downloaded weights are located under the `./trained` directory. 

Otherwise, you may need to change the corresponding paths in the scripts.

---


# Environment
```
Python 3.9
Pytorch 2.4.0
```
# Create a virtual environment and activate it.
```
conda create -n RoSe python=3.9
conda activate RoSe
```

# Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install chardet
pip install imageio
pip install thop
pip install timm==0.5.4
```

# 1. Prepare training data
To evaluate/train ADStereo, you will need to download the required datasets.

[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

[KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

[Middlebury](https://vision.middlebury.edu/stereo/submit3/)

[ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

By default `datasets.py` will search for the datasets in these locations.

```bash
DATA
├── KITTI
│   ├── kitti_2012
│   │   └── training
        └── testing
│   ├── kitti_2015
│   │   └── training
        └── testing
└── SceneFlow
    ├── Driving
    │   ├── disparity
    │   └── frames_finalpass
    ├── FlyingThings3D
    │   ├── disparity
    │   └── frames_finalpass
    └── Monkaa
        ├── disparity
        └── frames_finalpass
└── Middlebury
    ├── trainingH
    ├── trainingH_GT
└── ETH3D
    ├── two_view_training
    ├── two_view_training_gt
```

## 📎 Citation
If you use or refer to RoSe in your work, please cite our paper:

```bibtex
@article{wang2025rose,
  title={RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions},
  author={Wang, Yun and Hu, Junjie and Hou, Junhui and Zhang, Chenghao and Yang, Renwei and Wu, Dapeng Oliver*},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

---

## 📝 License
This project is distributed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
