# RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions [TCSVT 2025](https://arxiv.org/pdf/2509.19165)

## 📖 Introduction
This repository hosts the official resources for the paper *“RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions”* (published in **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**).

RoSe addresses the challenge of **stereo matching under adverse weather conditions** such as rain, fog, and snow, by introducing a **robust self-supervised learning framework** that eliminates the dependency on dense ground-truth disparity annotations.

## 🎓 Motivation
Self-supervised methods for vision tasks perform well in normal conditions but suffer significant performance degradation under adverse lighting and weather (night, rain, fog). This is because these conditions introduce noise and reduce visibility, which break the assumption of reliable photometric consistency across different views. As a result, supervision signals become unreliable, weakening the model’s learning process.

<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/assumption.png"/></div>

## 📢 Superior performance against adverse weather
<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/comparison_1.png"/></div>

---

## 🚀 Key Contributions
- **Robust Self-supervised Framework**: Learns disparity estimation without requiring ground-truth labels, leveraging image reconstruction and disparity consistency losses.  
- **Multi-scale Guidance**: Mitigates the degradation effects of adverse weather on stereo matching.  
- **Cross-domain Generalization**: Achieves superior robustness and generalization across diverse datasets and weather scenarios.  
- **Balanced Design**: Delivers competitive accuracy while maintaining efficiency for practical deployment.

---

## 😊 Overview

<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/framework_v2.png"/></div>

An overview of our RoSe. 
  (a) denotes the self-supervised scene correspondence learning. Both branches share weights except for the feature extractors.
  (b) In Step 2, the frozen stereo model from the first step acts as the teacher model, generating high-quality pseudo labels on clear samples and guiding the student model (trainable) with mixed clear and adverse inputs. Both the teacher and student models share the same architecture.

## Visual Comparison with two steps

<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/ablation_comparison.png"/></div>

Visual comparison of RoSe on the weather validation set. Baseline denotes that RAFTStereo was trained using the vanilla photometric consistency and disparity smooth losses. Step 1 indicates the self-supervised scene corresponding learning step. Step 2 denotes the adverse weather distillation step. The predicted disparity maps and their corresponding error maps are displayed in columns 2, 3, and 4, respectively.

---

## 📊 Highlights from the Paper
- Extensive experiments on **KITTI 2015**, **DrivingStereo Weather Subset**,  **MS2 dataset**, and synthetic adverse weather datasets demonstrate RoSe’s superior performance.  
- Particularly strong results in **foggy and rainy** environments compared to both supervised and self-supervised baselines.  
- See *Tables 2–5* and *Figures 6–8* in the paper for detailed benchmarks.

## 	👊 Robust Generalization Performance

<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/generalization.png"/></div>

## 💪 Robust Performance in Adverse Weather Conditions

<img width="1200" src="https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions/blob/main/images/ds_performance.png"/></div>

---

## 🏢 Model Zoo

All pretrained models are available in the [RoSe](https://drive.google.com/drive/folders/1FVwilmYtGXq0iQaEVT46nbXKaHXj44p5?dmr=1&ec=wgc-drive-globalnav-goto) 

We assume the downloaded weights are located under the `./trained` directory. 

Otherwise, you may need to change the corresponding paths in the scripts.

---

## Installation

Our code is developed based on pytorch 1.9.0, CUDA 10.2 and python 3.8. Higher version pytorch should also work well.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create -f conda_environment.yml
conda activate RoSe
```

Alternatively, we also support installing with pip:

```
bash pip_install.sh
```

## 1. Prepare training data
To evaluate/train RoSe, you will need to download the required datasets.

[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

[KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

[DrivingStereo](https://drivingstereo-dataset.github.io/)

[MS2](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset)

By default, `datasets.py` will search for the datasets in these locations.

```bash
DATA
├── AdverseKITTI
│   ├── kitti_2012
│   │   └── robust_train
│   │      └── clear
│   │      └── foggy
│   │      └── rainy
│   │      └── night
        └── testing
│   ├── kitti_2015
│   │   └── robust_train
│   │      └── clear
│   │      └── foggy
│   │      └── rainy
│   │      └── night
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
├── DrivingStereo
│   ├── robust_train
│   │   └── clear
│   │   └── foggy
│   │   └── rainy
│   │   └── night
    └── AdverseWeather testset

├── MS2
│   ├── robust_train
│   │   └── clear
│   │   └── foggy
│   │   └── rainy
│   │   └── night
    └── AdverseWeather testset
```
## 2. Create AdverseWeather subset
We employ cycleGAN-turbo [img2img-turbo](https://github.com/GaParmar/img2img-turbo) to generate the adverse weather subset.  The toolkit can be downloaded and used with the officially released day2night weights to generate the corresponding subsets by following the provided instructions. As day2rainy and day2foggy weights are not available in the official release, we train the model ourselves and additionally provide the trained day2rainy and day2foggy weights [day2foggy \& day2rainy](https://drive.google.com/drive/folders/1IWwIMiAeEtgjhJ-M-EKAh6h7C26WVjDu?dmr=1&ec=wgc-drive-globalnav-goto), thereby extending the applicability of the toolkit to a broader range of adverse weather conditions.

We also provide the created AdverseKITTI dataset [dataset](https://drive.google.com/drive/folders/1PI-H2bD1VGv-DoBzi62ul3nhwYKQTtx9?dmr=1&ec=wgc-drive-globalnav-goto).

## 3. Train on SceneFlow
Run `main_stereo.py` to train on the SceneFlow dataset. Please update the datapath in `main_stereo.py` to your training data path.
Please note that the hyperparameter `enhancement=False` is used in the pre-training process.
```
parser.add_argument('--enhancement', default=False, type=bool, help='optional feature refinement')
```

## 4. Finetune \& Inference 
Run `main_stereo_unp.py` to finetune on the different real-world datasets (Step 1). Please update the datapath in `main_stereo_unp.py` to your training data path.

Run `main_stereo_pseudo.py` to finetune on the different real-world datasets (Step 2). Please update the datapath in `main_stereo_pseudo.py` to your training data path.

Run `evaluate_stereo.py` to evaluate on the different real-world datasets. Please update the datapath in `evaluate_stereo.py` to your training data path.

---

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
