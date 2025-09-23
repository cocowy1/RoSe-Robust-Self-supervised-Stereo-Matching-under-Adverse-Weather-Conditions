# RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions

## 📖 Introduction
This repository hosts the official resources for the paper *“RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions”* (published in **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**).

RoSe addresses the challenge of **stereo matching under adverse weather conditions** such as rain, fog, and snow, by introducing a **robust self-supervised learning framework** that eliminates the dependency on dense ground-truth disparity annotations.

---

## 🚀 Key Contributions
- **Robust Self-supervised Framework**: Learns disparity estimation without requiring ground-truth labels, leveraging image reconstruction and disparity consistency losses.  
- **Multi-scale Guidance**: Mitigates the degradation effects of adverse weather on stereo matching.  
- **Cross-domain Generalization**: Achieves superior robustness and generalization across diverse datasets and weather scenarios.  
- **Balanced Design**: Delivers competitive accuracy while maintaining efficiency for practical deployment.  

---

## 📊 Highlights from the Paper
- Extensive experiments on **KITTI 2015**, **SceneFlow**, and synthetic adverse weather datasets demonstrate RoSe’s superior performance.  
- Particularly strong results in **foggy and rainy** environments compared to both supervised and self-supervised baselines.  
- See *Tables 2–5* and *Figures 6–8* in the paper for detailed benchmarks.

---

## 📎 Citation
If you use or refer to RoSe in your work, please cite our paper:

```bibtex
@article{rose2023tcsvt,
  title={RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions},
  author={...},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023}
}
```

---

## 📝 License
This project is distributed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
