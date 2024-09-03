# From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos

 <img width="1024" alt="image" src="https://github.com/user-attachments/assets/c629e924-cec2-46c9-9e9a-369b4e6d0aef">


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-dfew)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-dfew?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-mafw)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-mafw?p=from-static-to-dynamic-adapting-landmark-1)<br>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=from-static-to-dynamic-adapting-landmark-1)<br>

>[From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos](https://arxiv.org/pdf/2312.05447)<br>
>Yin Chen$^{†}$, Jia Li$^{†∗}$, Shiguang Shan, Meng Wang, and Richang Hong 

## 📰 News
**[2024.9.2]** The code and pre-trained models are aviable.
 
**[2023.12.5]** ~~Code and pre-trained models will be released here~~. 

## 🚀 Main Results

### Dynamic Facial Expression Recognition
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/2144837d-9fd5-4f88-8447-1f6049b38e9a">

<img width="1024" alt="image" src="https://github.com/user-attachments/assets/4a80731e-666e-4cef-9f74-5f794eea7116">


### Static Facial Expression Recognition
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/89a47ea3-1036-4124-927c-563af8007d1f">

### Visualization 
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/aea1385d-0d1b-4f5e-8775-087a30363751">

<img width="512" alt="image" src="https://github.com/user-attachments/assets/9f93923a-478f-4083-b677-5e17da650d51">

## Fine-tune with pre-trained weights
1、 Download the pre-trained weights from [here](https://pan.baidu.com/s/1J5eCnTn_Wpn0raZTIUCfgw?pwd=dji4) and move to ckpts directory.

2、 Run the following command to fine-tune the model on the target dataset.
```bash
conda create -n s2d python=3.9
conda activate s2d
pip install -r requirements.txt
bash run.sh
```


## ✏️ Citation

If you find this work helpful, please consider citing:
```bibtex
@article{chen2023static,
  title={From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos},
  author={Chen, Yin and Li, Jia and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={arXiv preprint arXiv:2312.05447},
  year={2023}
}
```

