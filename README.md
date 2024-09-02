# From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos
The code will be released soon!

## Results on Dynamic Facial Expression Benchmarks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-dfew)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-dfew?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-mafw)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-mafw?p=from-static-to-dynamic-adapting-landmark-1)<br>

## Results on Static Facial Expression Benchmarks
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=from-static-to-dynamic-adapting-landmark-1)<br>

 
## Fine-tune with pre-trained model
1、 Download the pre-trained model from [here](https://pan.baidu.com/s/1J5eCnTn_Wpn0raZTIUCfgw?pwd=dji4) and move to ckpts directory.

2、 Run the following command to fine-tune the model on the target dataset.
```bash
conda create -n s2d python=3.9
conda activate s2d
pip install -r requirements.txt
bash run.sh
```
