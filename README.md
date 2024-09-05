# From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos

 <img width="1024" alt="image" src="https://github.com/user-attachments/assets/c629e924-cec2-46c9-9e9a-369b4e6d0aef">


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-dfew)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-dfew?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/dynamic-facial-expression-recognition-on-mafw)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-mafw?p=from-static-to-dynamic-adapting-landmark-1)<br>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=from-static-to-dynamic-adapting-landmark-1)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-static-to-dynamic-adapting-landmark-1/facial-expression-recognition-on-raf-db)](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db?p=from-static-to-dynamic-adapting-landmark-1)<br>

>[From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos](https://ieeexplore.ieee.org/document/10663980)<br>
>Yin Chen$^{†}$, Jia Li$^{†∗}$, Shiguang Shan, Meng Wang, and Richang Hong 

## 📰 News
**[2024.9.5]** The fine-tuned checkpoints are available.

**[2024.9.2]** The code and pre-trained models are aviable.

**[2024.8.28]** The paper is accepted by IEEE Transactions on Affective Computing.

**[2023.12.5]** ~~Code and pre-trained models will be released here~~. 

## 🚀 Main Results

### Dynamic Facial Expression Recognition
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/2144837d-9fd5-4f88-8447-1f6049b38e9a">

<img width="1024" alt="image" src="https://github.com/user-attachments/assets/4a80731e-666e-4cef-9f74-5f794eea7116">


### Static Facial Expression Recognition
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/89a47ea3-1036-4124-927c-563af8007d1f">

### Visualization 
<img width="1024" alt="image" src="https://github.com/user-attachments/assets/aea1385d-0d1b-4f5e-8775-087a30363751">


## Fine-tune with pre-trained weights
1、 Download the pre-trained weights from [here](https://pan.baidu.com/s/1J5eCnTn_Wpn0raZTIUCfgw?pwd=dji4) and move to ckpts directory.

2、 Run the following command to fine-tune the model on the target dataset.
```bash
conda create -n s2d python=3.9
conda activate s2d
pip install -r requirements.txt
bash run.sh
```

## 📋 Reported Results and Fine-tuned Weights
The fine-tuned checkpoints can be downloaded from [here](https://pan.baidu.com/s/1Xz5j8QW32x7L0bnTEorUbA?pwd=5drk).
<table border="1" cellspacing="0" cellpadding="5">
    <tr>
        <th rowspan="2">Datasets</th>
        <th colspan="2">w/o oversampling</th>
        <th colspan="2">w/ oversampling</th>
    </tr>
    <tr>
        <th>UAR</th>
        <th>WAR</th>
        <th>UAR</th>
        <th>WAR</th>
    </tr>
    <tr><td colspan="5" style="text-align: center;">FERV39K</td></tr>
    <tr>
        <td>FERV39K</td>
        <td>41.28</td>
        <td>52.56</td>
        <td>43.97</td>
        <td>46.21</td>
    </tr>
    <tr><td colspan="5" style="text-align: center;">DFEW</td></tr>
    <tr>
        <td>DFEW01</td>
        <td>61.56</td>
        <td>76.16</td>
        <td>64.80</td>
        <td>75.35</td>
    </tr>
    <tr>
        <td>DFEW02</td>
        <td>59.93</td>
        <td>73.99</td>
        <td>62.54</td>
        <td>72.53</td>
    </tr>
    <tr>
        <td>DFEW03</td>
        <td>61.33</td>
        <td>76.41</td>
        <td>66.47</td>
        <td>75.87</td>
    </tr>
    <tr>
        <td>DFEW04</td>
        <td>62.75</td>
        <td>76.31</td>
        <td>66.03</td>
        <td>74.48</td>
    </tr>
    <tr>
        <td>DFEW05</td>
        <td>63.51</td>
        <td>77.27</td>
        <td>67.43</td>
        <td>76.80</td>
    </tr>
    <tr>
        <td>DFEW</td>
        <td>61.82</td>
        <td>76.03</td>
        <td>65.45</td>
        <td>74.81</td>
    </tr>
    <tr><td colspan="5" style="text-align: center;">MAFW</td></tr>
    <tr>
        <td>MAFW01</td>
        <td>32.78</td>
        <td>46.76</td>
        <td>36.16</td>
        <td>44.21</td>
    </tr>
    <tr>
        <td>MAFW02</td>
        <td>40.43</td>
        <td>55.96</td>
        <td>41.94</td>
        <td>51.22</td>
    </tr>
    <tr>
        <td>MAFW03</td>
        <td>47.01</td>
        <td>62.08</td>
        <td>48.08</td>
        <td>61.48</td>
    </tr>
    <tr>
        <td>MAFW04</td>
        <td>45.66</td>
        <td>62.61</td>
        <td>47.67</td>
        <td>60.64</td>
    </tr>
    <tr>
        <td>MAFW05</td>
        <td>43.45</td>
        <td>59.42</td>
        <td>43.16</td>
        <td>58.55</td>
    </tr>
    <tr>
        <td>MAFW</td>
        <td>41.86</td>
        <td>57.37</td>
        <td>43.40</td>
        <td>55.22</td>
    </tr>
</table>

## ✏️ Citation

If you find this work helpful, please consider citing:
```bibtex
@ARTICLE{10663980,
  author={Chen, Yin and Li, Jia and Shan, Shiguang and Wang, Meng and Hong, Richang},
  journal={IEEE Transactions on Affective Computing}, 
  title={From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Adaptation models;Videos;Computational modeling;Feature extraction;Transformers;Task analysis;Face recognition;Dynamic facial expression recognition;emotion ambiguity;model adaptation;transfer learning},
  doi={10.1109/TAFFC.2024.3453443}}
```

