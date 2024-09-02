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
- DFEW
The fine-tuned checkpoints and logs across five folds on DFEW are provided as follows:

    - trained without oversampler

    | Flod    | UAR   | WAR   | Fine-tuned Model |
    | ------- | ----- | ----- | ---------------- |
    | 1       | 61.56 | 76.16 |
    | 2       | 59.93 | 73.99 |
    | 3       | 61.33 | 76.41 |
    | 4       | 62.75 | 76.31 |
    | 5       | 63.51 | 77.27 |
    | average | 61.82 | 76.03 |
    - trained with oversampler
    
    | Flod    | UAR   | WAR   | Fine-tuned Model |
    | ------- | ----- | ----- | ---------------- |
    | 1       | 64.80 | 75.35 |
    | 2       | 62.54 | 72.53 |
    | 3       | 66.47 | 74.87 |
    | 4       | 66.03 | 74.48 |
    | 5       | 67.43 | 76.80 |
    | average | 65.45 | 74.81 |
- FERV39K
    - trained without oversampler

    | UAR   | WAR   | Fine-tuned Model |
    | ----- | ----- | ---------------- |
    | 41.28 | 52.56 |

    - trained with oversampler

    | UAR   | WAR   | Fine-tuned Model |
    | ----- | ----- | ---------------- |
    | 43.97 | 46.21 |

- MAFW
    - trained without oversampler
    
    | Flod    | UAR   | WAR   | Fine-tuned Model |
    | ------- | ----- | ----- | ---------------- |
    | 1       | 32.78 | 46.76 |
    | 2       | 40.43 | 55.96 |
    | 3       | 47.01 | 62.08 |
    | 4       | 45.66 | 62.61 |
    | 5       | 43.45 | 59.42 |
    | average | 41.86 | 57.37 |

    - trained with oversampler
    
    | Flod    | UAR   | WAR   | Fine-tuned Model |
    | ------- | ----- | ----- | ---------------- |
    | 1       | 36.16 | 44.21 |
    | 2       | 41.94 | 51.22 |
    | 3       | 48.08 | 61.48 |
    | 4       | 47.67 | 60.64 |
    | 5       | 43.16 | 58.55 |
    | average | 43.40 | 55.22 |