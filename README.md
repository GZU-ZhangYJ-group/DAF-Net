# A Frequency-Domain Dynamic Amplitude Filtering Method for Single-Image Dehazing with Harmony Enhancement

Yabo Wu, Yongjun Zhang*, Ziyang Chen, Yong Zhao*

~~~
The official implementation of "A Frequency-Domain Dynamic Amplitude Filtering Method for Single-Image Dehazing with Harmony Enhancement".
This paper is under review processing... We will release all codes and results after finishing the review.
~~~

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Download the Datasets
- [reside-indoor](https://drive.google.com/drive/folders/1pbtfTp29j7Ip-mRzDpMpyopCfXd-ZJhC)
- [reside-outdoor](https://drive.google.com/drive/folders/1eL4Qs-WNj7PzsKwDRsgUEzmysdjkRs22)
## Training and Evaluation
### Training on ITS:
~~~
cd ITS
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --data_dir your_dataset_path/reside-indoor
~~~
### Training on OTS:
~~~
cd OTS
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --data_dir your_dataset_path/reside-outdoor
~~~
### Testing on SOTS-Indoor:
~~~
cd ITS
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_dir your_dataset_path/reside-indoor
~~~
### Testing on SOTS-Indoor:
~~~
cd OTS
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_dir your_dataset_path/reside-outdoor
~~~
## Results
|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Image Dehazing**|SOTS-Indoor|42.98 dB|0.997|
||SOTS-Outdoor|39.10 dB|0.995|
||Dense-Haze|17.91 dB|0.64|
||NH-HAZE|20.77 dB|0.82|
||HAZE4K|34.42 dB|0.99|


## Citation
~~~

~~~

## Contact
Please contact Yabo Wu (1394884511@qq.com) if you have any questions.

**Acknowledgment:** This code is based on the [MIMO-UNet](https://github.com/chosj95/MIMO-UNet/tree/main?tab=readme-ov-file#gpu-syncronization-issue-on-measuring-inference-time) and [ConvIR](https://github.com/c-yn/ConvIR).
