# Solving Inverse Computational Imaging Problems using Deep Pixel-level Prior

Code for Single Pixel Camera reconstruction corresponding to the paper : (https://arxiv.org/abs/1802.09850)
Derived from the original PixelCNN++ code which can be found [here](https://github.com/openai/pixel-cnn)

## Results

- Single Pixel Camera reconstruction with 25% measurement rate

Original Image          |  Masked Image      |  During Inference   | Recovered Image
:-------------------------:|:-------------------------: | :-------------------------:|:-------------------------:
![](https://github.com/adaveiitm/deep-pixel-level-prior/tree/master/images/parrot_cropped.jpg)  |  ![](https://github.com/adaveiitm/deep-pixel-level-prior/tree/master/images/initial_img.png) | ![](https://github.com/adaveiitm/deep-pixel-level-prior/tree/master/images/during_process.gif) | ![](https://github.com/adaveiitm/deep-pixel-level-prior/tree/master/images/reconstructed_img.png)

## Requirements

- Tensorflow 1.2.0

- GPU with CUDA 8.0

## Usage

1. A PixelCNN++ model trained on 64x64 ImageNet can be found [here](https://drive.google.com/drive/folders/1hc8ycxvQB0gKdFbraqI0dj5JbpMGHe6l). Download the three files and save them in the `saves` folder.

2. For SPC reconstruction at 25% measurement rate with default parameters, run `python reconstruct_spc.py`
