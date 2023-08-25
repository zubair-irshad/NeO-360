# NEO 360: Neural Fields for Sparse View Synthesis of Outdoor Scenes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)<img src="demo/Pytorch_logo.png" width="10%">

This repository is the pytorch implementation of our paper:
<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="demo/tri-logo.png" width="25%"/>
</a>

**NEO 360: Neural Fields for Sparse View Synthesis of Outdoor Scenes**<br>
[__***Muhammad Zubair Irshad***__](https://zubairirshad.com), [Sergey Zakharov](https://zakharos.github.io/), [Katherine Liu](https://www.thekatherineliu.com/), [Vitor Guizilini](https://www.linkedin.com/in/vitorguizilini), [Thomas Kollar](http://www.tkollar.com/site/), [Adrien Gaidon](https://adriengaidon.com/), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/), [Rares Ambrus](https://www.tri.global/about-us/dr-rares-ambrus) <br>
International Conference on Computer Vision (ICCV), 2023<br>

[[Project Page](https://zubair-irshad.github.io/projects/neo360.html)] [[arXiv](https://arxiv.org/abs/2308.12967)] [[PDF](https://arxiv.org/pdf/2308.12967.pdf)] [[Video](https://youtu.be/avmylyL_V8c?si=eeTPhl0xJxM3fSF7)]


<p align="center">
<img src="demo/NEO_Website_1.jpg" width="100%">
</p>

<p align="center">
<img src="demo/NEO_Architecture.JPG" width="100%">
</p>

### Code Coming Soon!

## 📊 Dataset

### NERDS 360 Multi-View dataset for Outdoor Scenes

NeRDS 360: "NeRF for Reconstruction, Decomposition and Scene Synthesis of 360° outdoor scenes” dataset comprising 75 unbounded scenes with full multi-view annotations and diverse scenes for generalizable NeRF training and evaluation.

<p align="center">
<img src="demo/github_dataset.gif" width="100%">
</p>

#### Download the dataset:
* [NERDS360 Training Set](https://tri-ml-public.s3.amazonaws.com/github/neo360/datasets/PDMultiObjv6.tar.gz) - 75 Scenes (19.5 GB)
* [NERDS360 Test Set](https://tri-ml-public.s3.amazonaws.com/github/neo360/datasets/PD_v6_test.tar.gz) - 5 Scenes (2.1 GB)

#### Visualizing the dataset (Coming Soon):
We will release our visualization scripts to generate visualziations like below i.e. plot accumulated pointclouds, multi-view camera annotations etc. 

<p align="center">
<img src="demo/cameras.gif" width="100%">
</p>

## Citation

If you find this repository or our NERDS 360 dataset useful, please consider citing:

```
@inproceedings{irshad2023neo360,
  title={NeO 360: Neural Fields for Sparse View Synthesis of Outdoor Scenes},
  author={Muhammad Zubair Irshad and Sergey Zakharov and Katherine Liu and Vitor Guizilini and Thomas Kollar and Adrien Gaidon and Zsolt Kira and Rares Ambrus},
  journal={Interntaional Conference on Computer Vision (ICCV)},
  year={2023},
  url={https://arxiv.org/abs/2308.12967},
}
```