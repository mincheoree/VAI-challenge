# Code for Hanriver Data Challenge

## Setup 

# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch >= 1.1
* CUDA >= 9.0
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv)

```
pip install -r requirements.txt 
```

```shell
python setup.py develop
```

## Training 
```
bash scripts/dist_train.sh 2 --cfg_file cfgs/da-kitti-custom_models/secondiou/secondiou_ped.yaml --extra_tag NAME_OF_MODEL --workers 4 --batch_size 8 
```

## Inference/Visualization
```
python tools/demo_custom.py --cfg_file cfgs/da-kitti-custom_models/secondiou/secondiou_ped.yaml --ckpt ../ckpt_challenge/ped_upsample_80.pth --data_path ../data/custom/challenge_data/

```

## Acknowledgement
The codebase is built upon DTS: Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection and OpenPCDet.
