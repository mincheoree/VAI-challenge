# Code for Hanriver Data Challenge

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
