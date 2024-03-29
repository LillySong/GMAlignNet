## Requirements
* python 3.8
* pytorch 0.4 or 1.0
* nibabel
* pickle 
* imageio
* pyyaml

## Implementation

Download the BraTS2018 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=10 is recommended.The total training time is about 12 hours and the average prediction time for each volume is 2.3 seconds when using randomly cropped volumes of size 128×128×128 and batch size 10 on two parallel Nvidia Tesla K40 GPUs for 800 epochs.

```
python train_all.py --gpu=0,1,3,4 --cfg=GMAlignNet --batch_size=12
```

### Test

You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=False --postprocess=True --snapshot=True --restore=model_last.pth --cfg=HDC_Net --gpu=0
```
Then make a submission to the online evaluation server.

## Acknowledge

1. [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
2. [BraTS2018-tumor-segmentation](https://github.com/ieee820/BraTS2018-tumor-segmentation)
3. [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

# GMAlignNet
