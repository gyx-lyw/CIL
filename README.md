# Official code for CCNet with ITBL for indoor crowd counting
## Code

### Dataset

Class B

### Install dependencies

torch >= 1.0 torchvision opencv numpy scipy  

###  Train and Test

1、 Pre-Process Data (resize image and split train/validation)

```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```

2、 Train model

```
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```

3、 Test Model
```
python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>
