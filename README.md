# RGI: Robust GAN inversion 

## Installation

1. Create a virtual environment via `conda`.

   ```shell
   conda create -n RGI python=3.9
   conda activate RGI
   ```

2. If you don't already have pytorch or torchvision please have a look at https://pytorch.org/ as the installation command may vary depending on your OS and your version of CUDA.


3. Install requirements

   ```shell
   pip install -r requirements.txt
   ```

## Train from scratch

We use a PGGAN model from [pytorch-gan-zoo](https://github.com/facebookresearch/pytorch_GAN_zoo). If you are using your own dataset, please follow the follwing steps:
1. Prepare you own training data (for example the celebaHQ dataset, can be .png or .jpg format) 
 under ```./utils/pytorch-gan-zoo/Data/celebaHQ/img_align_celeba_train/```

2. Prepare your training configuration file (details can be found at [pytorch-gan-zoo](https://github.com/facebookresearch/pytorch_GAN_zoo#configuration-file-of-a-training-session)), and put it under ```./utils/pytorch-gan-zoo/```. You can find examples of training config files, such as ```batd3.json``` for BATD dataset and ```celeba_cropped.json``` for celebaHQ dataset.

3. lunch PGGAN training procedure using:

```
python train.py PGAN -c celeba_cropped.json --restart -n celeba_cropped
```
After training, the trained model chekpoint will be stored at: ```./utils/pytorch-gan-zoo/output_networks/celeba_cropped/```

### After training, you can use the trained Generator and Discriminator for image inpainting and anomaly detection (see the following sections).

## For mask-free semantic inpainting

### **Train**

For Reproducibility purpose,  please download the pretrained model through https://drive.google.com/file/d/1q7qfPvkP6GJk7JkDktZtPKjJuZ64iQUo/view?usp=sharing, named 'output_networks.zip'. unzip and and put it under 

```shell
./utils/pytorch-gan-zoo/
```
### **Test (small scale)**
Notice that runing all the experiments are time-consuming. We prepare a small-scale test set on block missing of celebA dataset for a quick perofrmance verification. 

There are 5 test images in the test set.

Please run the following script: 
```shell
python Semantic_inpainting.py
```

The result will be stored in './Semantic_inpainting_output/celeba_small_scale'. 

**Qualititive results** for a sepecific method is store in the txt file name: 
```shell
method+'_' + 'loss_type:'+ args.loss_type + 'use_mask' +str(args.use_mask) + ' Sparse maskpenalty:' + str(args.lam_mask) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt'
```

**Result images** are stored in './Semantic_inpainting_output/celeba_small_scale'. 
```shell
'Rcnst_img_'+method+'_' +'#'+str(image_number)+ 'L2.png'
```

Notice that the result might be slightly different from the result reported sicne we are using a small subset.

### **Test (large scale)**
The large-scale test on crack type of defect can be done by running the following script:
```shell
bash RUN_celebA_inpainting.sh
```

The result will be stored in './Semantic_inpainting_output/celeba_cropped'. 

**Qualititive results** for a sepecific method is store in the txt file name: 
```shell
method+'_' + 'loss_type:'+ args.loss_type + 'use_mask' +str(args.use_mask) + ' Sparse maskpenalty:' + str(args.lam_mask) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt'
```

**Result images** are stored in './Semantic_inpainting_output/celeba_cropped'. 
```shell
'Rcnst_img_'+method+'_' +'#'+str(image_number)+ args.loss_type + '.png'
```
Notice that the result might be slightly different from the result reported due to randomness.

## For pixel-wise anomaly detection

### **Train**

We use a PGGAN model from [pytorch-gan-zoo](https://github.com/facebookresearch/pytorch_GAN_zoo). 

For Reproducibility purpose,  please download the pretrained model through https://drive.google.com/file/d/1q7qfPvkP6GJk7JkDktZtPKjJuZ64iQUo/view?usp=sharing, named 'output_networks.zip'. unzip and and put it under 

```shell
./utils/pytorch-gan-zoo/
```

### **Test (small scale)**
Notice that runing all the experiments are time-consuming. We prepare a small-scale test set on crack type of defect for a quick perofrmance verification. 

There are 11 test images in the test set.

Please run the following script: 
```shell
python Anomaly_segmentation.py 
```

The result will be stored in './Anomaly_detection_output/BTAD3_small_scale/'. 

**Qualititive results** for a sepecific method are stored in the txt file name: 
```shell
method+'_' + 'loss_type:'+ args.loss_type + 'use_mask' +str(args.use_mask) + ' Sparse maskpenalty:' + str(args.lam_mask) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt'
```

**Result images** are stored in './Anomaly_detection_output/BTAD3_small_scale/Res_img/'. 

Raw segemntation mask:
```shell
'Msk_img'+ method+'#'+str(image_number)+'lam'+str(args.lam_mask) + args.loss_type +'.png'
```
Gray scale segemntation mask:
```shell
'Msk_img_'+ method +'_gry#'+str(image_number) + '_' + 'lam'+str(args.lam_mask)+'_' + args.loss_type +'.png'
```
Binary thresholding of segemntation mask:
```shell
'Msk_img_'+ method +'_thrsd#'+str(image_number) + '_' + 'lam'+str(args.lam_mask)+'_' + args.loss_type +'.png'
```

Notice that the result might be slightly different from the result reported sicne we are using a small subset.

### **Test (large scale)**
The large-scale test on crack type of defect can be done by running the following script:
```shell
bash RUN_Synthetic_defects.sh
```

The result will be stored in './Anomaly_detection_output/BTAD3'. 

**Qualititive results** for a sepecific method is store in the txt file name: 
```shell
method+'_' + 'loss_type:'+ args.loss_type + 'use_mask' +str(args.use_mask) + ' Sparse maskpenalty:' + str(args.lam_mask) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt'
```

**Result images** are stored in './Anomaly_detection_output/BTAD3/Res_img/'. 

Raw segemntation mask:
```shell
'Msk_img'+ method+'#'+str(image_number)+'lam'+str(args.lam_mask)+args.loss_type +'.png'
```
Gray scale segemntation mask:
```shell
'Msk_img_'+ method +'_gry#'+str(image_number) + '_' + 'lam'+str(args.lam_mask)+'_' + args.loss_type +'.png'
```
Binary thresholding of segemntation mask:
```shell
'Msk_img_'+ method +'_thrsd#'+str(image_number) + '_' + 'lam'+str(args.lam_mask)+'_' + args.loss_type +'.png'
```

Notice that the result might be slightly different from the result reported due to randomness.