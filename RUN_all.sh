#!/usr/bin/sh
conda activate RGI

# semantic image inpainting
python Semantic_inpainting.py # small example
### celeba
# central block missing
python Semantic_inpainting.py --lam_mask_RGI  0.07 --lam_mask_R_RGI  0.07 --product_name celeba_cropped --data_dir './Data/celebaHQ/img_align_celeba_test/' --lam_dis 10.0
# random missing
python Semantic_inpainting.py --lam_mask_RGI  0.1 --lam_mask_R_RGI  0.1 --product_name celeba_cropped --data_dir './Data/celebaHQ/img_align_celeba_test/' --lam_dis 10.0 --missing_type random --missing_size 45 
# irregular missing
srun --exclusive -n 1 -N 1 --mem-per-gpu=24G python Semantic_inpainting.py --lam_mask_RGI  0.1 --lam_mask_R_RGI  0.1 --product_name celeba_cropped --data_dir './Data/celebaHQ/img_align_celeba_test/' --lam_dis 10.0 --missing_type irregular_mask 


### sf car
# central block missing
python Semantic_inpainting.py --lam_mask_RGI  0.9 --lam_mask_R_RGI  0.9 --product_name SF_car --gan_mdl_name car --data_dir './Data/car_crop/SF_car_test/' --missing_size 16 --loss_type L1 --iterations 3000 --lam_dis 0.0 --fix_mask &
# random missing
python Semantic_inpainting.py --lam_mask_RGI  1.2 --lam_mask_R_RGI  1.2 --product_name SF_car --gan_mdl_name car --data_dir './Data/car_crop/SF_car_test/'   --loss_type L1 --iterations 3000 --lam_dis 0.01 --fix_mask --missing_type random 
# irregular missing
python Semantic_inpainting.py --lam_mask_RGI  0.9 --lam_mask_R_RGI  0.9 --product_name SF_car --gan_mdl_name car --data_dir './Data/car_crop/SF_car_test/' --missing_size 16 --loss_type L1 --iterations 3000 --lam_dis 0.0 --fix_mask --missing_type irregular_mask

# lsun bedroom
# central block missing
python Semantic_inpainting.py --lam_mask_RGI  0.7 --lam_mask_R_RGI  0.7 --product_name LSUN_bedroom --gan_mdl_name lsun_bedroom --data_dir './Data/lsun/LSUN_bedroom_test/' --missing_size 16 --loss_type L1 --iterations 3000 --lam_dis 0.01 --fix_mask
# random missing
python Semantic_inpainting.py --lam_mask_RGI  1.2 --lam_mask_R_RGI  1.2 --product_name LSUN_bedroom --gan_mdl_name lsun_bedroom --data_dir './Data/lsun/LSUN_bedroom_test/'   --loss_type L1 --iterations 3000  --fix_mask --lam_dis 0.0 --missing_type random
# irregular missing
python Semantic_inpainting.py --lam_mask_RGI  0.8 --lam_mask_R_RGI  0.7 --product_name LSUN_bedroom --gan_mdl_name lsun_bedroom --data_dir './Data/lsun/LSUN_bedroom_test/' --missing_size 16 --loss_type L1 --iterations 3000 --fix_mask --lam_dis 10.0 --missing_type irregular_mask

# Anomaly detection
python Anomaly_segmentation.py --lam_mask_RGI  0.4 --lam_mask_R_RGI  0.12 # small example
# crack
python Anomaly_segmentation.py --lam_mask_RGI  0.4 --lam_mask_R_RGI  0.12 --defect_type crack --product_name BTAD3
# irregular
python Anomaly_segmentation.py --lam_mask_RGI  0.4 --lam_mask_R_RGI  0.1 --defect_type irregular --product_name BTAD3
# large
python Anomaly_segmentation.py --lam_mask_RGI  0.4 --lam_mask_R_RGI  0.06 --defect_type large --product_name BTAD3
# scratch
python Anomaly_segmentation.py --lam_mask_RGI  0.4 --lam_mask_R_RGI  0.14 --defect_type scratch --product_name BTAD3


