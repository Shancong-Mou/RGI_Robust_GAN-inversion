import argparse
import math, time
import torch
from itertools import product
import sys
import importlib
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2
import pickle
import numpy as np
from PIL import Image
import glob
import json
from tqdm import tqdm
import time
import os
from robust_gan_inversion import RGI_
from utils.read_data import read_test, read_test_lbl, transofrm_test_image
from utils.eval import eval
from utils.eval import standize
from utils.eval import diceloss
from utils.eval import rgb2gry

import warnings
from sklearn.metrics import roc_auc_score
import json


def main(args):
    warnings.filterwarnings("ignore")
    # product_names = ['transistor', 'capsule', 'metal_nut', 'pill', 'zipper','tile', 'hazelnut', 'carpet', 'cable', 'leather', 'screw', 'toothbrush', 'bottle', 'grid','wood' ]
    RGI = RGI_(args) 
    product_data_dir = args.data_dir +'/' + args.product_name  +'/'
    product_test_data_dir = product_data_dir + 'test/'
    product_gt_data_dir = product_data_dir +'ground_truth/'
    import os 
    defect_types = [args.defect_type]
    # sorted(os.listdir(product_test_data_dir)) # sorted defect types
    # if 'good' in defect_types:
    #     defect_types.remove('good')
    ### product level tracker
    # prodcut_run_info = {}
    # prodcut_run_info['product_name'] = args.product_name
    # prodcut_run_info['defect_types'] = defect_types
    # prodcut_run_info['use_mask'] = args.use_mask
    # prodcut_run_info['lr_G'] = args.lr_G
    # prodcut_run_info['lam_mask'] = args.lam_mask

    # prodcut_level_AnoGAN = {} # record the auroc of different defects
    # prodcut_level_AnoGAN['AUROC_mean'] = []
    # prodcut_level_AnoGAN['AUROC_std'] = []
    # prodcut_level_AnoGAN['AUROC_average'] = [] # this calculates the average AUROC by vectorized and concatenate all defects in the product 
    # prodcut_level_AnoGAN['Dice_mean'] = []
    # prodcut_level_AnoGAN['Dice_std'] = []
    # prodcut_level_AnoGAN['Dice_average'] = [] # this calculates the average Dice by vectorized and concatenate all defects in the product 

    # prodcut_level_RGI = {}
    # prodcut_level_RGI['AUROC_mean'] = []
    # prodcut_level_RGI['AUROC_std'] = []
    # prodcut_level_RGI['AUROC_average'] = []
    # prodcut_level_RGI['Dice_mean'] = []
    # prodcut_level_RGI['Dice_std'] = []
    # prodcut_level_RGI['Dice_average'] = []  


    # prodcut_level_R_RGI = {}
    # prodcut_level_R_RGI['AUROC_mean'] = []
    # prodcut_level_R_RGI['AUROC_std'] = []
    # prodcut_level_R_RGI['AUROC_average'] = []
    # prodcut_level_R_RGI['Dice_mean'] = []
    # prodcut_level_R_RGI['Dice_std'] = []
    # prodcut_level_R_RGI['Dice_average'] = []


    # S_all_product_anoGAN = []
    # S_all_product_RGI = []
    # S_all_product_R_RGI = []
    # x_test_lbl_all_product = []

    #####
    for defect_type in defect_types:
        print('product name:',args.product_name)
        print('Type of defects:', defect_types)
        print('Running:', defect_type)
        os.makedirs(args.output_dir, exist_ok=True)
        work_dir = args.output_dir +'/'+ args.product_name +'/' + defect_type +'/'
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(work_dir + 'Res_img/', exist_ok=True)
        # read test images
        test_img_dir = product_test_data_dir + defect_type +'/' # get test image deirectory 
        x_test = read_test(args, test_img_dir)

        x_raw = transofrm_test_image(x_test)
        # read test images labels
        gt_img_dir = product_gt_data_dir + defect_type + '/' # get test image deirectory 
        x_test_lbl = read_test_lbl(args, gt_img_dir)
        # prodcut level tracker
        # for i in range(len(x_test_lbl)):
        #     # print(len(standize(x_test_lbl[i]).round().astype(int).flatten()))
        #     x_test_lbl_all_product.extend(standize(x_test_lbl[i]).round().astype(int).flatten())
        # print(len(x_test_lbl_all_product))
        # generate 1000 random 500x1 vectors as the pool for latent vetor
        if os.path.exists('./checkpoints/Pool.pkl'):
            with open('./checkpoints/Pool.pkl', 'rb') as f:
                RGI.Pool= pickle.load(f).to(RGI.device) 
        else:
            RGI.generate_pool()

        #### AnoGAN (l1 only)
        S_msk_all_AnoGAN = [] # AnoGAN detected anomaly 
        L_generated_AnoGAN = [] # AnoGAN recovered background 
        # loop over  images
        from tqdm import tqdm
        for i in tqdm(range(len(x_raw)), desc = 'AnoGAN: Total Progress'):
            x = x_raw[i:i+1].clone().detach()
            args.loss_type == 'L1'
            [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = False, lam_mask=args.lam_mask_RGI, opt_G= False, args =args)
            S_msk_all_AnoGAN.append(torch.abs(x-L_generated))
            L_generated_AnoGAN.append(L_generated) 
            # # prodcut level tracker
            # S_all_product_anoGAN.extend(standize(rgb2gry(torch.abs(x-L_generated).numpy().squeeze())).flatten())
        # save 
        with open(work_dir +'L_generated_all_AnoGAN.pkl', 'wb') as f:
            pickle.dump(L_generated_AnoGAN, f)
        with open(work_dir +'S_msk_all_AnoGAN.pkl', 'wb') as f:
            pickle.dump(S_msk_all_AnoGAN, f)
        # save image
        cnt = 0
        for img in x_raw:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1
        # save true defect image
        with open(work_dir +'L_generated_all_AnoGAN.pkl', 'rb') as f:
            L_generated_AnoGAN = pickle.load(f)
            cnt = 0
            for img in L_generated_AnoGAN:
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Rcnst_img_AnoGAN#'+str(cnt)+ args.loss_type + '.png',  grid.permute(1, 2, 0).cpu().numpy())
                cnt = cnt + 1
        with open(work_dir +'S_msk_all_AnoGAN.pkl', 'rb') as f:
            S_msk_all_AnoGAN = pickle.load(f) 
            cnt = 0
            for img in S_msk_all_AnoGAN:
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Msk_img_AnoGAN#'+str(cnt)+ args.loss_type +'.png', grid.permute(1, 2, 0).cpu().numpy())
                cnt = cnt + 1

        dice_mean_AnoGAN, dice_std_AnoGAN, ROC_AUC_mean_AnoGAN, ROC_AUC_std_AnoGAN = eval(x_raw, x_test_lbl, work_dir, 'AnoGAN', args, args.lam_mask_RGI, write = True)

        ### product level tracker
        # prodcut_level_AnoGAN['AUROC_mean'].append(ROC_AUC_mean_AnoGAN)
        # prodcut_level_AnoGAN['AUROC_std'].append(ROC_AUC_std_AnoGAN) 
        # prodcut_level_AnoGAN['Dice_mean'].append(dice_mean_AnoGAN)
        # prodcut_level_AnoGAN['Dice_std'].append(dice_std_AnoGAN)
        
        ###

        # R-RGI

        S_msk_all_R_RGI = []
        L_generated_all_R_RGI = [] 

        for i in tqdm(range(len(x_raw)), desc = 'R_RGI: Total Progress'):
            x = x_raw[i:i+1].clone().detach()
            args.loss_type = 'L2'
            [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = True,lam_mask=args.lam_mask_R_RGI, opt_G= True,  args =args)
            if args.use_mask:
                S_msk_all_R_RGI.append(S_msk)
                L_generated_all_R_RGI.append(L_generated) 
                # prodcut level tracker
                # S_all_product_R_RGI.extend(standize(rgb2gry(torch.abs(torch.tensor(S_msk)).numpy().squeeze())).flatten())
            else:
                S_msk_all_R_RGI.append(torch.abs(x-L_generated))
                L_generated_all_R_RGI.append(L_generated)
                # prodcut level tracker
                # S_all_product_R_RGI.extend(standize(rgb2gry(torch.abs(torch.tensor(S_msk)).numpy().squeeze())).flatten())

        with open(work_dir +'L_generated_all_R_RGI_'+str(args.lam_mask_R_RGI)+'.pkl', 'wb') as f:
            pickle.dump(L_generated_all_R_RGI, f)

        with open(work_dir +'S_msk_all_R_RGI_'+str(args.lam_mask_R_RGI)+'.pkl', 'wb') as f:
            pickle.dump(S_msk_all_R_RGI, f)

        cnt = 0
        for img in x_raw:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1

        with open(work_dir +'L_generated_all_R_RGI_'+str(args.lam_mask_R_RGI)+'.pkl', 'rb') as f:
            L_generated_all_R_RGI = pickle.load(f)
            cnt = 0
            for img in L_generated_all_R_RGI:
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Rcnst_img_R_RGI#'+str(cnt)+'lam'+str(args.lam_mask_R_RGI)+args.loss_type +'.png',  grid.permute(1, 2, 0).cpu().numpy())
                cnt = cnt + 1

        with open(work_dir +'S_msk_all_R_RGI_'+str(args.lam_mask_R_RGI)+'.pkl', 'rb') as f:
            S_msk_all_2 = pickle.load(f) 
            cnt = 0
            for img in S_msk_all_2:
                cnt = cnt + 1
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Msk_img_R_RGI#'+str(cnt)+'lam'+str(args.lam_mask_R_RGI)+args.loss_type +'.png', grid.permute(1, 2, 0).cpu().numpy())
        dice_mean_R_RGI, dice_std_R_RGI, ROC_AUC_mean_R_RGI, ROC_AUC_std_R_RGI = eval(x_raw, x_test_lbl, work_dir, 'R_RGI', args, args.lam_mask_R_RGI, write = True)
        # prodcut_level_R_RGI['AUROC_mean'].append(ROC_AUC_mean_R_RGI)
        # prodcut_level_R_RGI['AUROC_std'].append(ROC_AUC_std_R_RGI)
        # prodcut_level_R_RGI['Dice_mean'].append(dice_mean_R_RGI)
        # prodcut_level_R_RGI['Dice_std'].append(dice_std_R_RGI)

#### RGI
        from tqdm import tqdm
        import time
        import os

        S_msk_all_RGI = []
        L_generated_all_RGI = [] 

        for i in tqdm(range(len(x_raw)), desc = 'RGI: Total Progress'):
            x = x_raw[i:i+1].clone().detach()
            args.loss_type = 'L2'
            [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = True, lam_mask=args.lam_mask_RGI, opt_G= False,  args =args)
            S_msk_all_RGI.append(S_msk)
            L_generated_all_RGI.append(L_generated) 
            # prodcut level tracker
            # S_all_product_RGI.extend(standize(rgb2gry(torch.abs(torch.tensor(S_msk)).numpy().squeeze())).flatten())

        with open(work_dir +'S_msk_all_RGI_' + str(args.lam_mask_RGI) + '.pkl', 'wb') as f:
            pickle.dump(S_msk_all_RGI, f)

        with open(work_dir +'L_generated_all_RGI_' + str(args.lam_mask_RGI) + '.pkl', 'wb') as f:
            pickle.dump(L_generated_all_RGI, f)

        with open(work_dir +'L_generated_all_RGI_' + str(args.lam_mask_RGI) + '.pkl', 'rb') as f:
            L_generated_all_RGI = pickle.load(f)
            cnt = 0
            for img in L_generated_all_RGI:
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Rcnst_img_RGI#'+str(cnt)+'lam'+str(args.lam_mask_RGI)+args.loss_type +'.png',  grid.permute(1, 2, 0).cpu().numpy())
                cnt = cnt + 1
                # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

        with open(work_dir +'S_msk_all_RGI_'+str(args.lam_mask_RGI)+'.pkl', 'rb') as f:
            S_msk_all_RGI = pickle.load(f) 
            cnt = 0
            for img in S_msk_all_RGI:
                grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imsave(work_dir +'Res_img/'+'Msk_img_RGI#'+str(cnt)+'lam'+str(args.lam_mask_RGI)+ args.loss_type + '.png', grid.permute(1, 2, 0).cpu().numpy())
                cnt = cnt + 1
            
        dice_mean_RGI, dice_std_RGI, ROC_AUC_mean_RGI, ROC_AUC_std_RGI = eval(x_raw, x_test_lbl, work_dir, 'RGI', args, args.lam_mask_RGI, write = True)
        # prodcut_level_RGI['AUROC_mean'].append(ROC_AUC_mean_RGI)
        # prodcut_level_RGI['AUROC_std'].append(ROC_AUC_std_RGI)
        # prodcut_level_RGI['Dice_mean'].append(dice_mean_RGI)
        # prodcut_level_RGI['Dice_std'].append(dice_std_RGI)

    #calculate product level results
    # AnoGAN
    # prodcut_level_AnoGAN['AUROC_average']  = roc_auc_score(x_test_lbl_all_product, S_all_product_anoGAN)
    # s_ = []
    # for thres in np.linspace(0,1,101):
    #     s_.append(diceloss(S_all_product_anoGAN,x_test_lbl_all_product,thres))
    # prodcut_level_AnoGAN['Dice_average'] =np.max(s_)
    # # RGI
    # prodcut_level_RGI['AUROC_average']  = roc_auc_score(x_test_lbl_all_product, S_all_product_RGI)
    # s_ = []
    # for thres in np.linspace(0,1,101):
    #     s_.append(diceloss(S_all_product_RGI,x_test_lbl_all_product,thres))
    # prodcut_level_RGI['Dice_average'] =np.max(s_)
    # #R_RGI
    # prodcut_level_R_RGI['AUROC_average']  = roc_auc_score(x_test_lbl_all_product, S_all_product_R_RGI)
    # s_ = []
    # for thres in np.linspace(0,1,101):
    #     s_.append(diceloss(S_all_product_R_RGI,x_test_lbl_all_product,thres))
    # prodcut_level_R_RGI['Dice_average'] =np.max(s_)    

    # # summrize
    # summary = [prodcut_run_info, prodcut_level_AnoGAN, prodcut_level_RGI, prodcut_level_R_RGI]
    # with open("result.json", "w") as final:
    #     json.dump(summary, final)
        
    # print('--------summary------')
    # print(summary)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust GAN inversion for unsupervised pixwl-wise anomaly detection')
    # Image Path and Saving Path
    parser.add_argument('--data_dir',
                        default='./Data/BTAD',
                        help='Directory with images for anomaly detection')
    parser.add_argument('--output_dir',
                        default='./Anomaly_detection_output',
                        help='Directory for storing results')
    parser.add_argument('--product_name', default= 'BTAD3_small_scale',
                        help='name of the product')  
    parser.add_argument('--defect_type', default= 'crack',
                        help='name of the defect_type') 
    # Parameters for robust GAN inversion
    parser.add_argument('--gan_mdl_name', default='batd3',
                        help='name for pretrained GAN model.')
    parser.add_argument('--latent_z_dim', default=512,
                        help='dimension of latent z.', type=int)
      

    # Loss Parameters
    parser.add_argument('--img_size', default=128,
                        help='Size of input images', type=int)
    parser.add_argument('--loss_type', default='L2',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    parser.add_argument('--lam_dis', default= 0.1,
                        help="discriminator loss weight", type=float)  
    parser.add_argument('--vgg_layer', default=20,
                        help='The layer used in perceptual loss.', type=int)
    parser.add_argument('--l1_lambda', default=0.1,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    parser.add_argument('--percpt_lambda', default=0.1,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
    parser.add_argument('--nll_lambda', default=0.1,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Negative log-likelihood loss.", type=float)

    parser.add_argument('--lam_mask_RGI', default=0.12,
                        help="Trade-off parameter for sparsity mask penalty.", type=float)

    parser.add_argument('--lam_mask_R_RGI', default=0.12,
                        help="Trade-off parameter for sparsity mask penalty.", type=float)

    # Optimization Parameters
    parser.add_argument('--generate_new_pool', default= False,
                        help="learning rate for z.", type= bool) 
    parser.add_argument('--lr_z', default= 0.1,
                        help="learning rate for z.", type= float) 
    parser.add_argument('--lr_M', default= 0.1,
                        help="learning rate for M.", type= float) 
    parser.add_argument('--lr_G', default= 0.00001,
                        help="learning rate for G.", type= float) 
    parser.add_argument('--iterations', default = 2000,
                        help='Number of optimization steps.', type=int)
    parser.add_argument('--start_fine_tune', default = 500,
                        help='Number of steps to start fine tune generator.', type=int)
    parser.add_argument('--fix_mask', default = False,
                        help='Number of steps to fix mask.', type=bool) 
    parser.add_argument('--task', default= 'Anomaly_segmentation',
                        help="task type") 
    # RGI/R-RGI defect detection setting
    parser.add_argument('--use_mask', default= True,
                        help="if using the mask as defect indicator, otherwise, use residual.", type= bool) 

    # Video Settings
    parser.add_argument('--video', type=bool, default=False, help='Save video. False for no video.')

    args, other_args = parser.parse_known_args()

    ### RUN
    import random
    import numpy as np
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    main(args)
