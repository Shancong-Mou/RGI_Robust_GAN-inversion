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
from utils.eval_inpainting import eval
import warnings
from utils.missing_generator import generate_missing
from utils.get_G import getGANTrainer


def main(args):
    import os
    import tqdm
    warnings.filterwarnings("ignore")
    RGI = RGI_(args) 
    test_img_dir = args.data_dir + '/'
    work_dir = args.output_dir + '/' + args.product_name + '/' +args.missing_type + '/'
    os.makedirs (work_dir, exist_ok= True)
    Res_img_dir = work_dir + 'Res_img/'
    os.makedirs (Res_img_dir, exist_ok= True)
    x_test = read_test(args, test_img_dir) # clean images
    x_test = transofrm_test_image(x_test)[:100]
    x_raw, mask = generate_missing(x_test, args) # image with missing
    if os.path.exists('./checkpoints/Pool.pkl'):
        with open('./checkpoints/Pool.pkl', 'rb') as f:
            RGI.Pool= pickle.load(f).to(RGI.device) 
    else:
        RGI.generate_pool()

    if os.path.exists(work_dir + 'Init_Z0.pkl'):
        with open(work_dir + 'Init_Z0.pkl', 'rb') as f:
            Init_Z0 = pickle.load(f) 
    else:
        Init_Z0 =[]
        RGI.Gnet, RGI.Dnet = getGANTrainer(args.gan_mdl_name)
        RGI.Gnet = RGI.Gnet.to(RGI.device)
        for i in range(len(x_raw)):# tqdm(range(len(x_raw)), desc = 'Initializing Progress'):
            RGI.input_img = x_raw[i:i+1].to(RGI.device)
            RGI.init_z0()
            Init_Z0.append(RGI.z0.clone())
        with open(work_dir + 'Init_Z0.pkl', 'wb') as f:
            pickle.dump(Init_Z0, f)
        # RGI.generate_pool()

    # save raw image
    cnt = 0
    for img in x_raw:
        grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
        cnt = cnt + 1
    
    from tqdm import tqdm
    import time
    import os

    #### RGI

    S_msk_all_RGI = []
    L_generated_all_RGI = [] 

    for i in tqdm(range(len(x_raw)), desc = 'RGI: Total Progress'):
        x = x_raw[i:i+1].clone().detach()
        [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = True, lam_mask=args.lam_mask_RGI, opt_G= False,  args =args, init_z0 = Init_Z0[i])
        S_msk_all_RGI.append(S_msk)
        L_generated_all_RGI.append(L_generated) 

    with open(work_dir +'S_msk_all_RGI_' + 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) + '.pkl', 'wb') as f:
        pickle.dump(S_msk_all_RGI, f)

    with open(work_dir +'L_generated_all_RGI_' + 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) + '.pkl', 'wb') as f:
        pickle.dump(L_generated_all_RGI, f)

    with open(work_dir +'L_generated_all_RGI_' + 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) + '.pkl', 'rb') as f:
        L_generated_all_RGI = pickle.load(f)
        cnt = 0
        for img in L_generated_all_RGI:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Rcnst_img_RGI#'+str(cnt)+'lam'+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+args.loss_type +'.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1
            # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

    with open(work_dir +'S_msk_all_RGI_'+ 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'.pkl', 'rb') as f:
        S_msk_all_RGI = pickle.load(f) 
        cnt = 0
        for img in S_msk_all_RGI:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Msk_img_RGI#'+str(cnt)+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) + '.png', grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1
        
    eval(x_raw, x_test, work_dir, 'RGI', args, args.lam_mask_RGI, write = True)

    #### R-RGI

    S_msk_all_R_RGI = []
    L_generated_all_R_RGI = [] 

    for i in tqdm(range(len(x_raw)), desc = 'R_RGI: Total Progress'):
        x = x_raw[i:i+1].clone().detach()
        [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = True, lam_mask=args.lam_mask_R_RGI, opt_G= True,  args =args, init_z0 = Init_Z0[i])
        S_msk_all_R_RGI.append(S_msk)
        L_generated_all_R_RGI.append(L_generated) 

    with open(work_dir +'L_generated_all_R_RGI_'+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+'.pkl', 'wb') as f:
        pickle.dump(L_generated_all_R_RGI, f)

    with open(work_dir +'S_msk_all_R_RGI_'+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+'.pkl', 'wb') as f:
        pickle.dump(S_msk_all_R_RGI, f)

    cnt = 0
    for img in x_raw:
        grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
        cnt = cnt + 1

    with open(work_dir +'L_generated_all_R_RGI_'+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+'.pkl', 'rb') as f:
        L_generated_all_R_RGI = pickle.load(f)
        cnt = 0
        for img in L_generated_all_R_RGI:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Rcnst_img_R_RGI#'+str(cnt)+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1

    with open(work_dir +'S_msk_all_R_RGI_'+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M)+'.pkl', 'rb') as f:
        S_msk_all_2 = pickle.load(f) 
        cnt = 0
        for img in S_msk_all_2:
            cnt = cnt + 1
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Msk_img_R_RGI#'+str(cnt)+'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(args.lam_mask_R_RGI) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'.png', grid.permute(1, 2, 0).cpu().numpy())
    eval(x_raw, x_test, work_dir, 'R_RGI', args, args.lam_mask_R_RGI, write = True)

    # Yeh without mask
    ##### Yeh_w_o_mask (l1 only)
    S_msk_all_Yeh_w_o_mask = [] # Yeh_w_o_mask detected mask 
    L_generated_Yeh_w_o_mask = [] # Yeh_w_o_mask recovered image 
    # loop over  images
    from tqdm import tqdm
    for i in tqdm(range(len(x_raw)), desc = 'Yeh w/o mask: Total Progress'):
        x = x_raw[i:i+1].clone().detach()
        args.loss_type = 'L1'
        [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = False, lam_mask=args.lam_mask_RGI, opt_G= False, args = args, true_msk = torch.zeros_like(x), init_z0 = Init_Z0[i])
        L_generated_Yeh_w_o_mask.append(L_generated) 
    # save 
    with open(work_dir +'L_generated_all_Yeh_w_o_mask.pkl', 'wb') as f:
        pickle.dump(L_generated_Yeh_w_o_mask, f)
    # save image
    cnt = 0
    for img in x_raw:
        grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
        cnt = cnt + 1
    # save true defect image
    with open(work_dir +'L_generated_all_Yeh_w_o_mask.pkl', 'rb') as f:
        L_generated_Yeh_w_o_mask = pickle.load(f)
        cnt = 0
        for img in L_generated_Yeh_w_o_mask:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Rcnst_img_Yeh_w_o_mask#'+str(cnt)+ args.loss_type + '.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1

    eval(x_raw, x_test, work_dir, 'Yeh_w_o_mask', args,args.lam_mask_RGI, write = True)    

    # Yeh with mask
    L_generated_Yeh_w_mask = [] # Yeh_w_mask recovered image 
    # loop over  images
    from tqdm import tqdm
    for i in tqdm(range(len(x_raw)), desc = 'Yeh w/ mask: Total Progress'):
        x = x_raw[i:i+1].clone().detach()
        true_mask =  mask[i:i+1]
        args.loss_type = 'L1'
        [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = False,lam_mask=args.lam_mask_RGI, opt_G = False, args = args, true_msk =  true_mask, init_z0 = Init_Z0[i])
        L_generated_Yeh_w_mask.append(L_generated) 
    # save 
    with open(work_dir +'L_generated_all_Yeh_w_mask.pkl', 'wb') as f:
        pickle.dump(L_generated_Yeh_w_mask, f)
    # save image
    cnt = 0
    for img in x_raw:
        grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
        cnt = cnt + 1
    # save true defect image
    with open(work_dir +'L_generated_all_Yeh_w_mask.pkl', 'rb') as f:
        L_generated_Yeh_w_mask = pickle.load(f)
        cnt = 0
        for img in L_generated_Yeh_w_mask:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Rcnst_img_Yeh_w_mask#'+str(cnt)+ args.loss_type + '.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1

    eval(x_raw, x_test, work_dir, 'Yeh_w_mask', args,args.lam_mask_RGI, write = True)    

    # Pan with mask 
    L_generated_Pan_w_mask = [] # Pan_w_o_mask recovered image 
    # loop over  images
    from tqdm import tqdm
    for i in tqdm(range(len(x_raw)), desc = 'Pan w/ mask: Total Progress'):
        x = x_raw[i:i+1].clone().detach()
        true_mask =  mask[i:i+1]
        args.loss_type = 'L1'
        [S_msk, L_generated] = RGI.RGI_optim(x, opt_M = False,lam_mask=args.lam_mask_RGI, opt_G= True, args = args, true_msk =  true_mask, init_z0 = Init_Z0[i])
        L_generated_Pan_w_mask.append(L_generated) 
    # save 
    with open(work_dir +'L_generated_all_Pan_w_mask.pkl', 'wb') as f:
        pickle.dump(L_generated_Pan_w_mask, f)
    # save image
    cnt = 0
    for img in x_raw:
        grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
        plt.imsave(work_dir +'Res_img/'+'raw_img#'+str(cnt)+'.png',  grid.permute(1, 2, 0).cpu().numpy())
        cnt = cnt + 1
    # save true defect image
    with open(work_dir +'L_generated_all_Pan_w_mask.pkl', 'rb') as f:
        L_generated_Pan_w_mask = pickle.load(f)
        cnt = 0
        for img in L_generated_Pan_w_mask:
            grid = torchvision.utils.make_grid(torch.tensor(img).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imsave(work_dir +'Res_img/'+'Rcnst_img_Pan_w_mask#'+str(cnt)+ args.loss_type + '.png',  grid.permute(1, 2, 0).cpu().numpy())
            cnt = cnt + 1

    eval(x_raw, x_test, work_dir, 'Pan_w_mask', args, args.lam_mask_RGI, write = True)  



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust GAN inversion for unsupervised pixwl-wise anomaly detection')
    # Image Path and Saving Path
    parser.add_argument('--data_dir',
                        default='./Data/celebaHQ/img_align_celeba_test_small_scale/',
                        help='Directory with images for inpainting')
    parser.add_argument('--output_dir',
                        default='./Semantic_inpainting_output',
                        help='Directory for storing results')


    # Parameters for robust GAN inversion
    parser.add_argument('--gan_mdl_name', default='celeba_cropped',
                        help='name for pretrained GAN model.')
    parser.add_argument('--product_name', default= 'celeba_small_scale',
                        help='name of the product')
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

    parser.add_argument('--lam_mask_RGI', default=0.07,
                        help="Trade-off parameter for sparsity mask penalty for RGI.", type=float)

    parser.add_argument('--lam_mask_R_RGI', default=0.07,
                        help="Trade-off parameter for sparsity mask penalty for R-RGI.", type=float)

    # Optimization Parameters
    parser.add_argument('--generate_new_pool', default = False, action='store_true',
                        help="learning rate for z.") 
    parser.add_argument('--lr_z', default= 0.1,
                        help="learning rate for z.", type= float) 
    parser.add_argument('--lr_M', default= 0.1,
                        help="learning rate for M.", type= float) 
    parser.add_argument('--lr_G', default= 0.00001,
                        help="learning rate for G.", type= float)  
    parser.add_argument('--iterations', default = 2000,
                        help='Number of optimization steps.', type=int)
    parser.add_argument('--start_fine_tune', default = 1500,
                        help='Number of steps to start fine tune generator.', type=int)
    parser.add_argument('--fix_mask', default = True, action='store_false',
                        help='Fix mask when optimizing G net.')   
    parser.add_argument('--task', default= 'Semantic_inpainting',
                        help="task type")                    
    # RGI/R-RGI missing setting
    parser.add_argument('--missing_size', default= 32,
                        help="center missing_size", type= int) 

    parser.add_argument('--missing_type', default= 'central_block',
                        help="missing type: central_block, random or irregular_mask") 
    # Video Settings
    parser.add_argument('--video', default=False, action='store_true', help='Save video. False for no video.')

    args, other_args = parser.parse_known_args()
    
    print(args)
    ### RUN
    import random
    import numpy as np
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    main(args)
