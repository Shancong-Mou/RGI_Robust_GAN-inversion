
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision
import torch
import matplotlib.pyplot as plt
from utils.pytorch_ssim import ssim

def eval(x_raw,  x_test, work_dir, method, args, lam_mask, write = True):
    
    if method == 'Yeh_w_o_mask' or method == 'Yeh_w_mask' or method == 'Pan_w_mask':
        with open(work_dir +'L_generated_all_'+method+'.pkl', 'rb') as f:
            L_generated_all_2 = pickle.load(f)


    elif method == 'RGI' or method =='R_RGI':
        with open(work_dir +'L_generated_all_'+ method + '_' + 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(lam_mask) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) + '.pkl', 'rb') as f:
            L_generated_all_2 = pickle.load(f)

    ssim_res = []
    psnr_res = []
    for i in range(len(x_test)):
        ssim_res.append(ssim(x_test[i:i+1].float() , torch.tensor(L_generated_all_2[i:i+1])).item())
        psnr_res.append( psnr(x_test[i].cpu().numpy(), L_generated_all_2[i]) )
    ssim_res_mean = np.median(ssim_res)
    ssim_res_std = np.std(ssim_res)
    psnr_res_mean = np.median(psnr_res)
    psnr_res_std = np.std(psnr_res)

    print('loss_type:',args.loss_type, ' Sparse maskpenalty:',lam_mask, 'lam_discriminbator', args.lam_dis, ' iterations:', args.iterations, ' lr_z', args.lr_z, ' lr_G', args.lr_G, 'lr_M', args.lr_M )
    print('Result:----------'+method+'----------')
    print('ssim_mean:',ssim_res_mean)
    print('ssim_std:',ssim_res_std)            
    print('psnr_mean:',psnr_res_mean)
    print('psnr_std:',psnr_res_std)
    

    if write:
        with open(work_dir + method + '_' + 'loss_type:'+ args.loss_type  + ' Sparse maskpenalty:' + str(lam_mask) + 'lam_discriminbator:' +str(args.lam_dis) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt', 'w') as file:
            print('loss_type:',args.loss_type, ' Sparse maskpenalty:',lam_mask, 'lam_discriminbator', args.lam_dis, ' iterations:', args.iterations, ' lr_z', args.lr_z, ' lr_G', args.lr_G, 'lr_M', args.lr_M,  file=file )
            print('Result:----------'+method+'----------',file=file)
            print('ssim_mean:',ssim_res_mean, file=file)
            print('ssim_std:',ssim_res_std, file=file)             
            print('psnr_mean:',psnr_res_mean, file=file)
            print('psnr_std:',psnr_res_std, file=file)    
    return

def psnr(img1, img2):
    mse = np.linalg.norm(standize(img1)-standize(img2))**2/(128*128*3)
    max_val = np.max(standize(img1))**2
    return 10*np.log10(max_val/mse)

def rgb2gry(img):
    return 0.2989*img[0,:,:] + 0.5870*img[1,:,:] + 0.1140*img[2,:,:]

def standize(img):
    img_abs = img
    return (img_abs-img_abs.min())/(img_abs.max()-img_abs.min()+1e-10)