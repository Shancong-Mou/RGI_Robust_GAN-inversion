
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import torchvision
import torch
import matplotlib.pyplot as plt

def eval(x_raw, x_test_lbl, work_dir, method, args,lam_mask, write = True):
    
    if method == 'AnoGAN':
        # with open(work_dir +'L_generated_all_'+method+'.pkl', 'rb') as f:
        #     L_generated_all_2 = pickle.load(f)
        with open(work_dir +'S_msk_all_'+method+'.pkl', 'rb') as f:
            S_msk_all_2 = pickle.load(f) 

    elif method == 'RGI' or method == 'R_RGI':
        # with open(work_dir +'L_generated_all_'+ method + '_' + str(args.lam)+'.pkl', 'rb') as f:
        #     L_generated_all_2 = pickle.load(f)
        with open(work_dir +'S_msk_all_'+ method + '_' + str(lam_mask)+'.pkl', 'rb') as f:
            S_msk_all_2 = pickle.load(f) 



    ROC_AUC = []
    for i in range(len(x_raw)):
        S_msk = S_msk_all_2[i].squeeze()
        ROC_AUC.append(roc_auc_score(standize(x_test_lbl[i]).round().astype(int).flatten(), standize(rgb2gry(S_msk)).flatten() ))
    ROC_AUC_std = np.std(ROC_AUC)
    ROC_AUC_mean = np.mean(ROC_AUC)

    scoreGID = []
    thresholds = []
    for i in range(len(x_raw)):
        S_msk = np.asarray(S_msk_all_2[i]).squeeze()
        s_ = []
        for thres in np.linspace(0,1,101):
            s_.append(diceloss(rgb2gry(S_msk),x_test_lbl[i],thres))
        scoreGID.append(np.max(s_))
        # write thresholded image
        img = standize(rgb2gry(S_msk))
        plt.imsave(work_dir +'Res_img/'+'Msk_img_'+ method +'_gry#'+str(i) + '_' + 'lam'+str(lam_mask)+'_' + args.loss_type +'.png', img, cmap='gray', vmin=0, vmax=1)
        img[img > np.argmax(s_)/100] = 1
        img[img <= np.argmax(s_)/100] = 0
        plt.imsave(work_dir +'Res_img/'+'Msk_img_'+ method +'_thrsd#'+str(i) + '_' + 'lam'+str(lam_mask)+'_' + args.loss_type +'.png', img, cmap='gray', vmin=0, vmax=1)
    dice_mean = np.mean(scoreGID)
    dice_std = np.std(scoreGID)


    print('loss_type:',args.loss_type, ' Sparse maskpenalty:',lam_mask, ' iterations:', args.iterations, ' lr_z', args.lr_z, ' lr_G', args.lr_G, 'lr_M', args.lr_M )
    print('Result:----------'+method+'----------')
    print('dice_mean:',dice_mean)
    print('dice_std:',dice_std)            
    print('ROC_AUC_mean:',ROC_AUC_mean)
    print('ROC_AUC_std:',ROC_AUC_std)
    

    if write:
        with open(work_dir+method+'_' + 'loss_type:'+ args.loss_type + 'use_mask' +str( args.use_mask) + ' Sparse maskpenalty:' + str(lam_mask) + ' iterations:' +str(args.iterations) + ' lr_z' +str(args.lr_z) + ' lr_G' +str( args.lr_G) + 'lr_M' + str(args.lr_M) +'_log.txt', 'w') as file:
            print('loss_type:',args.loss_type, 'use_mask', args.use_mask, ' Sparse maskpenalty:',lam_mask, ' iterations:', args.iterations, ' lr_z', args.lr_z, ' lr_G', args.lr_G, 'lr_M', args.lr_M, file=file )
            print('Result:----------'+method+'----------',file=file)
            print('dice_mean:',dice_mean, file=file)
            print('dice_std:',dice_std, file=file)             
            print('ROC_AUC_mean:',ROC_AUC_mean, file=file)
            print('ROC_AUC_std:',ROC_AUC_std, file=file)    
    return dice_mean, dice_std, ROC_AUC_mean, ROC_AUC_std

def diceloss(y_pred,y_true, thres):
    yp = np.abs(y_pred)
    yp = (yp-np.min(yp))/(np.max(yp) - np.min(yp)+1e-12)
    
    yp[yp>thres] = 1
    yp[yp<=thres] = 0

    yt = y_true
    yt = (yt-np.min(yt))/(np.max(yt) - np.min(yt)+1e-12)
    yt[yt>thres] = 1
    yt[yt<=thres] = 0
    y_sum = yp+yt

    y_subt = yp*yt
    dice = 2*np.sum(y_subt)/(np.sum(y_sum)+1e-6)
    return dice

def rgb2gry(img):
    img_abs = np.abs(img)
    return 0.2989*img_abs[0,:,:] + 0.5870*img_abs[1,:,:] + 0.1140*img_abs[2,:,:]

def standize(img):
    img_abs = np.abs(img)
    return (img_abs-img_abs.min())/(img_abs.max()-img_abs.min()+1e-10)