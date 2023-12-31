
# import trained network

import os
import sys
import importlib
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import pickle
import time
import torch.nn.functional as F
from torch.autograd import Variable
from utils.get_G import getGANTrainer
from utils.losses import PerceptLoss
from utils.nethook import subsequence
# from models.utils.utils import getVal, getLastCheckPoint, loadmodule
# from models.utils.config import getConfigOverrideFromParser, \
#     updateParserWithConfig

import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

import json

class RGI_:
    def __init__(self, args):
        ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        self.gan_mdl_name = args.gan_mdl_name # gan_mdl_name 
        self.nz = args.latent_z_dim # dimension of latent z space
        # define optimizer parameters for RGI_optim.
        self.lams_percpt = args.percpt_lambda
        self.lam_l2 = args.l2_lambda
        self.lam_l1 = args.l1_lambda
        self.lam_nll = args.nll_lambda
        # recording the optimziation hsitory
        self.history =[]
    # prepare latent variable
    def _prepare_latent(self):
        self.z = Variable(self.z0, requires_grad=True)
        self.z_optim = torch.optim.Adam(
            [{'params': self.z}],
            weight_decay=0,
            eps=1e-8
        )
    # prepare mask
    def _prepare_mask(self):
        self.M = Variable(torch.clone(self.true_msk), requires_grad=True)
        self.M_optim = torch.optim.Adam(
            [{'params': self.M}],
            weight_decay=0,
            eps=1e-8
        )
    
    def rec_loss(self, input, target, args):
        if args.loss_type == 'L2':
            return F.mse_loss(input, target)
        elif args.loss_type == 'L1':
            return F.l1_loss(input, target)
        elif args.loss_type == 'Combine':
            vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
            self.ftr_net = subsequence(vgg.features, last_layer='20')
            self.PerceptLoss = PerceptLoss()                
            # calculate losses in the degradation space
            percpt_loss = self.PerceptLoss(self.ftr_net, input, target)
            mse_loss = F.mse_loss(input, target)
            # nll corresponds to a negative log-likelihood loss
            nll = self.z**2 / 2
            nll = nll.mean()
            l1_loss = F.l1_loss(input, target)
            return self.lams_percpt*percpt_loss + self.lam_l2*mse_loss + self.lam_l1*l1_loss + self.lam_nll*nll

    def res_show(self, epoch,loss):
           # imshow
        if epoch%100 == 0:
            if self.show:
                print('epch=',epoch, 'lss=', loss.detach().cpu().numpy())
                fig = plt.figure(figsize=(10, 40))
                # S1[abs(S1)>0.1]=1
                fig.add_subplot(1, 4, 1)
                grid = torchvision.utils.make_grid(self.input_img.clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title('raw')

                fig.add_subplot(1, 4, 2)
                grid = torchvision.utils.make_grid(self.Gnet(self.z).clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title('generated')
                
                fig.add_subplot(1, 4, 3)
                grid = torchvision.utils.make_grid((self.M).detach().clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title('Sparse')

                fig.add_subplot(1, 4, 4)
                grid = torchvision.utils.make_grid((self.input_img - self.Gnet(self.z)).abs().clamp(min=-1, max=1), scale_each=True, normalize=True)
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.title('Difference')
                plt.show() 
            return
   
    # GID optimzier ########
    def RGI_optim(self, input_img, opt_M, lam_mask, opt_G, args, true_msk = [], init_z0 = []):
        self.show = False
        self.lam_diss = args.lam_dis
        self.task = args.task
        # get pretrained Generator (Gnet)#
        self.Gnet, self.Dnet = getGANTrainer(self.gan_mdl_name)
        self.Gnet = self.Gnet.to(self.device)
        self.Dnet = self.Dnet.to(self.device)
        self.input_img = input_img.to(self.device, dtype=torch.float)  # input image
        if len(true_msk)>0:
            self.true_msk = true_msk.to(self.device, dtype=torch.float) # initial mask
        else:
            if self.task == 'Semantic_inpainting':
                self.true_msk = torch.ones_like(self.input_img).to(self.device, dtype=torch.float) 
            if self.task == 'Anomaly_segmentation':
                self.true_msk = torch.zeros_like(self.input_img).to(self.device, dtype=torch.float)
        if len(init_z0)>0:
            self.z0 = init_z0.clone().to(self.device) # input pool for z0s
        else:
            self.init_z0() # select the closest z0 from the pool, pool is randomly generated

        self.lsstype = args.loss_type
        self.lam = lam_mask
        self.iterations = args.iterations
        self.video = args.video
        
        self.start_fine_tune = args.start_fine_tune
        self.lr_z = args.lr_z
        self.lr_G = args.lr_G
        self.lr_M = args.lr_M
        self.fix_mask = args.fix_mask


        if opt_G:
            self.Gnet, self.Dnet = getGANTrainer(self.gan_mdl_name)
            self.Gnet = self.Gnet.to(self.device)
            # defroze Gnet
            for para in self.Gnet.parameters():
                para.requires_grad = True
            self.G_optim = torch.optim.Adam(
                [{'params': self.Gnet.parameters()}],
                weight_decay=0,
                eps=1e-8)

    
        self._prepare_latent()
        if opt_M:
            self._prepare_mask()
        else:
            self.M = torch.clone(self.true_msk)
        
        for epoch in range(self.iterations):
            if self.video:
                self.history.append([self.z.clone().detach().cpu().numpy(), self.M.clone().detach().cpu().numpy()])
            
            self.z_optim.zero_grad()
            self.adjust_learning_rate(self.z_optim, 'z', epoch)
            
            if opt_G:
                self.G_optim.zero_grad()
                self.adjust_learning_rate(self.G_optim, 'G', epoch)
            
            genrtd_img = self.Gnet(self.z)
            if opt_M:
                self.M_optim.zero_grad()
                self.adjust_learning_rate(self.M_optim, 'M', epoch)
                # Msk = torch.nn.Sigmoid()(self.M)
                if args.loss_type == 'L2':
                    rec_lss = torch.norm( (1-self.M)*(self.input_img-genrtd_img) )**2
                    l1_msk_pnty = torch.norm(self.M, 1)
                if args.loss_type == 'L1':
                    rec_lss = torch.norm( (1-self.M)*(self.input_img-genrtd_img), 1)
                    l1_msk_pnty = torch.norm(self.M, 1)
                if args.loss_type == 'VGG': 
                    rec_lss = self.rec_loss((1-self.M)*genrtd_img, (1-self.M)*self.input_img, args)
                    l1_msk_pnty = F.l1_loss(self.M, torch.zeros_like(self.input_img))
                l_dis = -self.Dnet(genrtd_img)
                if self.task == 'Semantic_inpainting':
                    loss =  rec_lss + self.lam*l1_msk_pnty + self.lam_diss*l_dis
                if self.task == 'Anomaly_segmentation':
                    loss =  rec_lss + self.lam*l1_msk_pnty + self.lam_diss*l_dis

                # print('epoch', epoch, 'rec_lss', rec_lss.item(), 'l1_msk_pnty', l1_msk_pnty.item())
            else:
                if args.loss_type == 'L2':
                    rec_lss = torch.norm( (1-self.true_msk)*(self.input_img-genrtd_img) )**2
                if args.loss_type == 'L1':
                    rec_lss = torch.norm( (1-self.true_msk)*(self.input_img-genrtd_img), 1)
                if args.loss_type == 'VGG': 
                    rec_lss = self.rec_loss((1-self.true_msk)*genrtd_img, (1-self.true_msk)*self.input_img, args)
                l_dis = -self.Dnet(genrtd_img)
                if self.task == 'Semantic_inpainting':
                    loss =  rec_lss + self.lam_diss*l_dis
                if self.task == 'Anomaly_segmentation':
                    loss =  rec_lss + self.lam_diss*l_dis
                

            
            loss.backward()

            self.z_optim.step()
            if opt_M:
                self.M_optim.step()
            if opt_G:
                self.G_optim.step()
            
            self.res_show(epoch, loss.item())
 
        S_msk = (self.M).detach().cpu().numpy().squeeze()
        L_generated = self.Gnet(self.z).detach().cpu().numpy().squeeze() # generated from latent spcae variable
        
        del self.z
        if opt_M:
            del self.M
        if opt_G:
            del self.Gnet
        torch.cuda.empty_cache()

        return [S_msk, L_generated]
        # return

    def adjust_learning_rate(self, optimizer, name, epoch):
        """decrease the learning rate"""
        if name == "z" :
            lr = self.lr_z

        if name ==  "M":
            lr = self.lr_M 
            if self.task == 'Semantic_inpainting':
                if epoch >= self.start_fine_tune:
                    if self.fix_mask:
                        lr = 0.0  
                    #     print('mask_fixed')
                    # else:
                    #     print('optimize_mask')
                    # self.lam_diss = 0
                              

        if name == "G":
            lr = 0.0
            if epoch >= self.start_fine_tune:
                lr = self.lr_G
                # self.lam_diss = 0

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return

    # generate 1000 latent z
    def generate_pool(self):
        pool_size=1000
        self.Pool = torch.rand([pool_size, self.nz],dtype=torch.float32).to(self.device) 
        with open('./checkpoints/Pool.pkl', 'wb') as f:
            pickle.dump(self.Pool, f)
        return

    def init_z0(self, echo=False): 
        # pool_size=1000
        ni = len(self.input_img)
        z0 = torch.tensor(np.zeros((ni,self.nz)),dtype=torch.float32).to(self.device)# initlize z0
        for l in range(ni):
            indx = 0
            lss = 1e10
            for i in range(len(self.Pool)):
                lss_ = torch.norm(torch.tensor(self.Gnet(self.Pool[i:i+1]).detach().cpu().numpy().squeeze()).to(self.device)-self.input_img[l])
                if lss_<=lss:
                    indx = i
                    lss = lss_
            z0[l] = self.Pool[indx]

        if echo == True:
            grid = torchvision.utils.make_grid(self.Gnet(z0).clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Closest one')
            plt.show()

            grid = torchvision.utils.make_grid(self.input_img.clamp(min=-1, max=1), scale_each=True, normalize=True)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Raw one')
            plt.show()
        self.z0 = z0
        return 

    def diceloss(self, y_pred,y_true, thres):
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

    def rgb2gry(self, img):
        return 0.2989*img[0,:,:] + 0.5870*img[1,:,:] + 0.1140*img[2,:,:]

    def max_diceloss(self, y_pred,y_true):
        # y_pred is pixelwise probability \in (0, 1)
        # y_true is the underline truth label of each pixel \in {0,1}
        dice_max = 0
        thres_opt = 0
        for i in range(100):
            thres = i/100 
            yp = y_pred.copy()    
            yp[ yp > thres ] = 1
            yp[ yp <= thres ] = 0
            y_sum = yp + y_true
            y_intct = yp * y_true
            dice = 2*np.sum(y_intct) / (np.sum(y_sum) + 1e-6)

            # print(dice)
            if dice > dice_max:
                dice_max = dice
                thres_opt = thres

        return dice_max, thres_opt
        
    def postProc(self, y_pred, y_lbl):
        y_pred_1d = np.sqrt(np.sum(y_pred**2,2))
        y_pred_1d_nmlz = (y_pred_1d - np.min(y_pred_1d))/(np.max(y_pred_1d)-np.min(y_pred_1d))
        y_true = (y_lbl/255).astype(np.uint8)
        return y_pred_1d_nmlz, y_true
    
    def nmlz(x):
        return torch.exp(x - x.max()) / torch.exp(x - x.max()).sum()
    
    def psnr(self, img1, img2):
        mse = np.linalg.norm(img1-img2)**2/(128*128*3)
        max_val = np.max(img1)**2
        return 10*np.log10(max_val/mse)
