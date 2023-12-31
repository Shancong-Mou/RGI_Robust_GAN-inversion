import torch
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
import random

from FyeldGenerator import generate_field
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def generate_missing(x_raw, args):
    missing_size = args.missing_size
    mask = torch.zeros_like(x_raw)
    x = x_raw.clone().detach()
    if args.missing_type == 'central_block':
        mask[:,:,60-missing_size:60+missing_size, 60-missing_size:60+missing_size] = 1  
        my_array = np.ones((128,128))
        indices = np.random.choice(my_array.shape[1]*my_array.shape[0], replace=False, size=int(4*missing_size**2))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Mss = torch.randn(2*missing_size, 2*missing_size)
                x[i,j,:,:][60-missing_size:60+missing_size, 60-missing_size:60+missing_size]= -1 + Mss

    if args.missing_type == 'random':
        my_array = np.ones((args.img_size,args.img_size))
        indices = np.random.choice(my_array.shape[1]*my_array.shape[0], replace=False, size=int(4*missing_size**2))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Mss = -1+ torch.randn(2*missing_size*2*missing_size)
                img = torch.reshape(x[i,j,:,:].clone(), (-1,)).float()
                img[indices] = Mss
                x[i,j,:,:]= torch.reshape(img,(my_array.shape[1], my_array.shape[0])).clone()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # add msk
                img = torch.reshape(mask[i,j,:,:].clone(), (-1,)).float()
                img[indices] = 1
                mask[i,j,:,:]= torch.reshape(img,(my_array.shape[1], my_array.shape[0])).clone() 


    if args.missing_type == 'irregular_mask':
        mask_path = './Data/mask/testing_mask_dataset/'
        two_d_mask =  []
        if args.missing_size<=20:
            for img in sorted(glob.glob(mask_path+'*'))[:1000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)
        else:
            for img in  sorted(glob.glob(mask_path+'*'))[4000:5000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)            

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Mss = -1+ torch.randn((args.img_size,args.img_size))
                x[i,j,:,:]= x_raw[i,j,:,:].clone().detach()*( 1 - two_d_mask[i]) +  Mss * two_d_mask[i]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # add mask
                mask[i,j,:,:]= torch.tensor(two_d_mask[i]).clone() 

# add a white mask experiment
    if args.missing_type == 'irregular_mask_white':
        mask_path = './Data/mask/testing_mask_dataset/'
        two_d_mask =  []
        if args.missing_size<=20:
            for img in sorted(glob.glob(mask_path+'*'))[:1000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)
        else:
            for img in  sorted(glob.glob(mask_path+'*'))[4000:5000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)            

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Mss = 1+ torch.randn((args.img_size,args.img_size))
                x[i,j,:,:]= x_raw[i,j,:,:].clone().detach()*( 1 - two_d_mask[i]) +  Mss * two_d_mask[i]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # add mask
                mask[i,j,:,:]= torch.tensor(two_d_mask[i]).clone() 

# add a white mask experiment
    if args.missing_type == 'irregular_mask_small_deviation':
        mask_path = './Data/mask/testing_mask_dataset/'
        two_d_mask =  []
        if args.missing_size<=20:
            for img in sorted(glob.glob(mask_path+'*'))[:1000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)
        else:
            for img in  sorted(glob.glob(mask_path+'*'))[4000:5000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)            

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                Mss =0.5 + torch.randn((args.img_size,args.img_size))
                x[i,j,:,:]= x_raw[i,j,:,:].clone().detach() +  Mss * two_d_mask[i]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # add mask
                mask[i,j,:,:]= torch.tensor(two_d_mask[i]).clone() 

# add a guassian markov random field mask experiment
    if args.missing_type == 'irregular_mask_guassian_markov_random_field':
        mask_path = './Data/mask/testing_mask_dataset/'
        two_d_mask =  []
        if args.missing_size<=20:
            for img in sorted(glob.glob(mask_path+'*'))[:1000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)
        else:
            for img in  sorted(glob.glob(mask_path+'*'))[4000:5000]:#random.sample(sorted(glob.glob(mask_path+'*')),len(x_raw)):
                my_array = np.ones((args.img_size,args.img_size))
                im = Image.open(img)
                imm = np.asarray (im.resize((x.shape[2], x.shape[3])))
                my_array [imm >1e-5] = 1.0
                my_array [imm<=1e-5] = 0.0 # convert to binary
                two_d_mask.append(my_array)            

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Mss =0.5 + 0.1*torch.randn((args.img_size,args.img_size))**2
                shape = (128, 128)
                field = generate_field(distrib, Pkgen(5), shape)
                # Mss = (field-field.min())/(field.max()-field.min())
                Mss = 2*(field-field.min())/(field.max()-field.min())
                Mss[Mss<0.5] = 0.5
                x[i,j,:,:]= x_raw[i,j,:,:].clone().detach() +  Mss * two_d_mask[i] + torch.randn((args.img_size,args.img_size)) * two_d_mask[i]
            
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # add mask
                mask[i,j,:,:]= torch.tensor(two_d_mask[i]).clone() 
    return x, mask

# Helper that generates power-law power spectrum
def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)
    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


