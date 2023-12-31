# import test data and lables
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
import torchvision
import torch

def read_test(args, test_img_dir):
    m = args.img_size # 128
    n = args.img_size
    # test_img_dir = product_test_data_dir + defect_type +'/' # get test image deirectory 
    f_name_test =[]
    cv_img = []
    cv_img_raw =[]

    for img in sorted(glob.glob(test_img_dir+'*')):
        f_name_test.append(img)
        im= Image.open(img).convert('RGB')
        imm = im.resize((m,n))
        cv_img_raw.append(imm)
        cv_img.append(np.asarray((imm)))
    x_test = np.array(cv_img)
    if x_test.max()>10:
        x_test = x_test.astype('float32') / 255.
    else:
        x_test = x_test.astype('float32')
    if len(x_test.shape)>3:
        x_test = x_test[:,:,:,:3]
    return x_test

def read_test_lbl(args, gt_img_dir):
    # gt_img_dir = product_gt_data_dir + defect_type + '/' # get test image deirectory 
    m = args.img_size # 128
    n = args.img_size
    f_name_lbl =[]
    cv_img = []
    cv_img_raw =[]
    for img in sorted(glob.glob(gt_img_dir+'*')):#sorted(glob.glob(gt_img_dir+'*')):
        f_name_lbl.append(img)
        im = Image.open(img)
        imm = im.resize((m,n))
        cv_img_raw.append(imm)
        cv_img.append(np.asarray((imm)))
    x_test_lbl = np.array(cv_img)
    if len(x_test_lbl.shape)>3:
        x_test_lbl = x_test_lbl[:,:,:,0]
    return x_test_lbl


def transofrm_test_image(x_test):
# transofrm test image
    ni= len(x_test)
    # define image transformation (from [0 255] -> [-1,1])
    class NumpyToTensor(object):

        def __init__(self):
            return

        def __call__(self, img):
            r"""
            Turn a numpy objevt into a tensor.
            """

            if len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)

            return Transforms.functional.to_tensor(img)
    import torchvision.transforms as Transforms
    transformList = [  NumpyToTensor(),
                        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = Transforms.Compose(transformList)

    x_raw = np.zeros([ni,3,128,128])
    for i in range(ni):
        x_raw[i,:,:,:] = transform(x_test[i])

    # grid = torchvision.utils.make_grid(torch.tensor(x_raw).clamp(min=-1, max=1), scale_each=True, normalize=True)
    # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    x_raw = torch.tensor(x_raw)
    return x_raw