import torch
import lpips
# from IPython import embed
import os
# import cv2
from PIL import Image
import numpy as np

use_gpu = True         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

if(use_gpu):
    loss_fn.cuda()

## Example usage with dummy tensors
file_path = r'/data/yuhe.ding/CODE/MODIFY/120'
src_path = r'/data/yuhe.ding/DATA/MetFaces/images'
im0_path_list = []
im1_path_list = []
for root, _, fnames in sorted(os.walk(file_path, followlinks=True)):
    for fname in fnames:
        path = os.path.join(root, fname)
        # if '_generated' in fname:
        # print(path)
        im0_path_list.append(path)
        # elif '_real' in fname:
            # im1_path_list.append(path)
for root, _, fnames in sorted(os.walk(src_path, followlinks=True)):
    for fname in fnames:
        path = os.path.join(root, fname)
        # if '_generated' in fname:
        # im0_path_list.append(path)
        # elif '_real' in fname:
        im1_path_list.append(path)

dist_ = []
for i in range(len(im0_path_list)):
    im0 = lpips.load_image(im0_path_list[i])
    im1 = lpips.load_image(im1_path_list[i])
    # im0 = Image.fromarray(im0)
    im1 = Image.fromarray(im1).resize((256,256))
    im1 = np.array(im1)
    # print(im0.shape,im1.shape)

    # dummy_im0 = lpips.im2tensor(lpips.load_image(im0_path_list[i]))
    # dummy_im1 = lpips.im2tensor(lpips.load_image(im1_path_list[i]))
    
    dummy_im0 = lpips.im2tensor(im0)
    dummy_im1 = lpips.im2tensor(im1)
    if(use_gpu):
        dummy_im0 = dummy_im0.cuda()
        dummy_im1 = dummy_im1.cuda()
    dist = loss_fn.forward(dummy_im0, dummy_im1)
    dist_.append(dist.mean().item())
print('Avarage Distances: %.3f' % (sum(dist_)/len(im0_path_list)))