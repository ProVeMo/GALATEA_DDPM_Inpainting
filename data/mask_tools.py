from copy import deepcopy

import torch
import random
import os
import cv2
import numpy as np
import yaml

import skimage.exposure
from numpy.random import default_rng

import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def generate_random_mask(img, minRadius=0.05, maxRadius=0.3, color=(255,255,255), mask_value=1.0):

    height, width = img.shape[-2:]
    img_size = min(height,width)
    r = random.randint(int(minRadius*img_size), int(maxRadius*img_size))

    cx = random.randint(r, img_size-r)
    cy = random.randint(r, img_size-r)

    mask = np.zeros((height,width))
    cv2.circle(mask, (cx,cy), r, color, thickness=-1)

    # define random seed to change the pattern
    seedval = 75
    rng = default_rng(seed=seedval)
    #
    # # create random noise image
    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    noise = np.where(mask == 255, noise, mask)
    #
    # # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    #
    # # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #mask = cv2.merge([mask, mask, mask])
    mask = mask.astype("float64")

    mask_3d = convert_gray2rgb(mask)[None]

    masked_img = deepcopy(img)
    m = mask_3d[0] == 255
    for i in range(3):
        masked_img[i][mask == 255] = mask_value

    # add mask to input
    #result1 = cv2.add(img, mask)
    # img = tensor2numpy(masked_img, img_size=img_size)
    # plt.imshow(img)
    # plt.show()

    mask = torch.from_numpy(mask/ 255)
    return masked_img, torch.unsqueeze(mask,dim=0)

def convert_gray2rgb(image):
    width, height = image.shape
    out = np.empty((3,width, height), dtype=np.uint8)
    out[0, :, :] = image
    out[1, :, :] = image
    out[2, :, :] = image

    return out

def tensor2numpy(tensor,img_size=128):
    result = torch.clone(tensor)
    result = torch.reshape(result, (3, img_size, img_size))
    result = result.detach().cpu().numpy()
    result = np.swapaxes(result, 0, -1)
    result = np.swapaxes(result, 0, 1)

    return result

def numpy2tensor(array):
    result = np.swapaxes(array, 0, -1)
    result = np.swapaxes(result, 1, -1)
    result = torch.from_numpy(result)

    return torch.unsqueeze(result,dim=0)

def load_predefined_mask_noshuffle(args, batch_size, chose_idx):
    if args.mask_type != 'RAND':
        return

    img_height = img_width = args.img_size
    masks = []

    mask_files = os.listdir(args.rand_mask_path)
    mask_files = sorted(mask_files)
    MASK_NUM = len(mask_files)
    
    chosen = []
    for j in range(batch_size):
        chosen.append(chose_idx)
        chose_idx = chose_idx + 1

        if chose_idx >= MASK_NUM:
            chose_idx = 0
        
    for i in range(batch_size):
        BINARY_MASK = cv2.imread(args.rand_mask_path + '/' + mask_files[chosen[i]], cv2.IMREAD_GRAYSCALE)
        w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]
        
        if w <img_width or w > img_width or h > img_height or h < img_height:
            BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        # get mask
        thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
        masks.append(BINARY_MASK)
 
    return masks, chose_idx


    

def load_predefined_mask(args, batch_size):
    img_height = img_width = args.img_size
    
    masks = []

    # if FIX
    if args.mask_type == 'FIX':
        mask_file = args.fix_mask_path
        
        BINARY_MASK = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]

        if w <img_width or w > img_width or h > img_height or h < img_height:
            BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

        # get mask
        thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
        for i in range(batch_size):
            masks.append(BINARY_MASK)

        

    # if rand
    elif args.mask_type == 'RAND':
        mask_files = os.listdir(args.rand_mask_path)
        mask_files = sorted(mask_files)
        MASK_NUM = len(mask_files)

        chosen = random.sample(range(0, MASK_NUM), batch_size)

        for i in range(batch_size):
            BINARY_MASK = cv2.imread(args.rand_mask_path + '/' + mask_files[chosen[i]], cv2.IMREAD_GRAYSCALE)
            w, h = BINARY_MASK.shape[0], BINARY_MASK.shape[1]
            
            if w <img_width or w > img_width or h > img_height or h < img_height:
                BINARY_MASK = cv2.resize(BINARY_MASK, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

            # get mask
            thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)
            masks.append(BINARY_MASK)
 
    return masks



# call when using predefined mask
def mask_image_predefined(args, x, masks):

    img_height = img_width = args.img_size

    # ------- masks : list of cv2 ndarray
    batch_size = len(masks)

    mask = torch.zeros((batch_size, 1, img_height, img_width), dtype=torch.float32)

    for i in range(batch_size):
        mask[i, :, :, :] = torch.Tensor(masks[i]) # / 255.0

    mask[mask!=1.0] = 0

    if x.is_cuda:
        mask = mask.cuda()
        
    result = x * (1. - mask) #first, occlude with big mask
    cropped_area = x * mask

    if x.is_cuda:
        cropped_area = cropped_area.cuda()

    return result, mask, cropped_area


if __name__ == '__main__':

    img = np.zeros((256,256))
    generate_random_mask(img)
