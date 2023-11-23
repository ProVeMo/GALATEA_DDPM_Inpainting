
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms
from munch import Munch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from frameSampling import FrameSampler

import torchvision.transforms.functional as TF

# natsort ist besser als sorted bei namen wie: file999; file1000 ...
import natsort
import yaml

from pycocotools.coco import COCO

from mask_tools import generate_random_mask

# Image size: [1245,954]
class ImageTransform:
    def __init__(self, size=256, enableflip=False, isGray=False):
        self.flip_prob = 0.5 if enableflip else 0.0
        self.isGray = isGray
        self.__set_transform(size)

    def set_size(self, size):
        self.__set_transform(size)

    def __set_transform(self, size):
        if self.isGray:
            mean = [0.5]
            std = [0.5]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.transform = transforms.Compose([
            transforms.CenterCrop(954),
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(self.flip_prob),
            transforms.RandomVerticalFlip(self.flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def transform(self):
        return self.transform


class SeqDataLoader:
    def __init__(self, data_dir, opt):
        self.transform = ImageTransform(opt.image_size)
        self.data_dir = data_dir

        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers
        self.train_steps = opt.num_iters
        self.conditional_classes = opt.conditional_classes
        self.region_ratio = opt.region_ratio
        self.image_size = opt.img_size
        self.masked_value = opt.mask_value
        self.setup()

    def setup(self):
        self.train_dataset = SampledSeqDatasetTrain(self.data_dir, self.conditional_classes,
                                                    self.train_steps, self.region_ratio, image_size=self.image_size, batch_size=self.batch_size, masked_value=self.masked_value)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

class SampledSeqDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data_dir, conditional_classes: dict = None, trainSteps=2500, region_ratio_train=None,
                image_size: int = 256, batch_size: int = 1, masked_value=1.0):
        super().__init__()
        self.pin_memory = True
        self.data_dir = data_dir
        self.transform = ImageTransform(size=image_size, enableflip=False)
        self.transform_gray = ImageTransform(size=image_size, enableflip=False, isGray=True)
        self.image_size = image_size

        self.frames_needed = [0]
        self.frame_interval = 0

        self.trainSteps = trainSteps
        self.region_ratio_trainB = region_ratio_train
        self.conditional_classes = conditional_classes

        self.cocoReaders = setup_COCO_readers(data_dir, self.conditional_classes)

        self.batch_size = batch_size

        train_path = os.path.join(self.data_dir, "trainB")
        # samples region and returns video and frame for that region ("get_targetframe")
        self.frameSampler = FrameSampler(folder_path=train_path,
                                           frame_interval=self.frame_interval,
                                           conditional_classes=self.conditional_classes, # only for conditional GAN
                                           region_ratio=self.region_ratio_trainB)

        self.maskValue = masked_value
        self.toPIL = torchvision.transforms.ToPILImage()
    def __len__(self):
        return self.trainSteps

    def __getitem__(self, idx):

        # samples region, then randomly selects video and frame from that region
        # returns dict{'video' : target_vid, 'target_frame': target_frame, 'condition': condition}
        frameset = self.frameSampler.get_targetframe(region='duodenum') # oseophagus

        Bi = Image.open(os.path.join(self.data_dir, "trainB", frameset['video'], f"frame{frameset['target_frame']}.jpg"))
        label = list(self.conditional_classes[frameset['region']]).index(frameset['condition'])

        B_real = self.transform.transform(Bi)
        # if frameset['condition'] == "normal":
        #     sample_masked, mask = generate_random_mask(B_real, mask_value=self.maskValue)
        # else:
        #     mask = getMaskOfFrame(self.cocoReaders, frameset['video'], frameset['condition'],
        #                                int(frameset['target_frame']), self.image_size)
        #     m = mask.expand(3,*mask.shape[1:])
        #     sample_masked = deepcopy(B_real)
        #     sample_masked[m == 1] = self.maskValue

            # import matplotlib.pyplot as plt
            # f, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(sample_masked.permute(1, 2, 0), 'gray')
            # ax2.imshow(Bi)
            # plt.show()
            # print("Stop!")


        return B_real, label


def setup_COCO_readers(data_dir, conditional_classes):

    with open(os.path.join(data_dir, "trainB", "classifications.yaml"), "r") as stream:
        try:
            s = yaml.safe_load(stream)
            frame_annotaions = s["annotations"]

            cocoReaders = {}
            for r in frame_annotaions: # region (duodenum, pylorus,etc)
                if r in conditional_classes.keys():  # conditional classes which shall be considered
                    for c in frame_annotaions[r]:
                        if c != "normal" and c in conditional_classes[r]:
                            for p in frame_annotaions[r][c]:
                                    if p not in cocoReaders.keys():
                                        cocoReaders[p] = {}
                                    cocoReaders[p][c] = COCO(os.path.join(data_dir, "segmentationMasks", p, 'result.json'))
                                    if p == "Ulcus_Duodeni_kvasir4":
                                        # necessary since label studio splitted dataset due to PC breakdown
                                        if f"{p}_2" not in cocoReaders.keys():
                                            cocoReaders[f"{p}_2"] = {}
                                        cocoReaders[f"{p}_2"][c] = COCO(os.path.join(data_dir, "segmentationMasks", f"{p}_2", 'result.json'))
        except yaml.YAMLError as exc:
            print(exc)

    return cocoReaders

def getMaskOfFrame(cocoReaders, patient, cls, frame, image_size, return_image=False, data_dir=''):

    transform_gray = ImageTransform(size=image_size, enableflip=False, isGray=True)
    toPIL = torchvision.transforms.ToPILImage()

    if patient == "Ulcus_Duodeni_kvasir4" and frame <= 181: # necessary since label studio splitted dataset due to PC breakdown
        coco = cocoReaders[f"{patient}_2"][cls]
    else:
        coco = cocoReaders[patient][cls]
    idx = 0
    # TODO: make look up more generic and efficient
    for entry in coco.imgs.items():
        d = entry[1]["file_name"]
        n = d.split("frame")[-1].split(".jpg")[0]
        if int(n) != frame:
            idx = idx + 1
        else:
            break

    try:
        img = coco.imgs[idx]
    except KeyError:
        print(f"KeyError: {patient}, {cls}, {frame}")
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    if return_image:
        t = ImageTransform(size=image_size, enableflip=False)
        B = Image.open(os.path.join(data_dir, "trainB", patient, f"frame{frame}.jpg"))
        B = t.transform(B)
        return transform_gray.transform(toPIL(255 * coco.annToMask(anns[0]))), B
    else:
        return transform_gray.transform(toPIL(255 * coco.annToMask(anns[0])))


def Get_mini_clip_for_val(dataDir : str, clip : str = '3', image_size: int = 256):
    path = os.path.join(dataDir, 'validate_mini_clip', clip)
    #path = os.path.join(dataDir, clip)
    img_list = [img for img in os.listdir(path)]
    img_list = natsort.natsorted(img_list)
    transform = ImageTransform(image_size, enableflip=False)
    A = []
    for image in img_list:
        Ai = Image.open(os.path.join(path, image))
        A.append(transform.transform(Ai))

    #A.reverse()
    return A