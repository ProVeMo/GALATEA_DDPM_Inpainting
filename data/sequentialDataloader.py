import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils.FrameConv import *

# natsort ist besser als sorted bei namen wie: file999; file1000 ...
import natsort


# Image size: [1245,954]
class ImageTransform():
    def __init__(self, size=256):
        self.__set_transform(size)

    def set_size(self, size):
        self.__set_transform(size)

    def __set_transform(self, size):
        self.transform = transforms.Compose([
            transforms.CenterCrop(954),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def transform(self):
        return self.transform


class SeqDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform, n_frames=3):
        super().__init__()
        self.transform = transform
        self.n_frames = n_frames
        self.A_real_filenames = natsort.natsorted(os.listdir(data_dir + '/trainA'))
        self.A_real_filenames = [data_dir + '/trainA/' + file_name for file_name in self.A_real_filenames]
        self.B_real_filenames = natsort.natsorted(os.listdir(data_dir + '/trainB'))
        self.B_real_filenames = [data_dir + '/trainB/' + file_name for file_name in self.B_real_filenames]

    def __len__(self):
        return (min(len(self.A_real_filenames), len(self.B_real_filenames)) - self.n_frames)

    def __getitem__(self, idx):
        A_real = []
        B_real = []

        for i in range(self.n_frames):
            Ai = Image.open(self.A_real_filenames[(idx + i)])
            A_real.append(self.transform.transform(Ai))

            Bi = Image.open(self.B_real_filenames[(idx + i)])
            B_real.append(self.transform.transform(Bi))

        A_real.reverse()
        B_real.reverse()
        return A_real, B_real



