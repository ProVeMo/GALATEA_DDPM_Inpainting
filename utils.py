import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data.sampled_seq_DL import SampledSeqDatasetTrain

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    size = args.image_size
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(954),
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomVerticalFlip(0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    conditional_classes = { "duodenum": ["normal", "ulcus", "ulcus_typ2"] }

    region_ratio = {
                    "none": 0,
                    "ausserhalb": 3000,
                    "oseophagus": 15000,
                    "z_linie": 6000,
                    "magen": 15000,
                    "pylorus": 6000,
                    "duodenum": 10000,
                    "inversion": 5000,
                    }

    dataset = SampledSeqDatasetTrain(args.dataset_path,
                                     conditional_classes=conditional_classes,
                                     trainSteps=5000,
                                     region_ratio_train=region_ratio,
                                     image_size=args.image_size,
                                     batch_size=args.batch_size,
                                     masked_value=1.0
                                     )
    #dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
