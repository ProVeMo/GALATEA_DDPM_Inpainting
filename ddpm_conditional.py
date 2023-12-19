import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

from schedules import cosine_beta_schedule

from pycocotools.coco import COCO
from sampled_seq_DL import ImageTransform

from unet import UNetModel

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", schedule="linear"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        if self.schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule == "cosine":
            # TODO: Something is not correct here, must be checked before use!!!
            return cosine_beta_schedule(self.noise_steps)
        else:
            raise NotImplementedError(f"Schedule type {self.schedule} is not implemented.")

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, test_set, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            noise = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            masks = test_set["masks"]
            x = (1 - masks) * test_set["images"] + masks * noise

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)

                if cfg_scale > 0:
                    #labels = None
                    labels = torch.zeros((n,3), dtype=torch.float32, device=self.device)
                    uncond_predicted_noise = model(x, t, labels)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


    def sample_repaint(self, model, test_set, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            x_gt = test_set["images"].to(self.device)
            masks = test_set["masks"].to(self.device)

            x_gt = (x_gt + 1) / 2

            U = 2
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                for u in range(1,U):
                    t = (torch.ones(n) * i).long().to(self.device)

                    Ɛ = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device) if i > 1 else 0

                    predicted_noise = model(x_t, t, labels)

                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x_t, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x_t)
                    else:
                        noise = torch.zeros_like(x_t)

                    x_t_known = torch.sqrt(alpha_hat) * x_gt + (1 - alpha_hat) * Ɛ
                    x_t_unknown = 1 / torch.sqrt(alpha) * (x_t - (beta / torch.sqrt(1-alpha_hat) ) * predicted_noise) + torch.sqrt(beta) * noise

                    x_t = masks * x_t_unknown + (1-masks) * x_t_known

                    # if u < U and i > 1:
                    #     beta_t_before = self.beta[t-1][:, None, None, None]
                    #     predicted_noise = model(torch.sqrt(1 - beta_t_before) * x_t, t, labels)
                    #
                    #     if cfg_scale > 0:
                    #         uncond_predicted_noise = model(x_t, t, None)
                    #         predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

        model.train()
        #x_t = (x_t * 0.5) + 0.5
        #x = (x_t.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8)
        x = x_t
        return x

def get_test_set(args):
    test_path = "/home/mwallrodt/Desktop/mnielsen/Desktop/Workspace/tempGAN/dataset/tempDataset_v2/real_duodenum_masked"
    mask_value = 1.0
    test_coco_reader = COCO(os.path.join(test_path, 'result.json'))
    test_set = {"images": [], "images_masked": [], "masks": []}
    transform = ImageTransform(size=args.image_size, enableflip=False)
    transform_gray = ImageTransform(size=args.image_size, enableflip=False, isGray=True)
    toPIL = torchvision.transforms.ToPILImage()
    for entry in test_coco_reader.imgs.items():
        if args.cdim == 1:
            img = Image.open(os.path.join(test_path, entry[1]['file_name']))
        else:
            img = Image.open(os.path.join(test_path, entry[1]['file_name'])).convert("L")
        img = transform.transform(img)
        test_set["images"].append(img)
        cat_ids = test_coco_reader.getCatIds()
        anns_ids = test_coco_reader.getAnnIds(imgIds=entry[1]['id'], catIds=cat_ids, iscrowd=None)
        anns = test_coco_reader.loadAnns(anns_ids)
        mask = transform_gray.transform(toPIL(255 * test_coco_reader.annToMask(anns[0])))
        mask = (mask + 1) / 2
        sample_masked = (1 - mask) * img
        test_set["images_masked"].append(torch.unsqueeze(sample_masked, dim=0))
        test_set["masks"].append(mask)

    test_images = torch.stack(test_set["images"])
    test_masks = torch.stack(test_set["masks"])
    test_set["images"] = test_images
    test_set["masks"] = test_masks

    return test_set

def inference(args):
    setup_logging(args.run_name)
    device = args.device
    schedule = args.schedule
    model = UNet_conditional(c_in=args.cdim, c_out=args.cdim, num_classes=args.num_classes).to(device)  # +args.num_classes
    diffusion = Diffusion(img_size=args.image_size, device=device, schedule=schedule)

    state_dict = torch.load("./models/DDPM_conditional/ema_ckpt.pt")
    model.load_state_dict(state_dict)

    test_set = get_test_set(args)
    masked_images = (1 - test_set["masks"]) * test_set["images"] - test_set["masks"]
    masked_images = (masked_images.clamp(-1, 1) + 1) / 2
    #masked_images = (masked_images * 255).type(torch.uint8)

    gt = (test_set["images"].clamp(-1, 1) + 1) / 2
    #gt = (gt * 255).type(torch.uint8)

    cfg_scale = 10
    #labels = torch.arange(args.num_classes).long().to(device)
    labels = torch.zeros(args.num_classes).long().to(device) + 2

    sampled_images = diffusion.sample_repaint(model, test_set, n=len(labels), labels=labels, cfg_scale=cfg_scale)
    images = torch.cat(( gt.to(device), masked_images.to(device), sampled_images), dim=2)
    plot_images(images)

def train(args):
    setup_logging(args.run_name)
    device = args.device
    schedule = args.schedule
    dataloader = get_data(args)
    #model = UNet_conditional(c_in=args.cdim, c_out=args.cdim, num_classes=args.num_classes).to(device) # +args.num_classes
    model = UNetModel(  image_size = args.image_size,
                        in_channels = args.cdim + args.num_classes,
                        model_channels = args.cdim,
                        out_channels = args.cdim,
                        num_res_blocks = 3,
                        attention_resolutions = (1,2,4),
                        num_classes=args.num_classes,
                        use_scale_shift_norm=False,
                      ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, schedule=schedule)
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    test_set = get_test_set(args)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels, masks) in enumerate(pbar):

            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            #images = (images + 1) / 2

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # x_t = (1 - masks) * images + masks * noise
            # noise = masks * noise

            #x_t = (x_t * 2) - 1
            #plot_images(images)
            #plot_images(x_t)
            if np.random.random() < 0.1:
                #labels = None
                labels = torch.zeros((args.batch_size,args.num_classes), dtype=torch.float32).to(device)

            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            #labels = torch.arange(args.num_classes).long().to(device)
            labels = torch.zeros((3,args.num_classes), dtype=torch.float32, device=device)
            labels[0][0] = 1
            labels[1][1] = 1
            labels[2][2] = 1
            for cfg_scale in [3]:
                sampled_images = diffusion.sample(model, test_set, n=len(labels), labels=labels, cfg_scale=cfg_scale)
                ema_sampled_images = diffusion.sample(ema_model, test_set, n=len(labels), labels=labels, cfg_scale=cfg_scale)
                #plot_images(sampled_images)
                save_images(sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch}_cfg_{cfg_scale}.jpg"))
                save_images(ema_sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch}_cfg_{cfg_scale}_ema.jpg"))
            # save model state dicts
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.cdim = 1
    args.epochs = 500
    args.batch_size = 1
    args.image_size = 128 # org. 64
    args.num_classes = 3 # CIFAR-10 = 10
    #args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
    args.dataset_path = r"/home/mwallrodt/Desktop/mnielsen/Desktop/Workspace/tempGAN/dataset/tempDataset_v2"
    args.device = "cuda"
    args.lr = 1e-4 # org. 3e-4
    args.schedule = "linear"


    train(args)
    #inference(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

