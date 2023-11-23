import torch
import torchvision.transforms as transforms


# Image of shape (B, C, W, H) or (C, W, H)

def image_to_tensor(img, trans=()):
    img_tensor = transforms.ToTensor()(img)
    return img_tensor


def tensor_to_image(tensor, unnormalize=False):
    if unnormalize:
        tensor = tensor * 0.5 + 0.5
    img = transforms.ToPILImage(mode='RGB')(tensor)
    return img


def batch_to_image(tensor, unnormalize=False):
    img = []
    # Kein Batch
    if tensor.ndim == 3:
        for i in range(int((tensor.size(0) / 3))):
            split_tensor = torch.split(tensor, 3, 0)[i]
            img.append(tensor_to_image(split_tensor, unnormalize=unnormalize))
    # Batch
    if tensor.ndim == 4:
        split = torch.split(tensor, 1, 0)
        for i in range(len(split)):
            temp_tens = torch.squeeze(split[i])
            for j in range(int((temp_tens.size(0) / 3))):
                split_tensor = torch.split(temp_tens, 3, 0)[j]
                img.append(tensor_to_image(split_tensor, unnormalize=unnormalize))
    return img


def tensorlist_to_cat(tens_list=[]):
    tens_cat = 0
    for i in range(tens_list.count()):
        tens_cat = tens_list[i] if i == 0 else torch.cat([tens_cat, tens_list[i]], dim=0)
    return tens_cat


def tensorcat_to_list(tens_cat, n_splits):
    return torch.split(tens_cat, n_splits, 0)
