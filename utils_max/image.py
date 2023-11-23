import torchvision
from utils.TensorImageConverter import batch_to_image

def denormalize(x):
    # range (-1, 1) to range (0, 1)
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, col_num, filename):
    x = denormalize(x)
    #x = batch_to_image(x,True)[0]
    torchvision.utils.save_image(x.cpu(), filename, nrow=col_num, padding=0)

