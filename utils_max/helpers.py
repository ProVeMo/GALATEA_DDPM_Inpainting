
import matplotlib.pyplot as plt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def plot_tensor(tensor):

    if tensor.is_cuda:
        t = tensor.cpu()
    else:
        t = tensor

    if len(tensor.shape) == 4:
        plt.imshow(t[0].permute(1,2,0))
        plt.show()
    else:
        plt.imshow(t.permute(1,2,0))
        plt.show()