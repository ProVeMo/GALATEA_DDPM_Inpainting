
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

if __name__ == '__main__':
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()