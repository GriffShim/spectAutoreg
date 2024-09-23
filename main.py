import os
import time
import torch
import torch.distributed as dist
from model import penis_net, adaptive_penis_net
from train import train

def setup():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup():
    dist.destroy_process_group()

def main():
    model = adaptive_penis_net(img_size=(31, 31), num_classes=11, hidden_dim=1024, slice_points=[1, 3, 8, 16]).cuda()
    NUM_EPOCHS = 1000
    rank, world_size = setup()
    train(model, NUM_EPOCHS, "realSpiel", rank, world_size)
    cleanup()

if __name__ == "__main__":
    main()
