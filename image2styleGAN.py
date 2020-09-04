import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=0)
    parser.add_argument("--w_plus", action="store_true")
    parser.add_argument("files", metavar="FILES", nargs="+")
    args = parser.parse_args()

    # 1. Load the pretrained model into memory
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    # 2. efficient embedding into W+

    # 3. basic operations on vectors to study the quality of the embedding 
    # 3a morphing 
    # 3b style transfer
    # 3c expression transfer
    