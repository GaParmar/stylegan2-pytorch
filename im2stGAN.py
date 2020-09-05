import argparse
import math
import os
import pdb
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision import utils
import lpips
from model import Generator

from perceptual_utils import PerceptualNetwork, vgg_norm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="550000.pt")
    parser.add_argument("--task", type=str, default="projection")
    parser.add_argument("--image_path", type=str, default="test_img.png")
    parser.add_argument("--w_init", type=str, default="mean_face")
    parser.add_argument("--mean_n", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    device=torch.device("cuda")


    # 0 (a). Load a pretrained VGG-16 network for perceptual loss
    M_perceptual = PerceptualNetwork().to(device)


    # 0. Load the Image to project back
    transform = transforms.Compose( [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = transform(Image.open(args.image_path).convert("RGB"))
    img = img.view(1,3,args.size,args.size).to(device)


    # 1. Load the pretrained model into memory
    M = Generator(args.size, 512, 8)
    M.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    M.eval().to(device)

    # 2. Initialize the W+ vector 
    if args.w_init=="mean_face":
        z = torch.randn(args.mean_n, 512, device=device)
        w_mean_face = M.style(z).mean(0)
        w_plus = w_mean_face.detach().clone().view(1, 1, -1).repeat(1, M.n_latent, 1)
    elif args.w_init=="random_uniform":
        w_plus = torch.randn(1, M.n_latent, 512)
    w_plus.requires_grad = True
    w_plus.to(device)

    # 3. Define the Optimizer
    optimizer = optim.Adam([w_plus], lr=args.lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    # 4. Iterate for num_iters
    #renormalize original image
    img_renorm = vgg_norm(img*0.5+0.5)
    feat_img = M_perceptual(img_renorm.to(device))
    pbar = tqdm(range(args.num_iters))
    for i in pbar:
        curr_rec, _ = M([w_plus], input_is_latent=True)
        # pixelwise mse loss
        total_loss = F.mse_loss(curr_rec, img) * args.lambda_mse

        # Compute Perceptual Distance
        # denorm by 0.5,0.5 and renorm using the vgg16 params
        curr_rec_renorm = vgg_norm(curr_rec*0.5 + 0.5)
        feat_rec = M_perceptual(curr_rec_renorm)
        for layer in ["conv1_1", "conv1_2", "conv3_2", "conv4_2"]:
            layer_diff = F.mse_loss(feat_rec[layer], feat_img[layer])
            total_loss += (layer_diff/ feat_rec[layer].numel())

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        pbar.set_description(f"iter:{i:04d}\tloss:{total_loss:.7f}")
        if i%100==0:
            utils.save_image( curr_rec, f"viz/recon_{i}.png", 
                    normalize=True, range=(-1, 1))
    
    # save image 
    utils.save_image( curr_rec, f'recon.png', nrow=1, 
                    normalize=True, range=(-1, 1))

