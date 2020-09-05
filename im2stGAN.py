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
    parser.add_argument("--image_path", type=str, default="morphing_test_B.png")
    parser.add_argument("--w_init", type=str, default="mean_face")
    parser.add_argument("--mean_n", type=int, default=5_000)
    parser.add_argument("--viz_frequency", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    # Specific to image morphing
    parser.add_argument("--path_w_a", type=str)
    parser.add_argument("--path_w_b", type=str)
    parser.add_argument("--blend_step", type=float, default=0.1)
    parser.add_argument("--output_file", type=str, default="viz/image_morphing.png")
    args = parser.parse_args()

    device=torch.device("cuda")

    if args.task == "projection":
        # 0. make relevant dirs if they dont exist
        if not os.path.exists("viz"):  os.makedirs("viz")
        if not os.path.exists("projected_wplus"):  os.makedirs("projected_wplus")

        # 1. Load a pretrained vgg16 for perceptual loss
        M_perceptual = PerceptualNetwork().to(device)
        
        # 2. prepare the input image
        transform = transforms.Compose( [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
         ] )
        img = transform(Image.open(args.image_path).convert("RGB"))
        img = img.view(1,3,args.size,args.size).to(device)
        img_renorm = vgg_norm(img*0.5+0.5) #renormalize original image
        feat_img = M_perceptual(img_renorm.to(device)) # vgg features
        name = os.path.basename(args.image_path).split(".")[0]

        # 3. Initialize the pretrained starganv2 model
        M = Generator(args.size, 512, 8)
        M.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        M.eval().to(device)

        # 4. Initialize the W+ vector
        if args.w_init=="mean_face":
            z = torch.randn(args.mean_n, 512, device=device)
            w_mean_face = M.style(z).mean(0)
            w_plus = w_mean_face.detach().clone().view(1, 1, -1).repeat(1, M.n_latent, 1)
        elif args.w_init=="random_uniform":
            w_plus = torch.rand(1, M.n_latent, 512)*2.0-1.0 # U[-1,+1]
        w_plus.requires_grad = True
        w_plus.to(device)

        # 5. Setup the optimizer
        optimizer = optim.Adam([w_plus], lr=args.lr, 
                        betas=(args.beta1, args.beta2), 
                        eps=args.adam_eps)
        
        # 6. Iterate for given iterations
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
            if i%args.viz_frequency==0:
                pbar.set_description(f"iter:{i:04d}  loss:{total_loss:.7f}")
                utils.save_image( curr_rec, f"viz/{name}_recon_{i}.png", 
                        normalize=True, range=(-1, 1))

        # 7. Save the projected vector to file
        outf = os.path.join("projected_wplus", f"{name}.pt")
        torch.save(w_plus.detach().cpu(), outf)

    elif args.task == "image_morphing":
        # make output dir if not exists
        if not os.path.exists("viz"):  os.makedirs("viz")
        # 1. Initialize the pretrained starganv2 model
        M = Generator(args.size, 512, 8)
        M.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        M.eval().to(device)
        # 2. Load the w+ vectors for A and B images
        A_w_plus = torch.load(args.path_w_a).to(device)
        B_w_plus = torch.load(args.path_w_b).to(device)
        lambda_blend = 1.0
        L_images = []
        while lambda_blend >= 0.0:
            curr_w_plus = A_w_plus*lambda_blend + B_w_plus*(1.0-lambda_blend) 
            with torch.no_grad():
                curr_img, _ = M([curr_w_plus], input_is_latent=True)
            L_images.append(curr_img.detach().cpu().view(3,args.size,args.size))
            lambda_blend -= args.blend_step
        # 3. Save the morphing to file
        utils.save_image( L_images, args.output_file, nrow=len(L_images),
                        normalize=True, range=(-1, 1))
    
    elif args.task == "style_transfer":
        pass
    
    elif args.task == "expression_transfer":
        pass

