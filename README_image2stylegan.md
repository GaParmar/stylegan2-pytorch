## Image2Style GAN in PyTorch

Implementation of "Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?" (https://arxiv.org/pdf/1904.03189.pdf) in PyTorch 
StyleGAN2 base code borrowed from (https://github.com/rosinality/stylegan2-pytorch)

## Environment Setup
 - Create and activate the conda environment with the following command: 
   ``
   ``

 - Download the pretrained StyleGAN2 weights from here:
  [Link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)

## Projecting an Image to extended W+
 - `python image2styleGAN.py --ckpt=<pretrained_stargan_model>  --task="projection" --image_path=<path_to_image>`
 - Saves the optimal projected W+ tensor at "projected_wplus/{image_name}.pth"
  Other Options: 
  - "--w_init" : "mean_face" or "random_uniform"

  
## Some Visual Results Obtained
 - 

## Credits
 - https://github.com/rosinality/stylegan2-pytorch
 - 