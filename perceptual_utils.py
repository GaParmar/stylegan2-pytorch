from collections import namedtuple

import torch
from torchvision import models

def vgg_norm(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (img - mean) / std

# ["conv1_1", "conv1_2", "conv3_2", "conv4_2"]
class PerceptualNetwork(torch.nn.Module):
    def __init__(self,):
        super(PerceptualNetwork, self).__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features
        self.mapping = {
            '0': "conv1_1",
            "1": "relu1_1",
            "2": "conv1_2",
            '3': "relu1_2",

            "5": "conv2_1",
            "6": "relu2_1",
            "7": "conv2_2",
            '8': "relu2_2",

            "10": "conv3_1",
            "11": "relu3_1",
            "12": "conv3_2",
            "13": "relu3_2",
            "14": "conv3_3",
            '15': "relu3_3",

            "17": "conv4_1",
            "18": "relu4_1",
            "19": "conv4_2",
            "20": "relu4_2",
            "21": "conv4_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.mapping.keys():
                output[self.mapping[name]] = x
        
        return output

