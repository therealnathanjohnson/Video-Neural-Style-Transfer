import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models
from torchvision.io import decode_image
from torch import optim

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import time

import argparse

class NST:
    def __init__(
        self,
        base_img_path,
        style_img_path
    ):
        weights = models.VGG19_Weights.DEFAULT
        self.vgg19 = models.vgg19(weights=weights).features
        #get necessary preprocessing function for vgg19
        self.preprocess = weights.transforms()

        #freeze weights
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        #decode and preprocess content image
        self.base_img = decode_image(base_img_path)
        self.p_base_img = self.preprocess(self.base_img).unsqueeze(0)
        
        #decode and preprocess style image
        self.style_img = decode_image(style_img_path)
        self.p_style_img = self.preprocess(self.style_img).unsqueeze(0)
        
        #combination image that we optimize
        self.comb_img_params = nn.ParameterDict({
            "comb_img": nn.Parameter(self.p_base_img)
        })
        
        self.layer_inds = {
            'block1_relu1': 1,
            'block2_relu1': 6,
            'block3_relu1': 11,
            'block4_relu1': 20,
            'block5_relu1': 29,
            'block5_relu2': 31 #content loss
        }

        #index to layer; opposite of layer_inds
        self.ind_layers = {v:k for k,v in self.layer_inds.items()}

        self.style_loss_layers = ['block1_relu1', 'block2_relu1', 'block3_relu1', 'block4_relu1', 'block5_relu1']
        self.content_loss_layer = 'block5_relu2'

        #weight of each loss
        self.style_weight = 1
        self.content_weight = 2.5e-8
        self.total_variation_weight = 1e-6
        
        self.optimizer = optim.LBFGS(
            self.comb_img_params.parameters(), 
            lr=1, 
            max_iter=1,#20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn=None
        )
        
        #get features of content and style images
        self.base_img_features = self.feature_extractor(self.p_base_img)
        self.style_img_features = self.feature_extractor(self.p_style_img)
        
        #get the gram matrices of the style image features
        self.style_gram_matrices = {k:self.gram_matrix(v) for k,v in self.style_img_features.items()}

    #reverse self.preprocess
    def deprocess_image(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)

        if img.dim() == 3:
            img = img.unsqueeze(0)  #(3,H,W) -> (1,3,H,W)

        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        return img.squeeze(0)
    
    def feature_extractor(self, x):
        out = {}
        #loop through layers and run inference
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            #store prespecified intermediary activations
            if i in self.ind_layers:
                out[self.ind_layers[i]] = x
            #if there are no more activations to store, break
            if len(out) == len(self.ind_layers):
                break

        return out
    
    def gram_matrix(self, x):
        assert len(x.shape) == 4 #[batch, channels, height, width]
        b, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) #shape: [batch, channels, height*width]
        return torch.matmul(x,x.permute(0,2,1))
    
    def content_loss(self, combination_img):
        return torch.sum(torch.square(combination_img-self.base_img_features[self.content_loss_layer]))
    
    #compare the gram matrix of a style image feature with the gram matrix of a combination image feature
    def style_loss(self, style_gram, combination_img):   
        batch, channels, height, width = combination_img.shape
        M = height * width

        comb_gram = self.gram_matrix(combination_img)

        return torch.sum(torch.square(comb_gram - style_gram))/(4*(channels**2)*(M**2))
    
    def total_variation_loss(self, x):
        #vertical differences: each pixel minus the pixel below, squared
        a = (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2
        #horizontal differences: each pixel minus the pixel to the right, squared
        b = (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2
        return torch.sum((a + b) ** 1.25)
    
    def compute_loss(self, comb_img):
        #extract features
        features = self.feature_extractor(comb_img)

        loss = torch.tensor(0.0)

        #compute content loss
        comb_img_features = features[self.content_loss_layer]
        content_loss_value = self.content_weight * self.content_loss(comb_img_features)
        loss += content_loss_value
        
        #compute style losses for all style feature layers
        for layer in self.style_loss_layers:
            style_gram = self.style_gram_matrices[layer]
            comb_img_features = features[layer]

            style_loss_value = self.style_loss(style_gram, comb_img_features)
            loss += (self.style_weight / len(self.style_loss_layers)) * style_loss_value

        loss += self.total_variation_weight * self.total_variation_loss(comb_img)

        return loss
    
    def train(self, steps=600, verbose=False):
        for i in range(steps):
            def closure():
                self.optimizer.zero_grad()
                loss = self.compute_loss(self.comb_img_params['comb_img'])
                loss.backward()
                
                #clamp gradients
                #self.comb_img_params['comb_img'].grad.data.clamp_(-1, 1)

                return loss
            self.optimizer.step(closure)
            
            #display image every 50 steps
            if verbose and i % 50 == 0:
                img = self.comb_img_params['comb_img'].detach().clone().reshape(3,224,224) #C,H,W -> H,W,C
                img = self.deprocess_image(img).permute(1, 2, 0)
                plt.imshow(img)
                plt.axis("off")
                plt.show()


if __name__ == "__main__":
    #get command line arguments
    parser = argparse.ArgumentParser(description="Neural Style Transfer")

    parser.add_argument("--content", type=str, default="data/simba.jpg",
                        help="Path to content image")
    
    parser.add_argument("--style", type=str, default="data/style.png",
                        help="Path to style image")
    
    args = parser.parse_args()
    
    start = time.time()
    nst = NST(base_img_path = args.content,
            style_img_path = args.style)
    nst.train()
    end = time.time()
    print(end-start, "seconds") #runtime in seconds

    #display combination image
    img = nst.comb_img_params['comb_img'].detach().clone().reshape(3,224,224) #C,H,W -> H,W,C
    img = nst.deprocess_image(img).permute(1, 2, 0)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
