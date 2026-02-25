
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time

#NNST imports
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.io import decode_image
import torch.nn.functional as F
import torchvision.transforms as T

from pathlib import Path

import argparse

class NNST:
    def __init__(
        self,
        base_img_path, #content img
        style_img_path, #style img
        scale = 1, #we run nnst at 4 scales [1/8, 1/4, 1/2, 1/1]
        previous_output = None, #output from previous scale
        alpha = 0, #weight of output from previous scale (i.e. the parameter that controls the amount of stylization),
        reduce_memory_at_full_scale = False #use less memory so it fits on gpu
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scale = scale
        self.reduce_memory_at_full_scale = reduce_memory_at_full_scale

        #pretrained vgg16 model
        weights = models.VGG16_Weights.DEFAULT
        self.vgg16 = models.vgg16(weights=weights).features.to(self.device).eval()

        #preprocessing function expected for vgg16
        self.preprocess = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        #freeze weights
        for param in self.vgg16.parameters():
            param.requires_grad = False

        #read content image
        self.base_img = decode_image(base_img_path)

        #if content image has 4 channels (RGBA), get rid of last channel (alpha channel)
        if self.base_img.shape[0] == 4:
            self.base_img = self.base_img[:3]

        #reshape to 480p
        _, h, w = self.base_img.shape
        min_dim = min(h, w)
        if min_dim > 480:
            resize_scale = 480/min_dim
            new_h = int(round(h*resize_scale))
            new_w = int(round(w*resize_scale))
            
            self.base_img = F.interpolate(
                self.base_img.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).squeeze(0)
        
        #preprocess content image
        self.p_base_img = self.preprocess(self.base_img).unsqueeze(0).to(self.device)
        _, _, self.base_height, self.base_width = self.p_base_img.shape
        
        #load style image and remove alpha channel if necessary
        self.style_img = decode_image(style_img_path)
        if self.style_img.shape[0] == 4:
            self.style_img = self.style_img[:3]
        #style image should be square. it should match the smallest dim of the content image
        smaller_dim = min(self.base_height, self.base_width)
        self.style_img = F.interpolate(
            self.style_img.unsqueeze(0),
            size=(smaller_dim, smaller_dim),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        #preprocess style image
        self.p_style_img = self.preprocess(self.style_img).unsqueeze(0).to(self.device)

        #if we are applying nnst at a smaller scale, downsample the content and style image by that scale
        assert scale in [1/1, 1/2, 1/4, 1/8]
        if scale != 1:
            self.p_base_img  = F.interpolate(self.p_base_img, scale_factor=scale, mode='bilinear', align_corners=False)
            self.p_style_img  = F.interpolate(self.p_style_img, scale_factor=scale, mode='bilinear', align_corners=False)
        if previous_output != None:
            #if we have a previous output from the previous scale, upsample and add to the content image with alpha weightage
            previous_output = F.interpolate(
                previous_output,
                size=self.p_base_img.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            self.p_base_img = alpha*previous_output.to(self.device) + (1-alpha)*self.p_base_img

        _, _, self.base_height, self.base_width = self.p_base_img.shape

        #combination image that we optimize. Stored as a laplacian pyramid
        self.comb_img = self.get_laplace_pyr_from_img(self.p_base_img)
        self.comb_img = [lvl.clone().detach().requires_grad_(True).to(self.device) for lvl in self.comb_img]

        self.optimizer = torch.optim.Adam(
            self.comb_img,
            lr=5e-3,
            betas=(0.9, 0.999)
        )

        #print(self.p_base_img.shape, self.p_style_img.shape)
        self.targets = self.get_target_representation()

    def feature_extractor(self, x):
        outs = [] #holds outputs of select layers
        num_maxpools = 0
        #loop through all layers of vgg16
        for i, layer in enumerate(self.vgg16):
            x = layer(x) #get layer outputs
            #extract layer type name ("Conv2d", "ReLU", "MaxPool2d", etc ...)
            layer_type = str(self.vgg16[i].__class__).split(".")[-1][:-2]
            if layer_type == "MaxPool2d":
                num_maxpools += 1
                #we only want the first 4 blocks of the vgg model and each block is delimited by a MaxPool2d layer
                if num_maxpools == 4:
                    break
            elif layer_type == "ReLU": #we want the outputs of activation layers, within the first 4 blocks
                #bilinear interpolation so all the feature maps are of the same size
                resized = F.interpolate(x, size=(self.base_height//4, self.base_width//4), mode='bilinear')
                outs.append(resized)
            else:
                continue

        #each tensor in outs is of shape [batch_size, channels, orig_height//4, orig_width//4]
        #we concatenate along the channels dim, to get the hypercolumns
        outs = torch.cat(outs, dim=1)
        return outs

    def get_target_representation(self):
        with torch.no_grad():
            #get features for base img
            #shape: [2688, base_height//4, base_width//4] == [2688, 56, 56] (if image has dimensions (224,224))
            content_f = self.feature_extractor(self.p_base_img).squeeze(0)
            if self.scale == 1/1 and self.reduce_memory_at_full_scale:
                #get features for style image with different rotations
                #only use two orientations so the similarity matrix computed later on, fits on the gpu
                style_f = self.feature_extractor(
                    torch.cat([
                        self.p_style_img, #0 degrees
                        torch.rot90(self.p_style_img, k=1, dims=(2,3)), #90 degrees
                        #torch.rot90(self.p_style_img, k=2, dims=(2,3)), #180 degrees
                        #torch.rot90(self.p_style_img, k=3, dims=(2,3))  #270 degrees
                    ])
                ).permute(1,0,2,3) #shape: [2688, 4, base_height//4, base_width//4]
            else:
                #get features for style image with different rotations
                style_f = self.feature_extractor(
                    torch.cat([
                        self.p_style_img, #0 degrees
                        torch.rot90(self.p_style_img, k=1, dims=(2,3)), #90 degrees
                        torch.rot90(self.p_style_img, k=2, dims=(2,3)), #180 degrees
                        torch.rot90(self.p_style_img, k=3, dims=(2,3))  #270 degrees
                    ])
                ).permute(1,0,2,3) #shape: [2688, 4, base_height//4, base_width//4]


            content_vecs = content_f.flatten(1) #flatten all dimensions starting from dim 1: [C, H*W] -> [2688, 3136]
            style_vecs = style_f.flatten(1) #[C, B*H*W] == [2688, 12544]

            #zero center
            content_vecs_zc = content_vecs - content_vecs.mean(dim=1, keepdims=True)
            style_vecs_zc = style_vecs - style_vecs.mean(dim=1, keepdims=True)

            #cosine similarity
            content_norm = F.normalize(content_vecs_zc, dim=0) #divide each vector (one per content feature) by its magnitude
            style_norm = F.normalize(style_vecs_zc, dim=0) #divide each vector (one per style feature) by its magnitude
            similarity = content_norm.T @ style_norm #similarity matrix of shape [num_content_features, num_style_features]: [3136, 12544]

            #get index of nearest style feature for each content feature
            top_style = torch.argmax(similarity, dim=1)
            targets = style_vecs[:, top_style] #shape: [C, H*W]
            
        return targets

    def compute_loss(self, comb_img):
        #get features of the combination image
        comb_img_features = self.feature_extractor(comb_img).squeeze(0).flatten(1) #[C, H*W]

        #cosine similarity between combination img feature and target feature at each spatial location
        sim = F.cosine_similarity(comb_img_features, self.targets, dim=0)
        return -sim.mean()

    def get_laplace_pyr_from_img(self, img, layers=8):
        assert len(img.shape) == 4

        #create gaussian pyramid
        #largest layer (224x224) -> smallest layer (1x1)
        gaussian_pyramid = [img]
        blur_transform = T.GaussianBlur(kernel_size=(5, 5), sigma=1.0)
        for i in range(layers-1): #8 layer pyramid
            #gaussian blur and downsample to create each successive level of pyramid
            x = blur_transform(gaussian_pyramid[-1])
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear') #downsample by 2
            gaussian_pyramid.append(x)
            if min(x.shape[-1], x.shape[-2]) in [1, 2]:
                break

        #create laplacian pyramid
        #smallest layer (1x1) -> largest layer (224x224)
        laplacian_pyramid = [gaussian_pyramid[-1]]
        for i in range(len(gaussian_pyramid)-1,0,-1):
            #upsample to size of next layer
            x = F.interpolate(
                gaussian_pyramid[i],
                size=gaussian_pyramid[i-1].shape[-2:],
                mode='bilinear'
            )
            #difference between next layer and upsampled current layer
            laplacian_pyramid.append(gaussian_pyramid[i-1]-x)

        return laplacian_pyramid

    def get_img_from_laplace_pyr(self, laplacian_pyramid):
        #this recreates the gaussian_pyramid from the laplacian pyramid (in reverse order: largest layer (full image) at bottom)
        back2img = [laplacian_pyramid[0]]
        for i in range(len(laplacian_pyramid)-1):
            #upsample to size of next layer
            x = F.interpolate(
                back2img[i],
                size=laplacian_pyramid[i+1].shape[-2:],
                mode='bilinear'
            )
            #add next layer and upsampled current layer
            back2img.append(laplacian_pyramid[i+1]+x)

        return back2img[-1] #return final image

    def display_image(self, img):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)

        if img.dim() == 3:
            img = img.unsqueeze(0)

        #unnormalize image
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        #display
        img = img.squeeze(0).permute(1,2,0)
        plt.imshow(img.detach().numpy())
        plt.axis("off")
        plt.show()

    def optimize(self, steps=200, verbose=False):
        for step in range(steps):
            self.optimizer.zero_grad()
            #reconstruct iamge from laplacian pyramid
            recon_img = self.get_img_from_laplace_pyr(self.comb_img)
            loss = self.compute_loss(recon_img)
            #backpropagate loss and step optimizer
            loss.backward()
            self.optimizer.step()

            #print(step, loss.item())
            #display image if necessary
            if verbose and step%50 == 0:
                recon_img = self.get_img_from_laplace_pyr(self.comb_img)
                self.display_image(recon_img)

def save_reconstruction(nnst, final_path, lossless=False):
    img = nnst.get_img_from_laplace_pyr(nnst.comb_img)

    #if lossless saving is required, save as torch tensor
    if lossless:
        if final_path[-3:] != ".pt":
            pt_path = final_path.split(".")[0]
            pt_path += ".pt"
        torch.save(img.detach().cpu(), pt_path)
        #return
    
    #mean and std used by vgg16 normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)

    #ensure image has 4 dims
    if img.dim() == 3:
        img = img.unsqueeze(0)

    #unnormalize
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    #move channels to last for image library
    img = img.squeeze(0).permute(1,2,0)
    
    #save image
    Image.fromarray((img.detach().cpu().numpy() * 255).astype(np.uint8)).save(final_path)

def complete_process(
        base_img_path, 
        style_img_path, 
        final_path, 
        alpha=0.5, 
        lossless=False, 
        scales = [1/8, 1/4, 1/2, 1/1],
        prev_out = None,
        reduce_memory_at_full_scale=False
    ):
    #run nnst at all scales
    for scale in scales:
        #print("----------------------- ", scale)
        nnst = NNST(
            base_img_path = base_img_path,
            style_img_path = style_img_path,
            scale=scale,
            previous_output = prev_out,
            alpha = alpha,
            reduce_memory_at_full_scale=reduce_memory_at_full_scale
        )
        nnst.optimize(steps = 200)
        save_reconstruction(nnst, final_path, lossless=lossless)

        if scale != scales[-1]:
            #apart from when at the last scale, reconstruct output, which will be used to initialize the next scale
            prev_out = nnst.get_img_from_laplace_pyr(nnst.comb_img).detach()


if __name__ == '__main__':
    #get command line arguments
    parser = argparse.ArgumentParser(description="Neural Neighbour Style Transfer")

    parser.add_argument("--content", type=str, default="data/simba.jpg",
                        help="Path to content image")
    
    parser.add_argument("--style", type=str, default="data/style.png",
                        help="Path to style image")

    parser.add_argument("--output", type=str, default="data/final.png",
                        help="Path to final output")

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Stylization strength")
    
    args = parser.parse_args()

    complete_process(
        args.content, 
        args.style, 
        args.output, 
        alpha=args.alpha, 
        lossless=False,
        reduce_memory_at_full_scale=True
    )

    #if the gpu runs out of memory when running NNST on the full-scale image, try the following:
    #> complete_process(base_img_path, style_img_path, final_path, alpha=0.5, lossless=False, reduce_memory_at_full_scale=True)

    #If the gpu memory overflow issue persists after setting `reduce_memory_at_full_scale` to true, try the following:
    #Setting lossless to true will create a `.pt` file of the last output. 
    #> complete_process(base_img_path, style_img_path, final_path, alpha=0.5, lossless=True, scales = [1/8, 1/4, 1/2])

    # Full-scale processing can then be run on cpu with this file as input
    #> last_scale_pt = "output.pt"
    #> complete_process(base_img_path, style_img_path, final_path, alpha=0.5, lossless=True, scales = [1/1], prev_out = torch.load(last_scale_pt))
    
