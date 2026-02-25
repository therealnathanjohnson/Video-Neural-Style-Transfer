import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.io import decode_image
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from PIL import Image
import random
import imageio
from pathlib import Path

import argparse

class Image_Analogies_Sweeps:
    def __init__(
        self,
        A_folder = "data/keyframes", #folder for keyframes
        A_prime_folder = "data/keyframes_stylized", #folder of stylized keyframes
        video_path = "data/video.mp4", #video to stylize
        data_dir_path = "data/output",
        use_edges = False,
        use_temporal_error_term = False,
        use_optical_flow = False
    ):
        self.use_edges = use_edges #use edge data when calculating errors
        self.use_temporal_error_term = use_temporal_error_term #use temporal error term when calculating errors

        #initalize NNFs aided by optical flows boolean
        self.use_optical_flow = use_optical_flow

        #we need optical flow for the temporal error term
        if self.use_temporal_error_term:
            self.use_optical_flow = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #we need a padding of 1 on all sides for 3x3 patches
        self.pad = 1

        #patch size
        self.p_size = (self.pad*2)+1

        #number of pyramid layers
        self.pyr_layer_count = 6

        self.data_dir_path = data_dir_path

        #objects to hold keyframe and stylized keyframe images
        self.A = {}
        self.A_prime = {}

        #A_folder should hold all the keyframe images
        #the filenames should have a "_{index}" at the end, before the extension, which corresponds to the index of the keyframe in the video
        for entry in Path(A_folder).iterdir():
            if entry.is_file():
                code = int(entry.stem.split("_")[-1])
                #read and add to object
                self.A[code] = plt.imread(entry)[:, :, :3]

        #A_prime_folder should hold all stylized keyframe images.
        #the filenames should have a "_{index}" at the end, before the extension, which corresponds to the index of the keyframe in the video
        for entry in Path(A_prime_folder).iterdir():
            if entry.is_file():
                code = int(entry.stem.split("_")[-1])
                #read and add to object
                self.A_prime[code] = plt.imread(entry)[:, :, :3]


        #keyframes indices of A and A' should match
        assert sorted(self.A.keys()) == sorted(self.A_prime.keys())
        self.num_keyframes = len(self.A)
        self.keyframes = sorted(self.A.keys())

        #obtain the gaussian pyramid of each frame of video, and store to file
        self.preprocess_video(video_path, data_dir_path, resize_scale=None)

        self.blur_transform = T.GaussianBlur(kernel_size=(self.p_size, self.p_size), sigma=0.6)

        #get gaussian pyramids with unfolded patches (all 3x3 patches extracted to a list) for each
        # keyframe/stylized keyframe at each pyramid level
        self.layer_sizes, self.data = self.get_unfolds()

        self.advected = None #frame advected by optical flow
        self.mask = None #mask with 1 for non-disoccluded and non-motion boundary regions

    #this function works for both single images and batches of images
    def get_gaussian_pyramid(self, img):
        squeeze_at_end = False
        if len(img.shape) < 4: #unbatched images
            img = img.unsqueeze(0)
            squeeze_at_end = True

        gaussian_pyramid = [img] #first layer consists of the image
        #each successive layer of the gaussian pyramid is formed by gaussian blurring and downsampling by 2
        blur_transform = T.GaussianBlur(kernel_size=(self.p_size, self.p_size), sigma=1.0)
        for i in range(self.pyr_layer_count-1): #6 layer pyramid
            x = blur_transform(gaussian_pyramid[-1])
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear') #downsample by 2
            gaussian_pyramid.append(x)

        #reverse so we have coarse to fine
        gaussian_pyramid.reverse()

        #pad each level so that we can extract 3x3 patches later without falling outside of the boundaries of the image
        gaussian_pyramid = [
            F.pad(level, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
            for level in gaussian_pyramid
        ]

        #if the input was just one image, we return the gaussian pyramid without
        # the batch dimension at each layer
        if squeeze_at_end:
            gaussian_pyramid = [x.squeeze(0) for x in gaussian_pyramid]

        return gaussian_pyramid

    def get_edges(self, gauss_pyr):
        #given a gaussian pyramid, for each layer, subtract blurred layer
        edges_list = []
        for i in range(len(gauss_pyr)):
            edges_list.append(gauss_pyr[i] - self.blur_transform(gauss_pyr[i]))
        return edges_list

    def opt_flow_processing(self, store):
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        frame_count, h, w, _ = store.shape

        start = time.time()
        forw_maps = []
        back_maps = []
        masks = []

        #for each position in the image, grid stores the coordinates
        grid = np.zeros((h,w,2), dtype=np.float32)
        grid[:, :, 0] = np.arange(w).reshape((1, w)) #x coordinates (width)
        grid[:, :, 1] = np.arange(h).reshape((h, 1)) #y coordinates (height)

        #gradient kernels to detect motion boundaries
        #horizontal gradient kernel
        kx = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3)
        #vertical gradient kernel
        ky = torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1)

        for img_index in range(frame_count-1):
            next_index = img_index+1

            #Convert to grayscale for deepflow
            gray1 = cv2.cvtColor(store[img_index], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(store[next_index], cv2.COLOR_RGB2GRAY)

            #Calculate optical flow
            forward_flow = deepflow.calc(gray1, gray2, None)
            backward_flow = deepflow.calc(gray2, gray1, None)

            #use the following in case cv2.optflow.createOptFlow_DeepFlow isn't available
            # forward_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # backward_flow = cv2.calcOpticalFlowFarneback(gray2, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            #add the forward flow to the grid to get postions after movement from frame 1 to frame 2
            forward_flow_map = grid.copy()
            forward_flow_map += forward_flow.astype(np.float32)
            forward_flow_map[:, :, 0] = np.clip(forward_flow_map[:, :, 0], 0, w - 1)
            forward_flow_map[:, :, 1] = np.clip(forward_flow_map[:, :, 1], 0, h - 1)

            #add the backward flow to the grid to get postions after movement from frame 2 to frame 1
            backward_flow_map = grid.copy()
            backward_flow_map += backward_flow.astype(np.float32)
            backward_flow_map[:, :, 0] = np.clip(backward_flow_map[:, :, 0], 0, w - 1)
            backward_flow_map[:, :, 1] = np.clip(backward_flow_map[:, :, 1], 0, h - 1)

            #forward flow warped by the backward flow. This should be roughly opposite of the backward flow apart from in regions
            # with disocclusions
            flow_tilde = cv2.remap(forward_flow,
                                backward_flow_map[:, :, 0].astype(np.float32),
                                backward_flow_map[:, :, 1].astype(np.float32),
                                interpolation=cv2.INTER_LINEAR
                                )

            flow_tilde = torch.tensor(flow_tilde)
            backward_flow = torch.tensor(backward_flow)

            #equation from Ruder et al. 2016 paper
            #detect disocclusions by finding points where th forward flow, warped by backward flow, isn't the opposite of bacward flow
            disoclussion = (torch.square(torch.norm(flow_tilde+backward_flow, dim=-1)) >
                            0.01*(torch.square(torch.norm(flow_tilde, dim=-1)) +
                                torch.square(torch.norm(backward_flow, dim=-1))) + 0.5
                        )

            #before calcualting the gradient, we first permute and squeeze backward flow to
            # get a shape of (2,1,h,w), from (h,w,2)
            reshaped_backward = backward_flow.permute(2,0,1).unsqueeze(1)
            #gradients along x and y axis to detect motion boundaries
            grad_x = F.conv2d(reshaped_backward, kx, padding=(0,1))
            grad_y = F.conv2d(reshaped_backward, ky, padding=(1,0))

            #back to (h,w,2)
            grad_x = grad_x.squeeze(1).permute(1,2,0)
            grad_y = grad_y.squeeze(1).permute(1,2,0)

            #equation from Ruder et al. 2016 paper for motion boundaries
            motion_boundaries = (torch.norm(grad_x, dim=-1)**2 +
                                torch.norm(grad_y, dim=-1)**2 >
                                0.01*torch.norm(backward_flow, dim=-1)**2 + 0.002)

            mask = ~(disoclussion | motion_boundaries)

            forw_maps.append(torch.tensor(forward_flow_map).round().long())
            back_maps.append(torch.tensor(backward_flow_map).round().long())
            masks.append(mask)
        end = time.time()
        print(end-start)

        #permute from (B,H,W,2) to (B,2,H,W) so we can use F.interpolate right away at a later point
        forw_maps = torch.stack(forw_maps).permute(0,3,1,2)
        back_maps = torch.stack(back_maps).permute(0,3,1,2)
        masks = torch.stack(masks) #shape: [B,H,W]

        torch.save(forw_maps, self.data_dir_path + '/forw_maps.pt')
        torch.save(back_maps, self.data_dir_path + '/back_maps.pt')
        torch.save(masks, self.data_dir_path + '/all_masks.pt')


    def preprocess_video(self, video_path, data_dir_path, resize_scale=2/3):
        """
        Opens video using CV2, computes gaussian pyramids for each frame and stores to file. The gaussian pyramids of all frames
        are stored together by level.

        Parameters:
            video_path: path of video to be stylized
            data_dir_path: path to store computed Gaussian Pyramids
            resize_scale: ratio 480 and smallest side of video. Set to None if the ratio is 1.
        """
        #check if a file for each gaussian pyramid level already exists
        files = os.listdir(data_dir_path)
        for i in range(self.pyr_layer_count):
            if f"level_{i}.pt" not in files:
                already_have_gauss_pyr = False
                break
            if i == self.pyr_layer_count-1:
                # print("No need to preprocess!")
                #return
                print("Already have Gaussian Pyramids")
                already_have_gauss_pyr = True

        if self.use_optical_flow:
            #check if the files from optical flow processing already exist
            if "forw_maps.pt" in files and "back_maps.pt" in files and "all_masks.pt" in files:
                print("opt flow files already exist")
                opt_flow_processing_done = True
            else:
                opt_flow_processing_done = False

            #if the gaussian pyramids and optical flows have been calculated, we can return
            if already_have_gauss_pyr and opt_flow_processing_done:
                print("No need to preprocess!")
                #return
        else:
            if already_have_gauss_pyr:
                print("No need to preprocess!")
                #return

        print("Preprocessing incomplete! Completing now ...")

        #open video
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        frame_count = 0
        store = []

        #read video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #OpenCV reads in BGR, convert to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #get smallest dim
            min_dim = min(frame_rgb.shape[0], frame_rgb.shape[1])
            #if the min_dim is greater than 480, we set resize_scale to resize to 480
            if min_dim > 480:
                resize_scale = 480/min_dim
            else:
                resize_scale = None

            #resize if necessary
            if resize_scale!=None:
              frame_rgb = cv2.resize(
                  frame_rgb,
                  None,
                  fx=resize_scale,
                  fy=resize_scale,
                  interpolation=cv2.INTER_AREA
              )

            #store
            store.append(frame_rgb)

            frame_count += 1

        cap.release()
        #stack frames, store as float32 instead of uint8, convert to torch tensor
        # and move the channels dim to the second position (for pytorch)
        store = np.stack(store)
        store = np.array(store, dtype=np.float32) / 255.0

        if self.use_optical_flow:
            #optical flow processing
            self.opt_flow_processing(store)

        store = torch.tensor(store, device=self.device)
        store = store.permute(0,3,1,2) #shape -> [num frames, channels, height, width]

        #get batched gaussian pyramids
        gauss_pyr = self.get_gaussian_pyramid(store)

        #save to gaussian pyramids with 1 file per level
        for i, level in enumerate(gauss_pyr):
            torch.save(level, data_dir_path + f'/level_{i}.pt')

    def get_unfolds(self):
        """
        coverts all keyframes and stylized keyframes from images to lists of 3x3 patches.
        """
        #store keyframe rgb gaussian pyramids
        A_rgb = {}
        #store stylized keyframe rgb gaussian pyramids
        A_prime_rgb = {}
        if self.use_edges:
            #store keyframe edge gaussian pyramids
            A_edge = {}
            #store stylized keyframe edge gaussian pyramids
            A_prime_edge = {}

        #get gaussian pyramids for all keyframes (A) and their stylized versions (A')
        for key in self.keyframes:
            A_rgb[key] = self.get_gaussian_pyramid(torch.tensor(self.A[key]).permute(2,0,1).to(self.device))
            A_prime_rgb[key] = self.get_gaussian_pyramid(torch.tensor(self.A_prime[key]).permute(2,0,1).to(self.device))
            if self.use_edges:
                A_edge[key] = self.get_edges(A_rgb[key])
                A_prime_edge[key] = self.get_edges(A_prime_rgb[key])

        #get height and with of each layer, unpadded, coarse to fine
        #we use A_prime_rgb[key] here but they should all have the same shapes
        layer_sizes = [[l.shape[-2]-(self.pad*2),l.shape[-1]-(self.pad*2)] for l in A_rgb[key]]

        #unfold each level of each pyramid for each keyframe, for both A and A'.
        #This gives us all the 3x3 patches of the images in each level
        A_rgb_unfold = {}
        Ap_rgb_unfold = {}
        if self.use_edges:
            A_edge_unfold = {}
            Ap_edge_unfold = {}
        for key in self.keyframes:
            A_rgb_unfold[key] = [
                F.unfold(layer, kernel_size=self.p_size, stride=1)
                for layer in A_rgb[key]
            ]
            Ap_rgb_unfold[key] = [
                F.unfold(layer, kernel_size=self.p_size, stride=1)
                for layer in A_prime_rgb[key]
            ]
            if self.use_edges:
                A_edge_unfold[key] = [
                    F.unfold(layer, kernel_size=self.p_size, stride=1)
                    for layer in A_edge[key]
                ]
                Ap_edge_unfold[key] = [
                    F.unfold(layer, kernel_size=self.p_size, stride=1)
                    for layer in A_prime_edge[key]
                ]

        #A_rgb_unfold has a key for each keyframe. for each key there is a guassian pyramid with 6 levels
        #We want to convert it so A_rgb_unfold becomes a list (instead of a dict) with one element per gaussian pyramid,
        # where each level has the data for all keyframes in one torch tensor
        A_rgb_unfold_cat = []
        Ap_rgb_unfold_cat = []
        if self.use_edges:
            A_edge_unfold_cat = []
            Ap_edge_unfold_cat = []
        for i in range(self.pyr_layer_count):
            A_rgb_unfold_cat.append(torch.stack([A_rgb_unfold[key][i] for key in self.keyframes]))
            Ap_rgb_unfold_cat.append(torch.stack([Ap_rgb_unfold[key][i] for key in self.keyframes]))
            if self.use_edges:
                A_edge_unfold_cat.append(torch.stack([A_edge_unfold[key][i] for key in self.keyframes]))
                Ap_edge_unfold_cat.append(torch.stack([Ap_edge_unfold[key][i] for key in self.keyframes]))


        #when refolding unfolded patches, F.fold computes the sum of overlapping pixels instead of the mean
        #we calculate the number of overlaps for each element by unfolding and folding a 1s tensor
        #Dividing by this, gives us the correct, averaged pixel values
        overlap_counts = [
            F.fold(
                F.unfold(torch.ones((1,h+(self.pad*2),w+(self.pad*2)), device=self.device).float(), kernel_size=self.p_size, stride=1),
                kernel_size=self.p_size,
                output_size=(h+(self.pad*2),w+(self.pad*2))
            )
            for (h,w) in layer_sizes
        ]

        data = {
            'A_rgb': A_rgb_unfold_cat,
            'Ap_rgb': Ap_rgb_unfold_cat,
            'overlap_counts': overlap_counts,
        }
        if self.use_edges:
            data['A_edge'] =  A_edge_unfold_cat
            data['Ap_edge'] = Ap_edge_unfold_cat

        return layer_sizes, data

    def merge_nnf(self, nnf1, nnf2, ratio=0.5, mask_nnf1=None):
        #create randomly merged nnf by choosing 50% (if ratio=0.5) of the patch offsets from nnf1 and 50% from nnf2
        #ATTENTION: A ratio lower than 0.5 favours NNF1, while ratios above 0.5 favour NNF2

        #The `mask_nnf1` argument, provides an optional mask where areas of
        # disocclusion and motion boundaries in NNF1 are set to False
        #We always want to select values from NNF2 at these locations, when merging

        h,w,_ = nnf1.shape

        if mask_nnf1 is None:
            #select from nnf1 everywhere if no maks is provided
            mask_nnf1 = torch.ones((h, w), dtype=torch.bool, device=self.device)
        else:
            #if mask_nnf1 is not None, we have to pad it. The masks are provided to us unpadded
            mask_nnf1 = F.pad(mask_nnf1, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)

        merge_mask = (torch.rand(h,w,1, device=self.device) > ratio) & mask_nnf1.view(h,w,1)
        merged_nnf = torch.where(
            merge_mask,
            nnf1,
            nnf2
        )

        return merged_nnf

    def upsample_nnf(self, nnf, level):
        #upsample nnf by 2 for next level of gaussian pyramid

        #get unpadded regions of nnf
        nnf_crop = nnf[self.pad:-self.pad,self.pad:-self.pad]
        #separate positions and keyframes
        nnf_crop_pos = nnf_crop[:, :, :2].clone()
        nnf_crop_keyframe = nnf_crop[:, :, 2].clone()

        nnf_crop_pos = nnf_crop_pos.permute(2,0,1).unsqueeze(0) #shape -> (1,2,H,W)
        nnf_crop_keyframe = nnf_crop_keyframe.unsqueeze(0).unsqueeze(0) #shape -> (1,1,H,W)

        #get min and max keyframe. We would like to ensusre that the upsampled NNF has the same min and max keyframe
        old_min_keyframe = nnf_crop_keyframe.min().item()
        old_max_keyframe = nnf_crop_keyframe.max().item()

        #scale size by 2 and values by 2
        new_h, new_w = self.layer_sizes[level]
        nnf_fine_pos = F.interpolate(
            nnf_crop_pos.float(),
            size=(new_h, new_w),
            mode='nearest'
        )*2 #mutliply by 2

        #scale keyframe tensor size by 2
        nnf_fine_keyframe = F.interpolate(
            nnf_crop_keyframe.float(),
            size=(new_h, new_w),
            mode='nearest'
        )

        #clamp keyframe values to old min and max
        nnf_fine_keyframe = torch.clamp(nnf_fine_keyframe, min=old_min_keyframe, max=old_max_keyframe)

        #add padding to positions and keyframes
        nnf_fine_pos = F.pad(nnf_fine_pos, (self.pad, self.pad, self.pad, self.pad), 'constant', 0).squeeze(0).permute(1,2,0).long()
        nnf_fine_keyframe = F.pad(nnf_fine_keyframe, (self.pad, self.pad, self.pad, self.pad), mode='replicate').squeeze(0).permute(1,2,0).long()

        #concatenate positions and keyframes
        nnf_fine = torch.cat([nnf_fine_pos, nnf_fine_keyframe], dim=2)

        #the following block of code ensures that offsets of the upsampled NNF don't lead to positions outside of the image
        #get indices of unpadded regions
        ind_y = torch.arange(new_h, device=self.device).view(new_h,1) + self.pad
        ind_x = torch.arange(new_w, device=self.device).view(1,new_w) + self.pad
        #convert from offsets to positions
        nnf_pos = nnf_fine[ind_y, ind_x, :2].clone() #skip index 2 of last dim since it denotes keyframe, not postition
        nnf_pos[:, :, 0] += ind_x
        nnf_pos[:, :, 1] += ind_y
        #clamp to boundaries of image
        nnf_pos[:, :, 0] = torch.clamp(nnf_pos[:, :, 0], min=self.pad, max=new_w+self.pad-1)
        nnf_pos[:, :, 1] = torch.clamp(nnf_pos[:, :, 1], min=self.pad, max=new_h+self.pad-1)
        #convert back to offsets
        nnf_pos[:, :, 0] -= ind_x
        nnf_pos[:, :, 1] -= ind_y
        #store clamped offsets
        nnf_fine[ind_y, ind_x, :2] = nnf_pos

        return nnf_fine

    def random_init_nnf(self, pyr_layer):
        """
        randomly initialize a nearest neighbour field (NNF)

        Parameters:
            pyr_layer: current gaussian level, which determines size of the NNF
        """
        h, w = self.layer_sizes[pyr_layer]
        #for each row, for each columln, the offset of the patch from the frame being optimized to its nearest neighbour in the keyframe
        #final dimension of the nnf has shape 2, instead of 3 -> (x,y,z) -> (x offset, y offset, index of keyframe)
        nnf = torch.zeros((h+(self.pad*2),w+(self.pad*2),3), dtype=torch.long, device=self.device)

        #indices of non-padding
        ind_y = torch.arange(h, device=self.device).view(h,1) + self.pad
        ind_x = torch.arange(w, device=self.device).view(1,w) + self.pad

        #random x and y positions in the image
        rand_y = torch.randint(0+self.pad, h+self.pad, (h, w), device=self.device)
        rand_x = torch.randint(0+self.pad, w+self.pad, (h, w), device=self.device)
        #z dimension tells us the index of the keyframe,
        #random_init_nnf is called at the very first frame, which is also the first keyframe, so the keyframe indices can be set to 0
        rand_z = torch.zeros((h, w), device=self.device).long()

        #calculate offsets
        nnf[ind_y, ind_x, 0] = rand_x - ind_x
        nnf[ind_y, ind_x, 1] = rand_y - ind_y
        #the keyframes aren't offsets but absolute values
        nnf[ind_y, ind_x, 2] = rand_z
        return nnf

    def get_error(self, *, b_data, keyframe_inds, patch_inds, l):
        """
        Get error between patches to be optimized and patches from keyframes, given by keyframe_inds and patch_inds

        Parameters:
            b_data: unfolded patches of frame to be optimized
            keyframe_inds: keyframe index of each selected nearest neighbour patch
            patch_inds: index of each selected nearest neighbour patch from selected keyframe
            l: current Gaussian pyramid level
        """
        assert keyframe_inds.shape == patch_inds.shape

        #error between rgb patches of frame and selected patches from keyframes
        rgb_error = torch.square(b_data['rgb']-self.data['A_rgb'][l][keyframe_inds, :, patch_inds].T).mean(dim=0)
        error = rgb_error

        if self.use_edges:
            #error between edge patches of frame and selected patches from keyframes
            edge_error = torch.square(b_data['edge']-self.data['A_edge'][l][keyframe_inds, :, patch_inds].T).mean(dim=0)
            error += 0.1*edge_error

        #self.advected is the prevous frame advected to the current frame (forward sweep) or the next frame advected
        # to the current frame (backward sweep), using optical flow
        #self.advected is None if we're at the first frame (there isn't a previous frame)
        if self.use_temporal_error_term and self.advected != None:
            h, w = self.layer_sizes[l]
            h_padded = h+(self.pad*2)
            w_padded = w+(self.pad*2)
            #obtain selected patches from the stylized keyframes
            select_patches = self.data['Ap_rgb'][l][keyframe_inds, :, patch_inds].T
            #refold patches into image
            refolded = F.fold(select_patches, kernel_size=self.p_size, output_size=(h_padded,w_padded))
            #F.fold sums all overlapping patches but we need the mean so we divide by overlap_counts
            final = refolded/self.data['overlap_counts'][l]
            final = final[:,self.pad:-self.pad,self.pad:-self.pad] #obtain unpadded region

            #convert the nnf from self.advected to a stylized image
            prev_advected = self.m_step(self.advected, l, source='Ap_rgb')

            #get error between the the stylized output of the advected previous (or next) frame and
            # the currently created stylized image for this frame
            #We want to penalize large changes across adjacent frames for temporal consistency
            temp_error = torch.square(prev_advected-final).mean(dim=0)
            #mask out dissocluded regions and motion boundaries
            temp_error[~self.mask] = 0
            temp_error = temp_error.flatten()

            error += 0.1*temp_error

        return error

    def propagation(self, nnf, *, b_data, pyr_layer, even=True):
        """
        If `even`, for each patch, we check whether the nearest neighbour of the patch to the left or above is a better solution for our patch,
        and modify the NNF offset for the current patch based on that.
        If not `even`, we check the patch to the right and the patch below instead of the patch the left and the patch above.

        Parameters:
            nnf: nearest neighbour field
            b_data: unfolded patches of frame to optimize
            pyr_layer: current gaussian pyramid level
            even: boolean
        """

        h, w = self.layer_sizes[pyr_layer]

        ind_y = torch.arange(h, device=self.device).view(h,1) + self.pad
        ind_x = torch.arange(w, device=self.device).view(1,w) + self.pad
        #convert from offsets to positions
        nnf_pos = nnf[:, :, :2].clone() #skip index 2 of last dim since it denotes keyframe, not postition
        nnf_pos[ind_y, ind_x, 0] += ind_x
        nnf_pos[ind_y, ind_x, 1] += ind_y

        #covert from positions to index of patch in unfolded patches
        patch_inds = ((nnf_pos[ind_y, ind_x,1]-self.pad)*w + (nnf_pos[ind_y, ind_x,0]-self.pad)).flatten()
        keyframe_inds = nnf[ind_y, ind_x, 2].flatten()

        #error between B patches and the patches in A indexed by keyframe_inds and patch_inds
        current_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer).reshape(h,w)
        current_pos = nnf_pos[ind_y, ind_x]
        current_keyframes = nnf[ind_y, ind_x, 2]

        if even:
            #compare error beteen each patch in B and the nearest neigbour in the keframes of the patch to the left

            #patches 1 pixel to the left (horizontal displacement)
            patch_inds = ((nnf_pos[ind_y, ind_x-1,1]-self.pad)*w + (nnf_pos[ind_y, ind_x-1,0]-self.pad)).flatten()
            keyframe_inds = nnf[ind_y, ind_x-1, 2].flatten()
            horiz_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer)

            #get positions of all patches one to the left and add 1. These are potential new positions.
            horiz_pos = nnf_pos[ind_y, ind_x-1].clone()
            horiz_pos[:, :, 0] = torch.clamp(horiz_pos[:, :, 0]+1, min=self.pad, max=w+self.pad-1)
            horiz_keyframes = nnf[ind_y, ind_x-1, 2]

            #mask out first column since it is now outside of the image
            horiz_error = horiz_error.reshape(h,w)
            horiz_error[:, 0] = float("inf")

            #compare error beteen each patch in B and the nearest neigbour in keyframes of the patch above
            #patches 1 pixel above (vertical displacement)
            patch_inds = ((nnf_pos[ind_y-1, ind_x,1]-self.pad)*w + (nnf_pos[ind_y-1, ind_x,0]-self.pad)).flatten()
            keyframe_inds = nnf[ind_y-1, ind_x, 2].flatten()
            vert_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer)

            #get positions of all patches 1 above and add 1. These are potential new positions
            vert_pos = nnf_pos[ind_y-1, ind_x].clone()
            vert_pos[:, :, 1] = torch.clamp(vert_pos[:, :, 1]+1, min=self.pad, max=h+self.pad-1)
            vert_keyframes = nnf[ind_y-1, ind_x, 2]

            #mask out top row since it is now outside of the image
            vert_error = vert_error.reshape(h,w)
            vert_error[0, :] = float("inf")

        else:
            #compare error beteen each patch in B and the nearest neigbour in keyframes of the patch to the right
            #patches one pixel to the right (horizontal displacement)
            patch_inds = ((nnf_pos[ind_y, ind_x+1,1]-self.pad)*w + (nnf_pos[ind_y, ind_x+1,0]-self.pad)).flatten()
            keyframe_inds = nnf[ind_y, ind_x+1, 2].flatten()
            horiz_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer)

            #get positions of all patches 1 to the left and subtract 1. These are potential new positions
            horiz_pos = nnf_pos[ind_y, ind_x+1].clone()
            horiz_pos[:, :, 0] = torch.clamp(horiz_pos[:, :, 0]-1, min=self.pad, max=w+self.pad-1)
            horiz_keyframes = nnf[ind_y, ind_x+1, 2]

            #mask out last column since it is now outside of the image
            horiz_error = horiz_error.reshape(h,w)
            horiz_error[:, -1] = float("inf")

            #compare error beteen each patch in B and the nearest neigbour in A of the patch below
            #patches 1 pixel below (vertical displacement)
            patch_inds = ((nnf_pos[ind_y+1, ind_x,1]-self.pad)*w + (nnf_pos[ind_y+1, ind_x,0]-self.pad)).flatten()
            keyframe_inds = nnf[ind_y+1, ind_x, 2].flatten()
            vert_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer)

            #get positions of all patches 1 below and subract 1. These are potential new positions
            vert_pos = nnf_pos[ind_y+1, ind_x].clone()
            vert_pos[:, :, 1] = torch.clamp(vert_pos[:, :, 1]-1, min=self.pad, max=h+self.pad-1)
            vert_keyframes = nnf[ind_y+1, ind_x, 2]

            #mask out last row since it is now outside of the image
            vert_error = vert_error.reshape(h,w)
            vert_error[-1, :] = float("inf")

        #stack errors and get min in the 0th dimension. -> shape: [3, h, w]; min shape: [h,w]
        min_error = torch.stack([current_error, horiz_error, vert_error]).argmin(dim=0)

        #get positions associated with min error
        all_pos = torch.stack([current_pos, horiz_pos, vert_pos])
        new_pos = all_pos[
            min_error.reshape(1,h,w,1),
            torch.arange(h, device=self.device).reshape(1,h,1,1),
            torch.arange(w, device=self.device).reshape(1,1,w,1),
            torch.arange(2, device=self.device).reshape(1,1,1,2)
        ].squeeze(0)

        #get keyframes associated with min error
        all_keyframes = torch.stack([current_keyframes, horiz_keyframes, vert_keyframes])
        new_keyframes = all_keyframes[
            min_error.reshape(1,h,w),
            torch.arange(h, device=self.device).reshape(1,h,1),
            torch.arange(w, device=self.device).reshape(1,1,w),
        ].squeeze(0)

        #convert new positions to offsets and store in nnf
        nnf[ind_y, ind_x, 0] = new_pos[:, :, 0] - ind_x
        nnf[ind_y, ind_x, 1] = new_pos[:, :, 1] - ind_y
        nnf[ind_y, ind_x, 2] = new_keyframes #store keyframes of the lowest error patches

        return nnf

    def random_search(self, nnf, *, current_frame, b_data, pyr_layer, alpha=0.5):
        """
        Docstring for random_search

        Parameters:
            nnf: nearest neighbour field
            current_frame: index of current frame
            b_data: unfolded patches of frame being optimized
            pyr_layer: current gaussian pyramid level
            alpha: window size parameter
        """
        h, w = self.layer_sizes[pyr_layer]
        max_size = max(h,w)

        ind_y = torch.arange(h, device=self.device).view(h,1) + self.pad
        ind_x = torch.arange(w, device=self.device).view(1,w) + self.pad
        #convert from offsets to positions
        nnf_pos = nnf[:, :, :2].clone()
        nnf_pos[ind_y, ind_x, 0] += ind_x
        nnf_pos[ind_y, ind_x, 1] += ind_y

        #gvien current_frame, we find the indices in self.keyframes of the two neighbouring keyframes
        #For example, if current_frame is 56, then the index of the two neigbouring keyframes are 2 and 3 (50 and 92)

        #the first keyframe and last keyframe should correspond to the first and last frames of the video, respectively
        assert current_frame >= self.keyframes[0] and current_frame <= self.keyframes[-1]

        #if the current frame is a keyframe, all random keyframes should equal that frame
        if current_frame in self.keyframes:
            rand_keyframes = torch.zeros((h,w), dtype=torch.long, device=self.device) + self.keyframes.index(current_frame)
        else:
            #get all keyframes that come after the current frame. get the first value from that list (i.e. next keyframe index)
            next_keyframe = torch.where(current_frame<torch.tensor(self.keyframes, dtype=torch.long, device=self.device))[0][0]
            #get previous keyframe
            prev_keyframe = next_keyframe-1

            #create a random tensor to store keyframes, and initalize to either the next or previous keyframe, depending on the distance
            # betwene the current frame to the adjacent keyframes. if the current frame is closer to the next keyframe, the next keyframe
            # should be chosen more often than the previous
            rand_keyframes = torch.rand((h,w), device=self.device)
            prob = (self.keyframes[next_keyframe] - current_frame) / (self.keyframes[next_keyframe] - self.keyframes[prev_keyframe])
            rand_keyframes[rand_keyframes<prob] = 0
            rand_keyframes[rand_keyframes>=prob] = 1
            rand_keyframes = rand_keyframes.long() + prev_keyframe

        power = 0
        while True:
            #covert from positions to index of patch in unfolded patches
            patch_inds = ((nnf_pos[ind_y, ind_x,1]-self.pad)*w + (nnf_pos[ind_y, ind_x,0]-self.pad)).flatten()
            keyframe_inds = nnf[ind_y, ind_x, 2].flatten()
            #error between B (and B') patches and the patches in A (and A') indexed by patch_inds
            current_error = self.get_error(b_data=b_data, keyframe_inds=keyframe_inds, patch_inds=patch_inds, l=pyr_layer).reshape(h,w)

            #window size
            win_size = int(max_size*alpha**power)

            win_min = nnf_pos.clone()
            win_max = nnf_pos.clone()

            #get the min and max x and y positions that define windows, clamped to the boundaries of the image
            win_min[ind_y, ind_x] = torch.clamp(nnf_pos[ind_y, ind_x] - win_size, min=self.pad)
            win_max[ind_y, ind_x, 0] = torch.clamp(nnf_pos[ind_y, ind_x, 0] + win_size, max=w+self.pad-1)
            win_max[ind_y, ind_x, 1] = torch.clamp(nnf_pos[ind_y, ind_x, 1] + win_size, max=h+self.pad-1)

            #pick random points within the windows
            rand_points = win_min + (win_max - win_min) * torch.rand(nnf_pos.shape, device=self.device)
            rand_points = rand_points.round().long()

            #covert from random positions to index of patch in unfolded patches
            patch_inds = ((rand_points[ind_y, ind_x,1]-self.pad)*w + (rand_points[ind_y, ind_x,0]-self.pad)).flatten()
            #error between B patches and the patches in keyframes indexed by random position patch_inds, and random keyframes `keyframe_inds`
            rand_error = self.get_error(
                b_data=b_data,
                keyframe_inds=rand_keyframes.flatten(),
                patch_inds=patch_inds,
                l=pyr_layer
            ).reshape(h,w)

            #stack error and find min along 0th dimension -> shape: [2, h, w]; min shape: [h,w]
            min_error = torch.stack([current_error, rand_error]).argmin(dim=0)

            #get positions associated with min error
            all_pos = torch.stack([nnf_pos[ind_y, ind_x], rand_points[ind_y, ind_x]])
            new_pos = all_pos[
                min_error.reshape(1,h,w,1),
                torch.arange(h, device=self.device).reshape(1,h,1,1),
                torch.arange(w, device=self.device).reshape(1,1,w,1),
                torch.arange(2, device=self.device).reshape(1,1,1,2)
            ].squeeze(0)


            #get keyframes associated with min error
            all_keyframes = torch.stack([nnf[ind_y, ind_x, 2], rand_keyframes])
            new_keyframes = all_keyframes[
                min_error.reshape(1,h,w),
                torch.arange(h, device=self.device).reshape(1,h,1),
                torch.arange(w, device=self.device).reshape(1,1,w),
            ].squeeze(0)

            #update nnf posiitons and keyframes with lower error patch data
            nnf_pos[ind_y, ind_x] = new_pos
            nnf[ind_y, ind_x, 2] = new_keyframes

            power += 1
            if win_size <= 1:
                break

        #covert from positions to offsets
        nnf[ind_y, ind_x, 0] = nnf_pos[ind_y, ind_x, 0] - ind_x
        nnf[ind_y, ind_x, 1] = nnf_pos[ind_y, ind_x, 1] - ind_y

        return nnf

    #all args after `*` are mandatory and keyword-only, other than `iters` since a default has been provided
    def e_step(self, nnf, *, current_frame, b_data, pyr_layer, iters=6):
        #call propagation and random search `iters` times
        for i in range(iters):
            even = i%2 == 0
            nnf = self.propagation(nnf, b_data=b_data, pyr_layer=pyr_layer, even=even)
            nnf = self.random_search(nnf, current_frame=current_frame, b_data=b_data, pyr_layer=pyr_layer)
        return nnf

    def m_step(self, nnf, pyr_layer, source='A_rgb'):
        """
        Patchvoting to create a new image based on the nearest neighbours in the keyframes (or stylized keyframes) of every patch
        in the image to be optimized

        Parameters:
            nnf: nearest neighbour field
            pyr_layer: gaussian pyramid level
            source: 'A_rgb' to create a new image based on keyframes, 'Ap_rgb' to create a new image based on stylized keyframes
        """

        h, w = self.layer_sizes[pyr_layer]

        ind_y = torch.arange(h, device=self.device).view(h,1) + self.pad
        ind_x = torch.arange(w, device=self.device).view(1,w) + self.pad
        #convert from offsets to positions
        nnf_pos = nnf[:, :, :2].clone()
        nnf_pos[ind_y, ind_x, 0] += ind_x
        nnf_pos[ind_y, ind_x, 1] += ind_y

        h_padded = h+(self.pad*2)
        w_padded = w+(self.pad*2)

        #go from position coordinates of nearest neighbours to patch indices in unfolded
        patch_inds = ((nnf_pos[ind_y, ind_x,1]-self.pad)*w + (nnf_pos[ind_y, ind_x,0]-self.pad)).flatten()
        keyframe_inds = nnf[ind_y, ind_x, 2].flatten()

        #shuffle (or duplicate) patches from source (either keyframes or stylized keyframes) using above patch indices
        shuffled_patches = self.data[source][pyr_layer][keyframe_inds, :, patch_inds].T

        #refold list of patches
        refolded = F.fold(shuffled_patches, kernel_size=self.p_size, output_size=(h_padded,w_padded))

        #F.fold sums all overlapping patches but we need the mean so we divide by overlap_counts
        final = refolded/self.data['overlap_counts'][pyr_layer]

        return final[:,self.pad:-self.pad,self.pad:-self.pad] #return only unpadded region

    def advect_nnf(self, nnf, flow_map):
        #advect nnf using an optical flow map

        h_pad, w_pad, _ = nnf.shape
        h = h_pad-(self.pad*2)
        w = w_pad-(self.pad*2)
        ind_y = torch.arange(h, device=self.device).view(h,1) + self.pad
        ind_x = torch.arange(w, device=self.device).view(1,w) + self.pad
        #convert from offsets to positions
        nnf_pos = nnf.clone()
        nnf_pos[ind_y, ind_x, 0] += ind_x
        nnf_pos[ind_y, ind_x, 1] += ind_y

        old_min_keyframe = nnf_pos[:, :, 2].min().item()
        old_max_keyframe = nnf_pos[:, :, 2].max().item()

        #the flow map tells us, for each pixel position, the new pixel position it moved to
        #we get the nnf_pos values at these new positions when advecting
        advected = nnf_pos[flow_map[:, :, 1]+self.pad, flow_map[:, :, 0]+self.pad]

        #clamp positions to image range
        advected[:, :, 0] = torch.clamp(advected[:, :, 0], min=self.pad, max=w+self.pad-1)
        advected[:, :, 1] = torch.clamp(advected[:, :, 1], min=self.pad, max=h+self.pad-1)

        #convert back from positions to offsets
        advected[:, :, 0] -= ind_x
        advected[:, :, 1] -= ind_y

        #clamp keyframe values to old min and max
        advected[:, :, 2] = torch.clamp(advected[:, :, 2], min=old_min_keyframe, max=old_max_keyframe)

        #pad. advected has shape (h,w,3), we permute to channels first and permute back
        advected = F.pad(advected.permute(2,0,1), (self.pad, self.pad, self.pad, self.pad), 'constant', 0).permute(1,2,0)

        return advected

    def patch_match_sweep(self, e_iters=10, backward_sweep=False):
        #holds the nnfs for each frame, at the previous and current gaussian pyramid levels, respectively
        previous_layer_nnfs = []
        current_layer_nnfs = []

        if self.use_optical_flow:
            #load forward and backward flow maps and masks (for disocclusion and motion boundaries)
            #forw_maps, back_maps and masks, all have a batch_size 1 less than the number of frames
            forw_maps = torch.load(self.data_dir_path + '/forw_maps.pt', map_location=self.device)
            back_maps = torch.load(self.data_dir_path + '/back_maps.pt', map_location=self.device)
            masks = torch.load(self.data_dir_path + '/all_masks.pt', map_location=self.device)

        for i in range(self.pyr_layer_count):
            #load the gaussian pyramid data for all frames, at the current level
            level = torch.load(f"{self.data_dir_path}/level_{i}.pt", map_location=self.device)
            num_frames, _, h, w = level.shape

            if self.use_optical_flow:
                #downsize (if necessary) forward and backward flows and masks
                scale = 2**(self.pyr_layer_count-i-1)
                if scale!=1:
                    forw_d = F.interpolate(forw_maps.float(), scale_factor=1/scale, mode='bilinear', align_corners=False) / scale
                    forw_d = forw_d.round().long().permute(0,2,3,1)
                    back_d = F.interpolate(back_maps.float(), scale_factor=1/scale, mode='bilinear', align_corners=False) / scale
                    back_d = back_d.round().long().permute(0,2,3,1)
                    mask_d = F.interpolate(masks.unsqueeze(1).float(), scale_factor=1/scale, mode='nearest').bool().squeeze(1)
                else:
                    forw_d = forw_maps.permute(0,2,3,1)
                    back_d = back_maps.permute(0,2,3,1)
                    mask_d = masks

            #forward sweep
            for j in range(num_frames):
                print(f"level {i} forward frame {j}")

                #get unfolded rgb (and edge, if neccessary) patches for current frame
                b_unfold_rgb = F.unfold(level[j], kernel_size=self.p_size, stride=1)
                b_unfold = {"rgb":b_unfold_rgb}
                if self.use_edges:
                    b_unfold_edge = F.unfold(level[j]-self.blur_transform(level[j]), kernel_size=self.p_size, stride=1)
                    b_unfold["edge"] = b_unfold_edge


                if previous_layer_nnfs == []:
                    #this will only apply at the coarsest level
                    #if there isn't a previous layer, randomly initlize the nnf for the previous layer
                    prev_l_nnf = self.random_init_nnf(pyr_layer=0)
                else:
                    #upsample the previous layer's nnf at the current frame, so it matches the size of the current gaussian pyramid level
                    prev_l_nnf = self.upsample_nnf(previous_layer_nnfs[j], level=i)


                if current_layer_nnfs == []:
                    #current_layer_nnfs only equals [] at the first frame (applies to all levels)
                    nnf = prev_l_nnf
                    self.advected = None
                else:
                    #randomly merge previous frame nnf (at current level) with previous level nnf (at current frame)

                    if self.use_optical_flow:
                        #advect previous frame to current frame using the optical flow between the current frame and
                        # previous frame (backward optical flow)
                        advected = self.advect_nnf(current_layer_nnfs[j-1], back_d[j-1])
                        #set self.advected and self.mask so it may be used during get_error
                        self.advected = advected
                        self.mask = mask_d[j-1]
                        nnf = self.merge_nnf(advected, prev_l_nnf, mask_nnf1=mask_d[j-1], ratio=0.5)
                    else:
                        nnf = self.merge_nnf(current_layer_nnfs[j-1], prev_l_nnf, ratio=0.5)

                #run e_step followed by m_step, 5 times
                for _ in range(5):
                    nnf = self.e_step(nnf, current_frame=j, b_data=b_unfold, pyr_layer=i, iters=e_iters)
                    b = self.m_step(nnf, i)
                    #after forming the new image in the m_step, we pad, unfold, and use it in the next e_step
                    b_padded = F.pad(b, (self.pad, self.pad, self.pad, self.pad), mode='replicate')

                    b_unfold_rgb = F.unfold(b_padded, kernel_size=self.p_size, stride=1)
                    b_unfold = {"rgb":b_unfold_rgb}
                    if self.use_edges:
                        b_unfold_edge = F.unfold(b_padded-self.blur_transform(b_padded), kernel_size=self.p_size, stride=1)
                        b_unfold["edge"] = b_unfold_edge
                level[j] = b_padded
                current_layer_nnfs.append(nnf)

            #backward sweep
            if backward_sweep:
                for j in range(num_frames-1, -1, -1):
                    print(f"level {i} backward frame {j}")
                    #skip the very last frame since we have no future information for this frame
                    if j == num_frames-1:
                        continue
                    #unfold
                    b_unfold_rgb = F.unfold(level[j], kernel_size=self.p_size, stride=1)
                    b_unfold = {"rgb":b_unfold_rgb}
                    if self.use_edges:
                        b_unfold_edge = F.unfold(level[j]-self.blur_transform(level[j]), kernel_size=self.p_size, stride=1)
                        b_unfold["edge"] = b_unfold_edge


                    if self.use_optical_flow:
                        #advect next frame to current frame using the optical flow between the current frame and
                        # next frame (forward optical flow)
                        advected = self.advect_nnf(current_layer_nnfs[j+1], forw_d[j])
                        self.advected = advected
                        self.mask = mask_d[j]
                        nnf = self.merge_nnf(advected, current_layer_nnfs[j], mask_nnf1=mask_d[j], ratio=0.5)
                    else:
                        #initialize nnf using output of this frame and future frame
                        nnf = self.merge_nnf(current_layer_nnfs[j], current_layer_nnfs[j+1], ratio=0.5)

                    #run e_step followed by m_step, 5 times
                    for _ in range(5):
                        nnf = self.e_step(nnf, current_frame=j, b_data=b_unfold, pyr_layer=i, iters=e_iters)
                        b = self.m_step(nnf, i)
                        b_padded = F.pad(b, (self.pad, self.pad, self.pad, self.pad), mode='replicate')

                        b_unfold_rgb = F.unfold(b_padded, kernel_size=self.p_size, stride=1)
                        b_unfold = {"rgb":b_unfold_rgb}
                        if self.use_edges:
                            b_unfold_edge = F.unfold(b_padded-self.blur_transform(b_padded), kernel_size=self.p_size, stride=1)
                            b_unfold["edge"] = b_unfold_edge

                    level[j] = b_padded
                    #store the frame
                    current_layer_nnfs[j] = nnf

            #when moving to the next pyramid layer, store current_layer_nnfs in previous layer_nnfs
            previous_layer_nnfs = current_layer_nnfs

            #reset current_layer_nnfs
            current_layer_nnfs = []
            print(f"finished layer {i}")

        return previous_layer_nnfs

    #pass output output of patch_match_sweep() to this function to construct stylized video
    def construct_video(self, nnf_list):
        output = []
        for i, nnf in enumerate(nnf_list):
            #construct the image from the nnf by us the stylized keyframes as the source
            construction = self.m_step(nnf, -1, source='Ap_rgb') #-1 to indicate we're at the largest pyramid level
            output.append(construction)

        return output
    
if __name__ == "__main__":
    #get command line arguments
    parser = argparse.ArgumentParser(description="Video Neural Style Transfer")

    parser.add_argument("--video", type=str, default="data/video.mp4",
                        help="Path to input video which must be stylized")
    
    parser.add_argument("--keyframes_dir", type=str, default="data/keyframes",
                        help="Directory to store extracted keyframes")
    
    parser.add_argument("--stylized_keyframes_dir", type=str, default="data/keyframes_stylized",
                        help="Directory to store stylized keyframes")
    
    parser.add_argument("--output_dir", type=str, default="data/output",
                        help="Directory for final output video")

    #with `action="store_true"`, use_edges will be false unless this option is used
    parser.add_argument("--use_edges", action="store_true",
                    help="Use edge features in Image Analogies")

    parser.add_argument("--use_temporal_error_term", action="store_true",
                        help="Enable temporal consistency penalty")
    
    parser.add_argument("--no_optical_flow", action="store_true",
                        help="Disable optical flow")
    
    parser.add_argument("--backward_sweep", action="store_true",
                        help="Run PatchMatch backward sweep")
    
    args = parser.parse_args()

    #create output dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #create sweeping Image Analogies instance
    ia = Image_Analogies_Sweeps(
        A_folder = args.keyframes_dir, #folder for keyframes
        A_prime_folder = args.stylized_keyframes_dir, #folder of stylized keyframes
        video_path = args.video, #video to stylize
        data_dir_path = args.output_dir,
        use_edges = args.use_edges, 
        use_temporal_error_term = args.use_temporal_error_term, 
        use_optical_flow = not args.no_optical_flow
    )
    #run patch match sweep processing
    nnf_list = ia.patch_match_sweep(backward_sweep=args.backward_sweep)
    #use NNFs to construct list of stylized frames
    output_list = ia.construct_video(nnf_list)
    #save stylized frames as video
    output_list_numpy = np.stack([x.cpu() for x in output_list])
    output_list_numpy = (output_list_numpy * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(ia.data_dir_path+"/final_output.mp4", np.permute_dims(output_list_numpy, (0,2,3,1)), fps=25)
