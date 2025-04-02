import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np
from random import choice

from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import math
import torch.utils.data as data

import random
import logging
import time

from datetime import datetime
import torchvision

import torch.optim as optim

import matplotlib.pyplot as plt

from swin_functions import *
from codebook_functions import *
from adaptive_functions import *

import argparse

codebook_path = './Codebook/codebook_4d_512clusters_mst.npy'
chunk_size = 4         # 4d vectors in the codebook
k = 512
TX_BINARY_BASE_PATH = './Binary/Transmitted_Binary/'
int_size = 8
NORMALIZE_CONSTANT = 20





save_directories = ["./recon/", "./Binary/Received_Text/", "./Binary/Received_Binary/", "./Binary/Transmitted_Binary/", "./Weights/", "./Datasets/"]

for save_dir in save_directories:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Args:
    def __init__(self):
        self.training = False  # Set to False if testing
        self.trainset = 'DIV2K'  # Choices: ['CIFAR10', 'DIV2K']
        self.testset = 'kodak'  # Choices: ['kodak', 'CLIC21', 'ffhq']
        self.distortion_metric = 'MSE'  # Choices: ['MSE', 'MS-SSIM']
        self.model = 'SwinJSCC_w/o_SAandRA'  # Choices: ['SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA']
        self.channel_type = 'rayleigh'  # Choices: ['awgn', 'rayleigh']
        self.C = '32'  # Bottleneck dimension, any string/number value can be set (32 = 1/48, 64 = 1/24, 96 = 1/16, 128 = 1/12)
        self.multiple_snr = '3'  # Random or fixed SNR, set as string (e.g., '10')
        self.model_size = 'base'  # Choices: ['small', 'base', 'large']

# Initialize the arguments
args = Args()


class config():
    seed = 42
    pass_channel = True
    #CUDA = True
    #device = torch.device("cuda:0")
    CUDA = torch.cuda.is_available()  # Check if CUDA is available
    device = torch.device("cuda:0" if CUDA else "cpu")  # Use GPU if available, otherwise CPU
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/D/Dataset/CIFAR10/"
        test_data_dir = "/media/D/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=channel_number,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        # train_data_dir = ["/media/D/Dataset/HR_Image_dataset/"]  
        base_path = "./Datasets/DIV2K"
        if args.testset == 'kodak':
            test_data_dir = ["./Datasets/Kodak"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/D/Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = ["/media/D/yangke/SwinJSCC/data/ffhq/"]

        train_data_dir = [base_path + '/DIV2K_train_HR/DIV2K_train_HR',
                          base_path + '/DIV2K_valid_HR/DIV2K_valid_HR']


        
        
        batch_size = 16
        downsample = 4
        if args.model == 'SwinJSCC_w/o_SAandRA' or args.model == 'SwinJSCC_w/_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 2, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 2, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        elif args.model_size =='large':
            encoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
                embed_dims=[128, 192, 256, 320], depths=[2, 2, 18, 2], num_heads=[4, 6, 8, 10], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
            decoder_kwargs = dict(
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128], depths=[2, 18, 2, 2], num_heads=[10, 8, 6, 4], C=channel_number,
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )

def load_weights(model_path):
    pretrained = torch.load(model_path, map_location=device)
    net.load_state_dict(pretrained, strict=False)
    del pretrained
    
if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(device)
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(device)


seed_torch()
logger = logger_configuration(config, save_log=False)
logger.info(config.__dict__)
torch.manual_seed(seed=config.seed)
net = SwinJSCC(args, config)
model_path = "./Weights/SwinJSCC_wo_SAandRA_Rayleigh_HRimage_snr3_psnr_C32.model"  
load_weights(model_path)
net = net.to(device)


# Without Codebook

def process_and_encode_image_to_binary(image_path, output_path, adaptive_patch_enabled=False, NORMALIZE_CONSTANT=20, int_size=8):
    
    with torch.no_grad():
        config.isTrain = False
        net.eval()

        # Load the image
        input_image, image_name = load_single_image(image_path, config)
        input_image = input_image.to(device)
        print(f"Image '{image_name}' loaded successfully")

        # Encode the image
        feature, chan_snr, channel_rate = encode_image(input_image, net)
        print("Image encoded successfully")

        # Quantize and save as binary file
        feature_np = feature.cpu().numpy()  # Convert PyTorch tensor to NumPy array

        if int_size == 8: quantized_feature = np.round(np.clip(feature_np / NORMALIZE_CONSTANT * 127, -127, 127)).astype(np.int8)
        elif int_size == 16: quantized_feature = np.round(np.clip(feature_np / NORMALIZE_CONSTANT * 32767, -32767, 32767)).astype(np.int16)

        # Prepare adaptive flag byte (7 redundant bits)
        bit_value = 1 if adaptive_patch_enabled else 0
        flag_byte = np.array([sum([bit_value << i for i in range(7)])], dtype=np.uint8)

        # Save flag byte and then quantized feature (append mode)
        flag_byte.tofile(output_path)

        with open(output_path, "ab") as f:
            quantized_feature.tofile(f)
        
        # quantized_feature.tofile(output_path)
        print(f"Encoded feature saved to '{output_path}'")


def main(image_path, img_no, use_codebook=False, adaptive=None):

    input_image, image_name= load_single_image(image_path, config)
    resized_image_np = input_image.squeeze(0).permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    # Convert RGB to BGR for OpenCV compatibility
    image = cv2.cvtColor(resized_image_np, cv2.COLOR_RGB2BGR)

    H_image, W_image = image.shape[:2]

    H_new, W_new = encode_image_adaptive(image, kernel_size=1,
                                        tl=100, th=200,
                                        v=50,       # quadtree edge threshold
                                        H=5,        # maximum quadtree depth
                                        Pm=28,      # base patch size before padding
                                        L=None,     # number of patches in the grid   (If None, it will be set to the number of adaptive patches)
                                        grid_image_file="patches_grid.png", coord_file="patch_coords.bin",
                                        padding=2)
    
    
    if adaptive is None:
        adaptive_patch_enabled = (H_new * W_new) < (0.7 * H_image * W_image)
    else:
        adaptive_patch_enabled = adaptive.lower() == "true"

    print(f"Original Resolution: {H_image} x {W_image}")
    print(f"Adaptive Patch enabled : {adaptive_patch_enabled}")

    if adaptive_patch_enabled:
        image_path = 'patches_grid.png'

    if use_codebook:
        print("Encoding with codebook...")
        codebook_path = 'Codebook/adaptive_patching_codebook_4d_512clusters_mst.npy' if adaptive_patch_enabled else 'Codebook/codebook_4d_512clusters_mst.npy'

        bin_path = TX_BINARY_BASE_PATH + "indices.bin"
        indices, _ = encode_image_with_nd_codebook(net, codebook_path, image_path, config, device, chunk_size)
        indices_np = indices.cpu().numpy()

        flag_byte = np.array([sum([1 << i for i in range(7)]) if adaptive_patch_enabled else 0], dtype=np.uint8)
        flag_byte.tofile(bin_path)

        with open(bin_path, "ab") as f:
            if k > 256:
                indices_np.astype(np.uint16).tofile(f)
            else:
                indices_np.astype(np.uint8).tofile(f)

        print(f"Encoded feature saved to {bin_path}")

    else:
        print("Encoding without codebook...")
        output_path = TX_BINARY_BASE_PATH + "encoded_feature.bin"
        process_and_encode_image_to_binary(image_path, output_path, adaptive_patch_enabled, NORMALIZE_CONSTANT, int_size)
    

    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--img_no", default="01")
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--adaptive", default=None, help="Set 'true' or 'false' to override threshold. Leave empty to use auto mode.")

    arguments = parser.parse_args()

    main(arguments.image_path, arguments.img_no, arguments.use_codebook, arguments.adaptive)


# Codebook mode, auto adaptive
# python transmitter.py --image_path Datasets/Kodak/kodim23.png --use_codebook

# # No codebook, force adaptive off
# python transmitter.py --image_path Datasets/Kodak/kodim23.png --adaptive false
