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





# save_directories = ["./recon/", "./Binary/Received_Text/", "./Binary/Received_Binary/", "./Binary/Transmitted_Binary/", "./Weights/", "./Datasets/"]

# for save_dir in save_directories:
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)


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


def decode_and_evaluate(received_binary_path, image_path=None, resolution = (512,768), NORMALIZE_CONSTANT = 20, int_size=8, adaptive=False):           # Image path is original image

    tensor_shape = (1, int((resolution[0]*resolution[1])/(16*16)), 32)          # (Image size is 512x768x3 --> tensor (1, H/16 * W/16, 32))

    # if int_size == 8: 
    #     received_feature = np.fromfile(received_binary_path, dtype=np.int8).reshape(tensor_shape)
    #     recovered_feature = received_feature / 127 * NORMALIZE_CONSTANT
    
    # elif int_size == 16:
    #     received_feature = np.fromfile(received_binary_path, dtype=np.int16).reshape(tensor_shape)
    #     recovered_feature = received_feature / 32767 * NORMALIZE_CONSTANT

    with open(received_binary_path, "rb") as f:
        _ = f.read(1)  # Skip adaptive flag byte

        if int_size == 8:
            received_feature = np.frombuffer(f.read(), dtype=np.int8).reshape(tensor_shape)
            recovered_feature = received_feature / 127 * NORMALIZE_CONSTANT

        elif int_size == 16:
            received_feature = np.frombuffer(f.read(), dtype=np.int16).reshape(tensor_shape)
            recovered_feature = received_feature / 32767 * NORMALIZE_CONSTANT



    recovered_feature_tensor = torch.from_numpy(recovered_feature).float().to(device)

    with torch.no_grad():
        
        net.eval()
        chan_snr = 10
        cbr_used = '1/48'

        recon_image = decode_image(recovered_feature_tensor, net, chan_snr, resolution)
        print("Image decoded successfully")

    if image_path is not None:
        input_image, image_name = load_single_image(image_path, config)   # To compare with the reconstructed
        input_image = input_image.to(device)

    recon_image = recon_image.clamp(0, 1)

    if adaptive == False:

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, recon_image)
            plot_images(input_image, recon_image, image_name)
            #plot_single_image(recon_image)
            print(f"PSNR: {round(psnr_value.item(), 3)} dB")
            print(f'CBR = {cbr_used}')

        else:
            plot_single_image(recon_image)

    else:
        save_image(recon_image, f"reconstructed_patches_grid.png")

        reconstructed_image = decode_image_adaptive(grid_image_file="./recon/reconstructed_patches_grid.png",
                                                    coord_file="patch_coords.bin",Pm=28, padding=2)
        reconstructed_image = reconstructed_image.clamp(0,1)
        

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, reconstructed_image.to(device))
            plot_images(input_image, reconstructed_image, image_name)
            print(f"PSNR: {psnr_value:.2f} dB")
        else: 
            plot_single_image(reconstructed_image)



    #save_image(recon_image, f"reconstructed_{image_name}")


def decode_indices_and_plot (received_binary_path , codebook_path, image_path=None ,chunk_size=4 ,resolution = (512,768), k=512, adaptive=False):

    # if k > 256: loaded_indices = np.fromfile(received_binary_path, dtype=np.uint16)   # Use the same dtype as during saving
    # else: loaded_indices = np.fromfile(received_binary_path, dtype=np.uint8)

    with open(received_binary_path, "rb") as f:
        _ = f.read(1)  # Skip the first byte (flag already read outside)

        if k > 256:
            loaded_indices = np.frombuffer(f.read(), dtype=np.uint16)
        else:
            loaded_indices = np.frombuffer(f.read(), dtype=np.uint8)


    
    recovered_indices = torch.from_numpy(loaded_indices)   # Convert back to a PyTorch tensor
    recovered_indices = recovered_indices.int().to(device)  # Convert to int32 (Convert indices to a compatible type for clamping)

    # Calculate the number of bits needed based on k
    num_bits = int(np.ceil(np.log2(k)))  # Log base 2 of k gives the number of bits
    bitmask = (1 << num_bits) - 1       # Create a bitmask with 'num_bits' bits set to 1

    # Apply the bitmask to constrain the indices
    valid_indices = recovered_indices & bitmask    

    #valid_indices = torch.clamp(recovered_indices, min=0, max=511)  #  Clamp indices to the valid range  K=512
    valid_indices = torch.clamp(valid_indices, min=0, max=k-1)
    #valid_indices = recovered_indices

    M = int((resolution[0]*resolution[1])/(16*16))

    if image_path is not None:

        input_image, image_name = load_single_image(image_path, config)
        input_image = input_image.to(device)

    reconstructed = decode_image_with_nd_codebook(valid_indices, M, codebook_path, net, resolution, device, chunk_size)
    
    recon_image = reconstructed.clamp(0, 1)

    if adaptive == False:

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, recon_image)
            plot_images(input_image, recon_image, image_name)
            print(f"PSNR: {round(psnr_value.item(), 3)} dB")
            print('CBR = 1/48')
            print(f'{chunk_size}d vectors in codebook')

        else:
            plot_single_image(recon_image)

    else:
        save_image(recon_image, f"reconstructed_patches_grid.png")
        reconstructed_image = decode_image_adaptive(grid_image_file="./recon/reconstructed_patches_grid.png",
                                                    coord_file="patch_coords.bin",Pm=28, padding=2)
        reconstructed_image = reconstructed_image.clamp(0,1)

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, reconstructed_image.to(device))
            plot_images(input_image, reconstructed_image, image_name)
            print(f"PSNR: {psnr_value:.2f} dB")
        else:
            plot_single_image(reconstructed_image)


        

    #save_image(recon_image, f"reconstructed_{image_name}")


def main(received_filename, image_path=None, use_codebook=False, resolution=(512, 768)):
    with open(received_filename, "rb") as f:
        flag_byte = f.read(1)
    flag_int = int.from_bytes(flag_byte, byteorder='little')
    bits = [(flag_int >> i) & 1 for i in range(7)]
    adaptive_patch_enabled = sum(bits) >= 4
    print(f"Adaptive Patching Enabled: {adaptive_patch_enabled}")

    if adaptive_patch_enabled:
        codebook_path_final = 'Codebook/adaptive_patching_codebook_4d_512clusters_mst.npy'
    else:
        codebook_path_final = 'Codebook/codebook_4d_512clusters_mst.npy'

    if use_codebook:
        decode_indices_and_plot(received_filename, codebook_path_final, image_path, chunk_size, resolution, k, adaptive_patch_enabled)
    else:
        decode_and_evaluate(received_filename, image_path, resolution, NORMALIZE_CONSTANT, int_size, adaptive_patch_enabled)

print("Before Main")


if __name__ == "__main__":
    print("In main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--received_file", required=True)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--res_h", type=int, default=512)
    parser.add_argument("--res_w", type=int, default=768)
    arguments = parser.parse_args()

    main(arguments.received_file, arguments.image_path, arguments.use_codebook, (arguments.res_h, arguments.res_w))


# python receiver.py --received_file ./Binary/Received_Binary/received_indices.bin --image_path Datasets/Kodak/kodim23.png --use_codebook
