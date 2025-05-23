import warnings
import subprocess

warnings.filterwarnings("ignore")

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
import struct

from datetime import datetime
import torchvision

import torch.optim as optim

import matplotlib.pyplot as plt

from swin_functions import *
from codebook_functions import *
from adaptive_functions import *

import argparse



#codebook_path = './Codebook/codebook_4d_512clusters_mst.npy'
# chunk_size = 4         # 4d vectors in the codebook
# k = 512

TX_BINARY_BASE_PATH = './Binary/Transmitted_Binary/'
int_size = 8
NORMALIZE_CONSTANT = 20

# codebook_path_wo_adaptive = './Codebook/codebook_4d_512clusters_mst.npy'
# codebook_path_adaptive = 'Codebook/adaptive_patching_codebook_4d_512clusters_mst.npy'

# paths --> (chunk_size, k) : {adaptive, wo_adaptive}

codebook_paths = {
    (4, 512): {
        "adaptive": 'Codebook/adaptive_patching_codebook_4d_512clusters_mst.npy',
        "wo_adaptive": './Codebook/codebook_4d_512clusters_mst.npy'
    },
    # You can add more predefined paths as needed
    (2, 256): {
        "adaptive": 'Codebook/adaptive_patching_codebook_2d_256clusters_mst.npy',
        "wo_adaptive": './Codebook/codebook_2d_256clusters_mst.npy'
    },
    (8, 1024): {
        "adaptive": 'Codebook/adaptive_patching_codebook_8d_1024clusters_mst.npy',
        "wo_adaptive": './Codebook/codebook_8d_1024clusters_mst.npy'
    }
}





save_directories = ["./recon/", "./Binary/Received_Text/", "./Binary/Received_Binary/", "./Binary/Transmitted_Binary/", "./Weights/", "./Datasets/"]

for save_dir in save_directories:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def capture_image_from_pi_camera(save_path='captured_image.png', timeout_ms=3000):
#     try:
#         # Add a delay to let the camera auto-adjust
#         command = ['libcamera-still', '--encoding', 'png', '-t', str(timeout_ms), '-o', save_path]
#         print(f"Capturing image using libcamera-still with {timeout_ms} ms delay...")
#         subprocess.run(command, check=True)
#         print(f"Image captured and saved to '{save_path}'")
#         return save_path
#     except subprocess.CalledProcessError as e:
#         print("Failed to capture image with libcamera-still.")
#         print(e)
#         exit(1)

# import subprocess

# def capture_image_from_pi_camera(save_path='captured_image.png', timeout_ms=3000, shutter_speed=10000):
#     try:
#         # Construct the command with the specified shutter speed
#         command = ['libcamera-still', '--encoding', 'png', '-t', str(timeout_ms), '--shutter', str(shutter_speed), '-o', save_path]

#         print(f"Capturing image using libcamera-still with {timeout_ms} ms delay and {shutter_speed} �s shutter speed...")

#         subprocess.run(command, check=True)

#         print(f"Image captured and saved to '{save_path}'")

#         return save_path

#     except subprocess.CalledProcessError as e:
#         print("Failed to capture image with libcamera-still.")
#         print(e)
#         exit(1)



def capture_image_from_pi_camera(save_path='captured_image.png', timeout_ms=3000, shutter_speed=10000, width=1024, height=768):
    try:
        # Construct the command with the specified resolution and shutter speed
        command = [
            'libcamera-still', '--encoding', 'png', 
            '-t', str(timeout_ms), '--shutter', str(shutter_speed), 
            '--width', str(width), '--height', str(height), 
            '-o', save_path
        ]

        print(f"Capturing image using libcamera-still with {timeout_ms} ms delay, {shutter_speed} s shutter speed, and resolution {width}x{height}...")

        subprocess.run(command, check=True)

        print(f"Image captured and saved to '{save_path}'")

        return save_path

    except subprocess.CalledProcessError as e:
        print("Failed to capture image with libcamera-still.")
        print(e)
        exit(1)



# Example usage:
#capture_image_from_pi_camera(save_path='captured_image.png', timeout_ms=3000, shutter_speed=10000)



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


def combine_binary_files(file1, file2, output_file):
    # output_path = os.path.join("/kaggle/working", output_file)
    
    with open(file1, "rb") as f1, open(file2, "rb") as f2, open(output_file, "wb") as out:
        data1 = f1.read()
        data2 = f2.read()      
        size1 = len(data1)  # Get size of first file        
        # Store size of first file as metadata (4 bytes for compatibility)
        out.write(struct.pack("I", size1))  # 'I' stores an unsigned 4-byte integer       
        # Write actual binary data
        out.write(data1)
        out.write(data2)
    print(f"Files {file1} and {file2} combined into {output_file} with metadata.")

seed_torch()
logger = logger_configuration(config, save_log=False)
logger.disabled = True
logger.info(config.__dict__)
torch.manual_seed(seed=config.seed)

# net = SwinJSCC(args, config)
# model_path = "./Weights/SwinJSCC_wo_SAandRA_Rayleigh_HRimage_snr3_psnr_C32.model"  
# load_weights(model_path)
# net = net.to(device)

model_fp32 = SwinJSCC(args, config)
net = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
net.load_state_dict(torch.load("swinjscc_quantized.pt"))

net = net.to(device)
net.eval()


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


def main(image_path, use_codebook=False, adaptive=None):

    image_size_kb = os.path.getsize(image_path) / 1024

    input_image, image_name= load_single_image(image_path, config)
    resized_image_np = input_image.squeeze(0).permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    # Convert RGB to BGR for OpenCV compatibility
    image = cv2.cvtColor(resized_image_np, cv2.COLOR_RGB2BGR)

    H_image, W_image = image.shape[:2]
    
    if adaptive is not None and adaptive.lower() == "false":
        adaptive_patch_enabled = False
        
    else:

        H_new, W_new, data_pixels, L = encode_image_adaptive(image, kernel_size=1,
                                            tl=50, th=200,
                                            v=50,       # quadtree edge threshold
                                            H=5,        # maximum quadtree depth
                                            Pm=28,      # base patch size before padding
                                            L=None,     # number of patches in the grid   (If None, it will be set to the number of adaptive patches)
                                            grid_image_file="patches_grid.png", coord_file="patch_coords.bin",
                                            padding=2, visualize=False)
        
    
        if adaptive is None:
            #adaptive_patch_enabled = (H_new * W_new) < (0.7 * H_image * W_image)
            adaptive_patch_enabled = (data_pixels < (0.7 * H_image * W_image)) or (L < 100)
        else:
            adaptive_patch_enabled = adaptive.lower() == "true"

    print(f"Original Resolution: {H_image} x {W_image}")
    print(f"Adaptive Patch enabled : {adaptive_patch_enabled}")

    if adaptive_patch_enabled:
        image_path = 'patches_grid.png'

    if use_codebook:
        print("Encoding with codebook...")
        codebook_path = codebook_path_adaptive if adaptive_patch_enabled else codebook_path_wo_adaptive
        print(f"Using codebook: {codebook_path}")
        if not adaptive_patch_enabled:
            resolution_data = f"Resolution: {H_image}x{W_image}".encode('utf-8')
            # Open the binary file in write-binary mode and overwrite it
            binary_file_path = "patch_coords.bin"  # Replace with your binary file path
            with open(binary_file_path, 'wb') as f:
                f.write(resolution_data)
            print(f"Binary file overwritten with resolution: {H_image}x{W_image}")

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

        bin_file_path = bin_path

        print(f"Encoded feature saved to {bin_path}")
        combine_binary_files("patch_coords.bin", bin_path, "combined_binary.bin")

    else:
        print("Encoding without codebook...")

        if not adaptive_patch_enabled:
            resolution_data = f"Resolution: {H_image}x{W_image}".encode('utf-8')
            # Open the binary file in write-binary mode and overwrite it
            binary_file_path = "patch_coords.bin"  # Replace with your binary file path
            with open(binary_file_path, 'wb') as f:
                f.write(resolution_data)
            print(f"Binary file overwritten with resolution: {H_image}x{W_image}")
        

        output_path = TX_BINARY_BASE_PATH + "encoded_feature.bin"
        process_and_encode_image_to_binary(image_path, output_path, adaptive_patch_enabled, NORMALIZE_CONSTANT, int_size)
        combine_binary_files("patch_coords.bin", output_path, "combined_binary.bin")
        bin_file_path = output_path
    
    # Print image and file size
    
    bin_size_kb = os.path.getsize(bin_file_path) / 1024
    print(f"Original image size: {image_size_kb:.2f} KB")
    print(f"Binary file size for transmission: {bin_size_kb:.2f} KB")

    compression_ratio = image_size_kb / bin_size_kb if bin_size_kb > 0 else 0
    print(f"Compression Ratio (Original / Transmitted): {compression_ratio:.2f}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=None, help="Path to image file.")
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--adaptive", default=None, help="Set 'true' or 'false' to override threshold. Leave empty to use auto mode.")
    parser.add_argument("--use_camera", action="store_true", help="Capture image using Pi Camera.")
    parser.add_argument("--k", type=int, default=512, help="Number of clusters in the codebook")
    parser.add_argument("--chunk_size", type=int, default=4, help="Size of vector chunks for quantization")

    arguments = parser.parse_args()

    k = arguments.k
    chunk_size = arguments.chunk_size

    key = (chunk_size, k)
    codebook_path_adaptive = codebook_paths[key]["adaptive"]
    codebook_path_wo_adaptive = codebook_paths[key]["wo_adaptive"]

    if arguments.use_camera:
        image_path = capture_image_from_pi_camera()
    else:
        if not arguments.image_path:
            print("Please provide --image_path or use --use_camera")
            exit(1)
        image_path = arguments.image_path

    main(image_path, arguments.use_codebook, arguments.adaptive)


# Codebook mode, auto adaptive
# python3 transmitter_pi.py --image_path Datasets/Kodak/kodim23.png --use_codebook

# python3 transmitter_pi.py --image_path Datasets/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/0862.png --use_codebook

# # No codebook, force adaptive off
# python transmitter.py --image_path Datasets/Kodak/kodim23.png --adaptive false

#python3 transmitter_pi.py --use_camera --use_codebook
