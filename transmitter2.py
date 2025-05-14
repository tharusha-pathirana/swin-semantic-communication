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
import shutil

from datetime import datetime
import torchvision

import torch.optim as optim

import matplotlib.pyplot as plt

from swin_functions import *
from codebook_functions import *
from adaptive_functions import *

import argparse
import warnings

warnings.filterwarnings("ignore")


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
    },
    (4, 1024): {
        "adaptive": 'Codebook/adaptive_patching_codebook_4d_1024clusters_mst.npy',
        "wo_adaptive": './Codebook/codebook_4d_1024clusters_mst.npy'
    },
    (2, 512): {
        "adaptive": 'Codebook/adaptive_patching_codebook_2d_512clusters_mst.npy',
        "wo_adaptive": './Codebook/codebook_2d_512clusters_mst.npy'
    }
}




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
net = SwinJSCC(args, config)
model_path = "./Weights/SwinJSCC_wo_SAandRA_Rayleigh_HRimage_snr3_psnr_C32.model"  
load_weights(model_path)
net = net.to(device)


# Without Codebook

def process_and_encode_image_to_binary(image_path, output_path, adaptive_patch_enabled=False, NORMALIZE_CONSTANT=20, int_size=8, patch_size=28):
    
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



        # Byte 1: adaptive flag (7-bit redundant)
        adaptive_flag_byte = 0b01111111 if adaptive_patch_enabled else 0b00000000

        # Byte 2: use codebook flag (always 0 here)
        use_codebook_flag_byte = 0b00000000

        # Byte 3: chunk size byte (set to 0 or a default, e.g., 2 → 01 → 000 111)
        chunk_byte = int('00000111', 2)  # Represents default "2" as 000 111 (unused for non-codebook)

        # Byte 4: codebook k size byte (set to 0 or a default, e.g., 256 → 000 111)
        k_byte = int('00000111', 2)  # Represents default "256" as 000 111 (unused for non-codebook)

        # Byte 5: patch size flag (based on patch_size)
        patch_flag_byte = 0b01111111 if patch_size == 60 else 0b00000000  # 0 for 28, 1 for 60

        # Combine into byte array
        flag_bytes = np.array([
            adaptive_flag_byte,
            use_codebook_flag_byte,
            chunk_byte,
            k_byte,
            patch_flag_byte
        ], dtype=np.uint8)

        # Save flag bytes + quantized feature
        flag_bytes.tofile(output_path)


        with open(output_path, "ab") as f:
            quantized_feature.tofile(f)
        
        # quantized_feature.tofile(output_path)
        print(f"Encoded feature saved to '{output_path}'")


def prepare_image_path(original_path):

    MAX_DIM = 3500  # Change this value easily if needed

    ext = os.path.splitext(original_path)[-1].lower()
    with Image.open(original_path) as img:
        w, h = img.size
        needs_conversion = ext in [".jpg", ".jpeg", ".dng"]
        needs_resize = max(w, h) > MAX_DIM

        if needs_conversion or needs_resize:
            scale = MAX_DIM / max(w, h) if needs_resize else 1.0
            new_size = (int(w * scale), int(h * scale)) if needs_resize else (w, h)

            if needs_resize:
                img = img.resize(new_size, Image.LANCZOS)

            base_name = os.path.splitext(os.path.basename(original_path))[0]
            temp_path = f"{base_name}.png"

            # temp_path = "image.png"
            img.save(temp_path, format="PNG")
            # print(f"Image processed and saved to {temp_path} with size {new_size}")
            return temp_path
        else:
            return original_path


def main(image_path, use_codebook=False, adaptive=None):

    image_size_kb = os.path.getsize(image_path) / 1024

    input_image, image_name= load_single_image(image_path, config)
    resized_image_np = input_image.squeeze(0).permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    # Convert RGB to BGR for OpenCV compatibility
    image = cv2.cvtColor(resized_image_np, cv2.COLOR_RGB2BGR)

    H_image, W_image = image.shape[:2]
    patch_size = 28

    ref_dim = H_image if H_image >= W_image else W_image

    depth = round(math.log2(ref_dim / 32))
    depth = min(depth, 7)  # Limit depth to a maximum of 7

    if H_image > 3000 or W_image > 3000: 
        # depth = 6 
        low_t,high_t = 100,200
        # patch_size = 60
        v_val = 100
    else : 
        # depth = 5
        low_t,high_t = 100,200
        # patch_size = 28
        v_val = 50


    if arguments.depth is not None:
        depth = arguments.depth
    if arguments.patch_size is not None:
        patch_size = arguments.patch_size

    print("Quadtree depth: ", depth)

    if adaptive is not None and adaptive.lower() == "false":
        adaptive_patch_enabled = False
    else:

        H_new, W_new, data_pixels, L = encode_image_adaptive(image, kernel_size=1,
                                            tl=low_t, th=high_t,  # 100,200
                                            v=v_val,       # quadtree edge threshold
                                            H=depth,        # maximum quadtree depth
                                            Pm=patch_size,      # base patch size before padding  # Pm=patch_size
                                            L=None,     # number of patches in the grid   (If None, it will be set to the number of adaptive patches)
                                            grid_image_file="patches_grid.png", coord_file="patch_coords.bin",
                                            padding=2, visualize=False)
        
    
        if adaptive is None:
            #adaptive_patch_enabled = (H_new * W_new) < (0.7 * H_image * W_image)
            adaptive_patch_enabled = ((data_pixels < (0.8 * H_image * W_image)) or (L < 100)) and ((H_new * W_new) < (H_image * W_image))
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


        # --- Adaptive flag byte ---
        adaptive_flag_byte = 0b01111111 if adaptive_patch_enabled else 0b00000000

        # --- Use codebook flag byte ---
        use_codebook_flag_byte = 0b01111111 if use_codebook else 0b00000000

        if chunk_size == 2:
            chunk_bit1, chunk_bit2 = '0', '1'
        elif chunk_size == 4:
            chunk_bit1, chunk_bit2 = '1', '0'
        elif chunk_size == 8:
            chunk_bit1, chunk_bit2 = '1', '1'
        else:
            raise ValueError(f"Unsupported chunk size: {chunk_size}")
        
        # Build: XX|bit1bit1bit1|bit2bit2bit2  (total 8 bits)
        chunk_byte_str = '00' + (chunk_bit1 * 3) + (chunk_bit2 * 3)
        chunk_byte = int(chunk_byte_str, 2)

        # --- Codebook size (k) encoding ---
        if k == 256:
            k_bit1, k_bit2 = '0', '1'
        elif k == 512:
            k_bit1, k_bit2 = '1', '0'
        elif k == 1024:
            k_bit1, k_bit2 = '1', '1'
        else:
            raise ValueError(f"Unsupported codebook size: {k}")

        k_byte_str = '00' + (k_bit1 * 3) + (k_bit2 * 3)
        k_byte = int(k_byte_str, 2)

        # --- Patch size flag byte ---
        patch_flag_byte = 0b01111111 if patch_size == 60 else 0b00000000 #  0 for 28, 1 for 60

        flag_bytes = np.array([
                    adaptive_flag_byte,
                    use_codebook_flag_byte,
                    chunk_byte,
                    k_byte,
                    patch_flag_byte
                ], dtype=np.uint8)
        
        
        flag_bytes.tofile(bin_path)

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

    label = "adaptive" if adaptive_patch_enabled else "non_adaptive"
    if use_codebook:
        label += f'_{chunk_size}d_{k}k'


    combined_binary_name = f'./Binary/Transmitted_Binary/{image_name.split(".")[0]}_combined_binary_{label}.bin'
    shutil.copyfile("combined_binary.bin", combined_binary_name)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--adaptive", default=None, help="Set 'true' or 'false' to override threshold. Leave empty to use auto mode.")
    parser.add_argument("--k", type=int, default=512, help="Number of clusters in the codebook")
    parser.add_argument("--chunk_size", type=int, default=4, help="Size of vector chunks for quantization")
    parser.add_argument("--patch_size", type=int, choices=[28, 60], default=None,
                        help="Override patch size; choices are 28 or 60")
    parser.add_argument("--depth", type=int, choices=[4, 5, 6, 7], default=None,
                        help="Override quadtree depth; choices are 4, 5, 6 or 7")


    arguments = parser.parse_args()

    k = arguments.k
    chunk_size = arguments.chunk_size

    key = (chunk_size, k)
    codebook_path_adaptive = codebook_paths[key]["adaptive"]
    codebook_path_wo_adaptive = codebook_paths[key]["wo_adaptive"]

    processed_path = prepare_image_path(arguments.image_path)

    main(processed_path, arguments.use_codebook, arguments.adaptive)


# Codebook mode, auto adaptive
# python transmitter.py --image_path Datasets/Kodak/kodim23.png --use_codebook

# # No codebook, force adaptive off
# python transmitter.py --image_path Datasets/Kodak/kodim23.png --adaptive false

# python transmitter_new.py --image_path Datasets/Kodak/kodim23.png --use_codebook

#image_path = "./Datasets/Clic2021/06.png"
#image_path = "./Datasets/Div2K/DIV2K_valid_HR/DIV2K_valid_HR/0862.png"

# python transmitter2.py --image_path Datasets/Wildlife/leopard2.png --use_codebook --depth 5
