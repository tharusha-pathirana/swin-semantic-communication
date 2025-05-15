import warnings

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
import shutil

from datetime import datetime
import torchvision

import torch.optim as optim

import matplotlib.pyplot as plt

from swin_functions import *
from codebook_functions import *
from adaptive_functions import *

import argparse

from pytorch_msssim import ms_ssim as msssim
from lpips import LPIPS
import pandas as pd
from PIL import ImageFile

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
        self.channel_type = 'awgn'  # Choices: ['awgn', 'rayleigh']
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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

def separate_binary_file(combined_file, output_file1, output_file2):
    # input_path = os.path.join("/kaggle/working", combined_file)  # Adjust input path if needed
    # output_path1 = os.path.join("/kaggle/working", output_file1)
    # output_path2 = os.path.join("/kaggle/working", output_file2)   
    with open(combined_file, "rb") as f:
        # Read the first 4 bytes to extract metadata (size of first file)
        size1 = struct.unpack("I", f.read(4))[0]  # Unpack the stored size   
        # Read actual binary data
        data1 = f.read(size1)  # Read first file's data
        data2 = f.read()       # Read remaining data (second file)
    with open(output_file1, "wb") as f1:
        f1.write(data1)
    with open(output_file2, "wb") as f2:
        f2.write(data2)
    print(f"File {combined_file} split into {output_file1} and {output_file2}.")

seed_torch()
logger = logger_configuration(config, save_log=False)
logger.disabled = True
logger.info(config.__dict__)
torch.manual_seed(seed=config.seed)
net = SwinJSCC(args, config)
model_path = "./Weights/SwinJSCC_wo_SAandRA_Rayleigh_HRimage_snr3_psnr_C32.model"  
load_weights(model_path)
net = net.to(device)

def save_image(image, filename, base_path="./recon/"):
    os.makedirs(base_path, exist_ok=True)
    torchvision.utils.save_image(image, os.path.join(base_path, filename))

def calculate_ms_ssim(image_path, output_image_path, resolution):
    """
    Calculate the MS-SSIM between two images.

    Parameters:
        image_path (str): Path to the first image (original image).
        output_image_path (str): Path to the second image (output image).
        resolution (int): Desired resolution for cropping the images.

    Returns:
        float: MS-SSIM value between the two images.
    """
    # Image Preprocessing
    preprocess = transforms.Compose([
        transforms.CenterCrop((resolution)),  # Resize images to a consistent size
        transforms.ToTensor(),          # Convert image to tensor
    ])

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Load two images
    img1 = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)  # Add batch dimension
    img2 = preprocess(Image.open(output_image_path).convert('RGB')).unsqueeze(0)  # Add batch dimension

    # Ensure images are scaled between [0, 1]
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    # Compute MS-SSIM
    ms_ssim_score = msssim(img1, img2, data_range=1.0, size_average=True)  # data_range is 1.0 for [0, 1] images

    return ms_ssim_score.item()

def calculate_lpips(image_path, output_image_path, resolution):
    """
    Calculate the MS-SSIM between two images.

    Parameters:
        image_path (str): Path to the first image (original image).
        output_image_path (str): Path to the second image (output image).
        resolution (int): Desired resolution for cropping the images.

    Returns:
        float: MS-SSIM value between the two images.
    """
    # Initialize LPIPS metric
    lpips_metric = LPIPS(net='alex')

    # Load and preprocess images
    transform = transforms.Compose([
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])

    # Load images
    image1 = Image.open(image_path).convert("RGB")  # Ensure images are RGB
    image2 = Image.open(output_image_path).convert("RGB")

    # Apply transformations
    image1_tensor = transform(image1).unsqueeze(0)  # Add batch dimension
    image2_tensor = transform(image2).unsqueeze(0)  # Add batch dimension

    similarity_score = lpips_metric(image1_tensor, image2_tensor)


    return similarity_score.item()

def get_file_size(file_path):
    """
    Returns the size of the file at the given file_path in bytes.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        int: The size of the file in bytes, or None if the file doesn't exist or is inaccessible.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If there is an issue accessing the file.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' does not exist or is not a file.")
            return None
        
        # Get the file size in bytes
        file_size = os.path.getsize(file_path)
        return file_size
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except OSError as e:
        print(f"Error accessing file '{file_path}': {e}")
        return None
    
import pickle

# Load codeword list
with open("index_to_codeword.pkl", "rb") as f:
    index_to_codeword = pickle.load(f)

codeword_to_index = {cw: idx for idx, cw in enumerate(index_to_codeword)}


def hamming_distance(a, b):
    return bin(a ^ b).count('1')


def decode_value(received_val):
    if received_val in codeword_to_index:
        return codeword_to_index[received_val]
    
    # Fallback to nearest match
    min_dist = 17
    best_idx = None
    for idx, codeword in enumerate(index_to_codeword):
        dist = hamming_distance(received_val, codeword)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
            if dist == 0:
                break
    return best_idx


def process_and_encode_image_to_binary_sim(image_path, output_path, adaptive_patch_enabled=False, NORMALIZE_CONSTANT=20, int_size=8, snr_val = 12):
    
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

        feature = pass_through_channel(feature, net, snr_val)
        print(f"Feature passed through channel SNR: {snr_val} dB")

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

    MAX_DIM = 3000  # Change this value easily if needed

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


def simulate_awgn_on_file(in_path: str, out_path: str, snr_db: float):
    """
    Reads in a binary file, treats each bit as a BPSK symbol (0→–1, 1→+1),
    adds AWGN at the specified Eb/N0, then re-demodulates and writes a new file.
    """
    # load bytes and unpack to bits
    data = np.fromfile(in_path, dtype=np.uint8)
    bits = np.unpackbits(data)

    # map to BPSK symbols
    symbols = 2*bits.astype(np.float32) - 1.0

    # noise standard deviation for Eb/N0 = snr_db
    # assume Eb = 1, so N0/2 = 1/(2 * 10^(snr_db/10))
    sigma = np.sqrt(1.0 / (2 * (10**(snr_db/10.0))))
    noise = np.random.normal(0, sigma, size=symbols.shape)

    # add noise and hard-decide
    received = symbols + noise
    bits_rx = (received > 0).astype(np.uint8)

    # pack back into bytes and write
    data_rx = np.packbits(bits_rx)
    data_rx.tofile(out_path)
    print(f"Noisy data written to {out_path} (Eb/N0={snr_db} dB)")


def transmit_code(image_path, use_codebook=False, adaptive=None, noise=10.0, patch_size=28, low_th=100, high_th=200, v_val=50, kernel=1):

    image_size_kb = os.path.getsize(image_path) / 1024

    input_image, image_name= load_single_image(image_path, config)
    resized_image_np = input_image.squeeze(0).permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    resized_image_np = (resized_image_np * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    # Convert RGB to BGR for OpenCV compatibility
    image = cv2.cvtColor(resized_image_np, cv2.COLOR_RGB2BGR)

    H_image, W_image = image.shape[:2]

    ref_dim = H_image if H_image >= W_image else W_image


    depth = round(math.log2(ref_dim / 32))
    depth = min(depth, 7)  # Limit depth to a maximum of 7

    if arguments.depth is not None:
        depth  = arguments.depth


    if adaptive is not None and adaptive.lower() == "false":
        adaptive_patch_enabled = False
    else:

        H_new, W_new, data_pixels, L = encode_image_adaptive(image, kernel_size=kernel,
                                            tl=low_th, th=high_th,
                                            v=v_val,       # quadtree edge threshold
                                            H=depth,        # maximum quadtree depth
                                            Pm=patch_size,      # base patch size before padding
                                            L=None,     # number of patches in the grid   (If None, it will be set to the number of adaptive patches)
                                            grid_image_file="patches_grid.png", coord_file="patch_coords.bin",
                                            padding=2, visualize=False)
        
    
        if adaptive is None:
            #adaptive_patch_enabled = (H_new * W_new) < (0.7 * H_image * W_image)
            adaptive_patch_enabled = ((data_pixels < (0.8 * H_image * W_image)) or (L < 100)) and (H_new * W_new) < (H_image * W_image)
               
        else:
            adaptive_patch_enabled = adaptive.lower() == "true"

    print(f"Original Resolution: {H_image} x {W_image}")
    print(f"Adaptive Patch enabled : {adaptive_patch_enabled}")

    if adaptive_patch_enabled:
        image_path = 'patches_grid.png'
        adaptive_string = "adaptive"
    else:
        adaptive_string = "wo_adaptive"

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

        bin_path = "./Binary/indices.bin"
        indices, _ = encode_image_with_nd_codebook(net, codebook_path, image_path, config, device, chunk_size)
        indices_np = indices.cpu().numpy()

        if k==512:
            indices_np = np.array([index_to_codeword[int(i)] for i in indices_np], dtype=np.uint16)

        

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
        # combine_binary_files("patch_coords.bin", bin_path, "combined_binary.bin")

        simulate_awgn_on_file(bin_path, './Binary/noisy.bin', noise)
        combine_binary_files("patch_coords.bin", './Binary/noisy.bin', "./Binary/simulated.bin")

    else:
        print("Encoding without codebook...")

        if not adaptive_patch_enabled:
            resolution_data = f"Resolution: {H_image}x{W_image}".encode('utf-8')
            # Open the binary file in write-binary mode and overwrite it
            binary_file_path = "patch_coords.bin"  # Replace with your binary file path
            with open(binary_file_path, 'wb') as f:
                f.write(resolution_data)
            print(f"Binary file overwritten with resolution: {H_image}x{W_image}")
        

        output_path = "./Binary/encoded_feature.bin"
        process_and_encode_image_to_binary_sim(image_path, output_path, adaptive_patch_enabled, NORMALIZE_CONSTANT, int_size, int(noise))
        #combine_binary_files("patch_coords.bin", output_path, "combined_binary.bin")
        combine_binary_files("patch_coords.bin", output_path, "./Binary/simulated.bin")
        bin_file_path = output_path

        # simulate_awgn_on_file(output_path, './Binary/noisy.bin', noise)
        # combine_binary_files("patch_coords.bin", './Binary/noisy.bin', f"./Binary/Received_Binary/combined_binary_{dataset}_{img_no}_noisy.bin")



    
    # Print image and file size
    
    bin_size_kb = os.path.getsize(bin_file_path) / 1024
    print(f"Original image size: {image_size_kb:.2f} KB")
    print(f"Binary file size for transmission: {bin_size_kb:.2f} KB")

    compression_ratio = image_size_kb / bin_size_kb if bin_size_kb > 0 else 0
    print(f"Compression Ratio (Original / Transmitted): {compression_ratio:.2f}")


save_image_folder = './recon/'

def majority(bits):
    return int(sum(bits) >= (len(bits) / 2))

def decode_redundant_byte(byte_val, method="7bit"):
    if method == "7bit":
        bits = [(byte_val >> i) & 1 for i in range(7)]
        return majority(bits)
    elif method == "2bit_3x":
        bit1_group = [(byte_val >> i) & 1 for i in range(5, 2, -1)]  # bits 5,4,3
        bit2_group = [(byte_val >> i) & 1 for i in range(2, -1, -1)]  # bits 2,1,0
        bit1 = majority(bit1_group)
        bit2 = majority(bit2_group)
        return f"{bit1}{bit2}"
    else:
        raise ValueError("Unknown decode method")
    

def decode_and_evaluate(received_binary_path, image_path=None, resolution = (512,768), NORMALIZE_CONSTANT = 20, int_size=8, adaptive=False):           # Image path is original image

    tensor_shape = (1, int((resolution[0]*resolution[1])/(16*16)), 32)          # (Image size is 512x768x3 --> tensor (1, H/16 * W/16, 32))

    # if int_size == 8: 
    #     received_feature = np.fromfile(received_binary_path, dtype=np.int8).reshape(tensor_shape)
    #     recovered_feature = received_feature / 127 * NORMALIZE_CONSTANT
    
    # elif int_size == 16:
    #     received_feature = np.fromfile(received_binary_path, dtype=np.int16).reshape(tensor_shape)
    #     recovered_feature = received_feature / 32767 * NORMALIZE_CONSTANT

    with open(received_binary_path, "rb") as f:
        _ = f.read(5)  # Skip adaptive flag byte

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
            #print(f"PSNR: {round(psnr_value.item(), 3)} dB")
            # print(f'CBR = {cbr_used}')

        else:
            plot_single_image(recon_image)

    else:
        save_image(recon_image, f"reconstructed_patches_grid.png")

        reconstructed_image = decode_image_adaptive(grid_image_file="./recon/reconstructed_patches_grid.png",
                                                    coord_file="patch_coord_received.bin",Pm=patch_size, padding=2)
        reconstructed_image = reconstructed_image.clamp(0,1)
        

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, reconstructed_image.to(device))
            plot_images(input_image, reconstructed_image, image_name)
            #print(f"PSNR: {psnr_value:.2f} dB")
        else: 
            plot_single_image(reconstructed_image)


    #save_image(reconstructed_image if adaptive else recon_image, f"reconstructed_{image_name if image_path else 'default_image.png'}")

    #save_image(recon_image, f"reconstructed_{image_name}")

    output_image_base_path = f"./recon/without_codebook/adaptive={adaptive}/"
    save_image(reconstructed_image if adaptive else recon_image, 'simulated_image.png')
    output_image_path = os.path.join('./recon/', 'simulated_image.png')
    print('Reconstructed image saved at: ',output_image_path)
    if image_path is not None:
        ms_ssim_value = calculate_ms_ssim(image_path, output_image_path, resolution)
        lpips_value = calculate_lpips(image_path, output_image_path, resolution)
        input_image_size = get_file_size(image_path)/1024
        output_image_size = get_file_size(output_image_path)/1024
        reconstructed_binary_size = get_file_size(received_binary_path)/1024
        Compression_Ratio = input_image_size/reconstructed_binary_size
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"MS-SSIM: {ms_ssim_value:.4f}")
        print(f"LPIPS: {lpips_value:.4f}") 
        print(f"Input Image Size: {input_image_size:.2f} KB")
        print(f"Output Image Size: {output_image_size:.2f} KB")
        print(f"Reconstructed Binary Size: {reconstructed_binary_size:.2f} KB")
        print(f"Compression Ratio: {Compression_Ratio:.2f}")
        # append_to_excel(image_name, psnr_value.item(), ms_ssim_value, lpips_value,input_image_size, output_image_size, reconstructed_binary_size, Compression_Ratio, excel_file=f"results_without_codebook_{'Adaptive = true'if adaptive else 'Adaptive = false'}.xlsx")
    #save_image(recon_image, f"reconstructed_{image_name}")


def decode_indices_and_plot (received_binary_path , codebook_path, image_path=None ,chunk_size=4 ,resolution = (512,768), k=512, adaptive=False):

    # if k > 256: loaded_indices = np.fromfile(received_binary_path, dtype=np.uint16)   # Use the same dtype as during saving
    # else: loaded_indices = np.fromfile(received_binary_path, dtype=np.uint8)

    with open(received_binary_path, "rb") as f:
        _ = f.read(5)  # Skip the first byte (flag already read outside)

        if k > 256:
            loaded_indices = np.frombuffer(f.read(), dtype=np.uint16)
        else:
            loaded_indices = np.frombuffer(f.read(), dtype=np.uint8)

    if k==512:
        loaded_indices = np.array([decode_value(int(v)) for v in loaded_indices], dtype=np.uint16)


    
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
    output_image_base_path = f"./recon/{chunk_size}d_{k}k/adaptive={adaptive}/"
    grid_image_base_path = "./recon/"

    if image_path is not None:

        input_image, image_name = load_single_image(image_path, config)
        input_image = input_image.to(device)

    reconstructed = decode_image_with_nd_codebook(valid_indices, M, codebook_path, net, resolution, device, chunk_size)
    
    recon_image = reconstructed.clamp(0, 1)

    if adaptive == False:

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, recon_image)
            plot_images(input_image, recon_image, image_name)
            #print(f"PSNR: {round(psnr_value.item(), 3)} dB")
            # print('CBR = 1/48')
            # print(f'{chunk_size}d vectors in codebook')

        else:
            plot_single_image(recon_image)

    else:
        save_image(recon_image, f"reconstructed_patches_grid.png", grid_image_base_path)
        reconstructed_image = decode_image_adaptive(grid_image_file="./recon/reconstructed_patches_grid.png",
                                                    coord_file="patch_coord_received.bin",Pm=patch_size, padding=2)
        reconstructed_image = reconstructed_image.clamp(0,1)

        if image_path is not None:
            psnr_value = calculate_psnr(input_image, reconstructed_image.to(device))
            plot_images(input_image, reconstructed_image, image_name)
            #print(f"PSNR: {psnr_value:.2f} dB")
        else:
            plot_single_image(reconstructed_image)

    #save_image(reconstructed_image if adaptive else recon_image, f"reconstructed_{image_name if image_path else 'default_image.png'}")

    output_image_base_path = f"./recon/{chunk_size}d_{k}k/adaptive={adaptive}/"
    save_image(reconstructed_image if adaptive else recon_image, 'simulated_image.png')
    output_image_path = os.path.join('./recon/', 'simulated_image.png')
    print('Reconstructed image saved at: ',output_image_path)
    if image_path is not None:
        ms_ssim_value = calculate_ms_ssim(image_path, output_image_path, resolution)
        lpips_value = calculate_lpips(image_path, output_image_path, resolution)
        input_image_size = get_file_size(image_path)/1024
        output_image_size = get_file_size(output_image_path)/1024
        reconstructed_binary_size = get_file_size(received_binary_path)/1024
        Compression_Ratio = input_image_size/reconstructed_binary_size
        print(f"PSNR: {round(psnr_value.item(), 3)} dB")
        print(f"MS-SSIM: {ms_ssim_value:.4f}")
        print(f"LPIPS: {lpips_value:.4f}") 
        print(f"Input Image Size: {input_image_size:.2f} KB")
        print(f"Output Image Size: {output_image_size:.2f} KB")
        print(f"Reconstructed Binary Size: {reconstructed_binary_size:.2f} KB")
        print(f"Compression Ratio: {Compression_Ratio:.2f}")
        # append_to_excel(image_name, psnr_value.item(), ms_ssim_value, lpips_value, input_image_size, output_image_size, reconstructed_binary_size, Compression_Ratio, excel_file=f"results_{chunk_size}d_{k}k_{'Adaptive = true'if adaptive else 'Adaptive = false'}.xlsx")


    #save_image(recon_image, f"reconstructed_{image_name}")


def receive_code(received_filename, image_path=None, use_codebook=False, resolution_args=(None, None), adaptive_override=None):
    global k, chunk_size, adaptive_patch_enabled, patch_size, codebook_enabled, key, codebook_path_adaptive, codebook_path_wo_adaptive
    separate_binary_file(received_filename, "patch_coord_received.bin", "image_data_received.bin")
    received_bin_file = 'image_data_received.bin'

    # with open(received_bin_file, "rb") as f:
    #     flag_byte = f.read(1)
    # flag_int = int.from_bytes(flag_byte, byteorder='little')
    # bits = [(flag_int >> i) & 1 for i in range(7)]
    # adaptive_patch_enabled = sum(bits) >= 4

    # --- Read 5 control bytes ---
    with open(received_bin_file, "rb") as f:
        flag_bytes = f.read(5)

    adaptive_patch_enabled = bool(decode_redundant_byte(flag_bytes[0], "7bit"))
    codebook_enabled = bool(decode_redundant_byte(flag_bytes[1], "7bit"))
    chunk_bits = decode_redundant_byte(flag_bytes[2], "2bit_3x")
    k_bits = decode_redundant_byte(flag_bytes[3], "2bit_3x")
    patch_size_flag = bool(decode_redundant_byte(flag_bytes[4], "7bit"))

    # --- Resolve final chunk_size ---
    
    if chunk_bits == "01":
        chunk_size = 2
    elif chunk_bits == "10":
        chunk_size = 4
    elif chunk_bits == "11":
        chunk_size = 8
    else:
        chunk_size = 4 # Default value if decoding fails

    # --- Resolve final k ---
 
    if k_bits == "01":
        k = 256
    elif k_bits == "10":
        k = 512
    elif k_bits == "11":
        k = 1024
    else:
        k = 512 # Default value if decoding fails

    use_codebook = codebook_enabled

    patch_size = 60 if patch_size_flag else 28




    if adaptive_override is not None:
        adaptive_patch_enabled = adaptive_override.lower() == "true"

    
    key = (chunk_size, k)
    codebook_path_adaptive = codebook_paths[key]["adaptive"]
    codebook_path_wo_adaptive = codebook_paths[key]["wo_adaptive"]

    

    #print(f"Adaptive Patching Enabled: {adaptive_patch_enabled}")
    print(f"Codebook Enabled: {use_codebook}")
    if use_codebook:
        print(f"Chunk Size: {chunk_size}")
        print(f"Codebook k Size: {k}")
    

    print(f"Adaptive Patching Enabled: {adaptive_patch_enabled}")
    if adaptive_patch_enabled:
        print(f"Patch Size: {patch_size}")

    if adaptive_patch_enabled:
        codebook_path_final = codebook_path_adaptive
        #print(f"Using codebook: {codebook_path_final}")

        binary_file_path = "patch_coord_received.bin"
        with open(binary_file_path, "rb") as f:
            # Read grid size (rows, cols)
            grid_size = np.fromfile(f, dtype=np.uint16, count=2)
            rows, cols = grid_size[0], grid_size[1]

            effective_patch_size = patch_size + 2 * 2
            print(f"Resolution read from file: {rows * effective_patch_size}x{cols * effective_patch_size}")

        resolution = (rows * effective_patch_size, cols * effective_patch_size)

        #     print(f"Resolution read from file: {rows*32}x{cols*32}")
        # resolution = (rows*32, cols*32)


    
    else:
        codebook_path_final = codebook_path_wo_adaptive
        #print(f"Using codebook: {codebook_path_final}")

        binary_file_path = "patch_coord_received.bin"  # Replace with your binary file path
        with open(binary_file_path, 'rb') as f:
            data = f.read()
        resolution_str = data.decode('utf-8')
        if resolution_str.startswith("Resolution: "):
            resolution_part = resolution_str[len("Resolution: "):]  # Remove the prefix
            height, width = map(int, resolution_part.split('x'))  # Split and convert to integers
            print(f"Resolution read from file: {height}x{width}")
        resolution = (height,width)

    if resolution_args[0] is not None and resolution_args[1] is not None:
        resolution = (resolution_args[0], resolution_args[1])

    if use_codebook:
        print(f"Codebook path: {codebook_path_final}")
        decode_indices_and_plot(received_bin_file, codebook_path_final, image_path, chunk_size, resolution, k, adaptive_patch_enabled)
    else:
        decode_and_evaluate(received_bin_file, image_path, resolution, NORMALIZE_CONSTANT, int_size, adaptive_patch_enabled)















# Arg parse if transmit or receive or both

if __name__ == "__main__":
    #print("In main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default=None, choices=["tx", "rx", "both"])
    parser.add_argument("--received_file", default="./Binary/simulated.bin")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--use_codebook", action="store_true")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--adaptive", default=None, choices=["true", "false"],
                        help="Override adaptive patching detection. Use 'true' or 'false'.")
    parser.add_argument("--patch_size", type=int, choices=[28, 60], default=28,
                    help="Override patch size; choices are 28 or 60")
    parser.add_argument("--noise", type=float, default=10.0,
                    help="AWGN noise level in dB")
    parser.add_argument("--low_th", type=int, default=100,
                    help="Low threshold for adaptive patching")
    parser.add_argument("--high_th", type=int, default=200,
                    help="High threshold for adaptive patching")
    parser.add_argument("--v_val", type=int, default=100,
                    help="V value for adaptive patching")
    parser.add_argument("--kernel", type=int, default=1,
                    help="Kernel size for adaptive patching")
    parser.add_argument("--depth", type=int, choices=[4, 5, 6, 7], default=None,
                        help="Override quadtree depth; choices are 4, 5, 6 or 7")
    
    
    
    arguments = parser.parse_args()

    low_th = arguments.low_th
    high_th = arguments.high_th
    v_val = arguments.v_val
    kernel = arguments.kernel
    patch_size = arguments.patch_size
    k = arguments.k
    chunk_size = arguments.chunk_size

    # --- Set codebook paths based on chunk_size and k ---
    key = (chunk_size, k)
    codebook_path_adaptive = codebook_paths[key]["adaptive"]
    codebook_path_wo_adaptive = codebook_paths[key]["wo_adaptive"]


    if arguments.type == "tx":
        # Transmit
        if arguments.image_path is None:
            print("Error: --image_path is required for transmission.")
        else:
            processed_path = prepare_image_path(arguments.image_path)
            transmit_code(processed_path, use_codebook=arguments.use_codebook ,adaptive=arguments.adaptive,
                        noise=arguments.noise, patch_size=arguments.patch_size, low_th=low_th, high_th=high_th,
                        v_val=v_val, kernel=kernel)
    elif arguments.type == "rx":
        # Receive
        processed_path = prepare_image_path(arguments.image_path) if arguments.image_path else None
        receive_code(arguments.received_file, image_path=processed_path)

    elif arguments.type == "both":
        if arguments.image_path is None:
            print("Error: --image_path is required for transmission.")
        else:
            processed_path = prepare_image_path(arguments.image_path)
            transmit_code(processed_path, use_codebook=arguments.use_codebook, adaptive=arguments.adaptive,
                        noise=arguments.noise, patch_size=arguments.patch_size, low_th=low_th, high_th=high_th,
                        v_val=v_val, kernel=kernel)
            receive_code(arguments.received_file, image_path=processed_path)

    else:
        print("Error: --type must be 'tx', 'rx', or 'both'.")

