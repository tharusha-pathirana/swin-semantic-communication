
from swin_functions import load_single_image, encode_image, decode_image
import torch
import numpy as np


def quantize_feature_nd(feature, codebook_nd_torch, chunk_size):
    """
    Quantize a feature [1, M, 32] by splitting into n-d chunks and finding closest codewords.
    feature: [1, M, 32]
    codebook_nd_torch: [K, chunk_size]
    
    Returns indices of shape [M*(32/chunk_size)] (one index per chunk).
    """
    vectors = feature.squeeze(0)  # [M, 32]
    M = vectors.shape[0]

    # Ensure the feature dimension is divisible by chunk_size
    assert vectors.shape[1] % chunk_size == 0, f"Feature size must be divisible by chunk_size={chunk_size}"
    
    # Reshape to [M*(32/chunk_size), chunk_size]
    num_chunks = vectors.shape[1] // chunk_size
    vectors_nd = vectors.view(M * num_chunks, chunk_size)

    # Compute distances to codebook entries
    distances = torch.cdist(vectors_nd.unsqueeze(0), codebook_nd_torch.unsqueeze(0)).squeeze(0)  # [M*(32/chunk_size), K]

    nearest_indices = torch.argmin(distances, dim=1)  # [M*(32/chunk_size)]
    return nearest_indices, M


def dequantize_feature_nd(indices, M, codebook_nd_torch, chunk_size):
    """
    Reconstruct the feature from indices [M*(32/chunk_size)] using the n-d codebook.
    indices: [M*(32/chunk_size)]
    M: the original M dimension
    codebook_nd_torch: [K, chunk_size]

    Returns [1, M, 32]
    """
    # Reconstruct each n-d vector
    reconstructed_nd = codebook_nd_torch[indices]  # [M*(32/chunk_size), chunk_size]

    # Reshape to [M, (32/chunk_size), chunk_size] then to [M, 32]
    num_chunks = 32 // chunk_size
    reconstructed_vectors = reconstructed_nd.view(M, num_chunks, chunk_size).reshape(M, 32)
    reconstructed_feature = reconstructed_vectors.unsqueeze(0)  # [1, M, 32]
    return reconstructed_feature


def run_inference_with_nd_codebook(net, codebook_path, image_path, config, device, chunk_size):
    """
    Run inference using the n-d codebook.
    """
    # Load the n-d codebook
    codebook_nd = np.load(codebook_path)  # [K, chunk_size]
    codebook_nd_torch = torch.from_numpy(codebook_nd).float().to(device)
    
    net.eval()
    with torch.no_grad():
        # Load and encode image
        input_image, image_name = load_single_image(image_path, config)
        input_image = input_image.to(device)
        feature, chan_snr, channel_rate = encode_image(input_image, net) # [1, M, 32]
        
        # Quantize feature into n-d chunks
        indices, M = quantize_feature_nd(feature, codebook_nd_torch, chunk_size)
        
        # "Transmit" indices (for demonstration, we just keep them here)
        
        # Dequantize at receiver
        reconstructed_feature = dequantize_feature_nd(indices, M, codebook_nd_torch, chunk_size)
        
        # Decode image
        res = (input_image.shape[2], input_image.shape[3])  # (H, W)
        recon_image = decode_image(reconstructed_feature, net, chan_snr, res)
        
        return recon_image, feature, indices, reconstructed_feature


def encode_image_with_nd_codebook(net, codebook_path, image_path, config, device, chunk_size):
    """
    Encode image using the n-d codebook.
    """
    # Load the n-d codebook
    codebook_nd = np.load(codebook_path)  # [K, chunk_size]
    codebook_nd_torch = torch.from_numpy(codebook_nd).float().to(device)

    net.eval()
    with torch.no_grad():
        # Load and encode image
        input_image, image_name = load_single_image(image_path, config)
        input_image = input_image.to(device)
        feature, chan_snr, channel_rate = encode_image(input_image, net) # [1, M, 32]

        # Quantize feature into n-d chunks
        indices, M = quantize_feature_nd(feature, codebook_nd_torch, chunk_size)
        
    return indices, M

def decode_image_with_nd_codebook(indices, M, codebook_path, net, res, device, chunk_size, chan_snr=10):
    """
    Decode image from the n-d codebook.
    """

    codebook_nd = np.load(codebook_path)  # [K, chunk_size]
    codebook_nd_torch = torch.from_numpy(codebook_nd).float().to(device)

    # Dequantize at receiver
    reconstructed_feature = dequantize_feature_nd(indices, M, codebook_nd_torch, chunk_size)

    # Decode image
    with torch.no_grad():
        recon_image = decode_image(reconstructed_feature, net, chan_snr, res)

    return recon_image




