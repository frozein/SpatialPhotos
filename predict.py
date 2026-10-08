import argparse
import math
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import pipeline

import exporter

# ------------------------------------------- #
# Hardware configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ------------------------------------------- #

def get_supported_extensions():
    return {".webp", ".png", ".jpg", ".jpeg"}

def generate_spatial_from_depth(image_path, depth_map_np, out_path, block_size=32, focal_y=1000.0, quality=75, opaque_only=True, depth_scale=0.5):
    img = Image.open(image_path).convert("RGBA")
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    width, height = img.size
    
    # Pad to block_size
    pad_w = (block_size - (width % block_size)) % block_size
    pad_h = (block_size - (height % block_size)) % block_size
    
    padded_w = width + pad_w
    padded_h = height + pad_h
    
    atlas = np.zeros((padded_h, padded_w, 4), dtype=np.uint8)
    atlas[:height, :width] = np.array(img)
    atlas[:height, :width, 3] = 255 # solid alpha for main image
    
    grid_w = padded_w // block_size
    grid_h = padded_h // block_size
    
    # Depth map normalization / conversion to metric-like scale
    depth_map_np = np.fliplr(depth_map_np)
    d_min, d_max = depth_map_np.min(), depth_map_np.max()
    if d_max > d_min:
        normalized_depth = (depth_map_np - d_min) / (d_max - d_min)
    else:
        normalized_depth = np.zeros_like(depth_map_np)
        
    # Inverse depth means higher = closer. 
    # Apple spatial format expects Z where higher is further away.
    inverted_depth = 1.0 - normalized_depth
    
    # Map to [zMin, zMax]. Scale the intensity using depth_scale.
    z_min = 1.0
    z_max = 1.0 + (4.0 * depth_scale)
    metric_depth = z_min + inverted_depth * (z_max - z_min)
    
    # Resize depth to grid size + 1
    depth_grid = cv2.resize(metric_depth, (grid_w + 1, grid_h + 1))
    
    # Generate blocks
    blocks = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            blocks.append((0, gx * block_size, gy * block_size, gx * block_size, gy * block_size))
            
    N = len(blocks)
    sliceIdx = np.zeros(N * 4, dtype=np.uint16)
    
    cornerGridX = np.zeros((N, 4), dtype=int)
    cornerGridY = np.zeros((N, 4), dtype=int)
    
    for i, (s_idx, sx, sy, ax, ay) in enumerate(blocks):
        gx = sx // block_size
        gy = sy // block_size
        cornerGridX[i] = [gx, gx + 1, gx + 1, gx]
        cornerGridY[i] = [gy + 1, gy + 1, gy, gy]
        
    depths = np.zeros(N * 4, dtype=np.float32)
    for i in range(N):
        for c in range(4):
            depths[i*4 + c] = depth_grid[cornerGridY[i, c], cornerGridX[i, c]]
            
    sourceX = (cornerGridX * block_size).flatten().astype(np.uint16)
    sourceY = (cornerGridY * block_size).flatten().astype(np.uint16)
    atlasX = sourceX.copy()
    atlasY = sourceY.copy()
    
    vertices = (sliceIdx, depths, sourceX, sourceY, atlasX, atlasY)
    
    exporter.export_spatial(
        atlas=atlas,
        vertices=vertices,
        imageWidth=padded_w,
        imageHeight=padded_h,
        originalWidth=width,
        originalHeight=height,
        focal=focal_y,
        blockSize=block_size,
        outPath=out_path,
        opaqueOnly=opaque_only,
        quality=quality
    )

def predict(
    inputPath: Path,
    outputPath: Path,
    blockSize: int = 32,
    quality: int = 75,
    opaqueOnly: bool = True,
    depthScale: float = 0.5,
) -> None:

    if not inputPath.exists():
        raise FileNotFoundError(f"Input path does not exist: {inputPath}")

    inputIsFile = inputPath.is_file()
    imagePaths = []
    if inputIsFile:
        if inputPath.suffix.lower() in get_supported_extensions():
            imagePaths = [inputPath]
    else:
        if outputPath.is_file() or outputPath.suffix.lower() == ".spatial":
            print(f"Output path must be a directory when input is a directory.")
            return
        for ext in get_supported_extensions():
            imagePaths.extend(list(inputPath.glob(f"**/*{ext}")))

    if not imagePaths:
        print(f"No valid images found in {inputPath}")
        return

    print(f"- Processing {len(imagePaths)} valid image files -")
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=DEVICE)

    singleOutputFile = inputIsFile and outputPath.suffix.lower() == ".spatial"
    outputDirectory = outputPath.parent if singleOutputFile else outputPath
    outputDirectory.mkdir(exist_ok=True, parents=True)

    for imagePath in imagePaths:
        print(f"\nProcessing {imagePath}...")
        img = Image.open(imagePath).convert("RGB")
        
        # Depth Anythng V2 Inference
        print("Predicting depth map...")
        result = pipe(img)
        depth_np = np.array(result["depth"]).astype(np.float32)
        
        outFile = outputPath if singleOutputFile else outputPath / f"{imagePath.stem}.spatial"
        print(f"Generating Spatial Photo -> {outFile}")
        
        generate_spatial_from_depth(
            image_path=imagePath,
            depth_map_np=depth_np,
            out_path=outFile,
            block_size=blockSize,
            focal_y=max(img.height, img.width),
            quality=quality,
            opaque_only=opaqueOnly,
            depth_scale=depthScale,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3D representation of a photo using Depth Anything V2.")
    parser.add_argument("input", type=Path, help="Input image or image directory")
    parser.add_argument("output", type=Path, help="Output .spatial file or directory")
    parser.add_argument("--quality", type=int, default=75, help="JPEG encoding quality")
    parser.add_argument("--block-size", type=int, default=32, help="Block size in pixels, multiple of 8.")
    parser.add_argument("--opaque-only", action="store_true", default=True, help="Use opaque rendering (smaller files).")
    parser.add_argument("--depth-scale", type=float, default=0.5, help="Intensity of the 3D depth effect (default 0.5)")
    
    args = parser.parse_args()
    
    predict(
        args.input, args.output,
        blockSize=args.block_size,
        quality=args.quality,
        opaqueOnly=args.opaque_only,
        depthScale=args.depth_scale,
    )
