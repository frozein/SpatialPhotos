import os
import io
import math
import ddgs
import numpy as np
import torch
import base64
import rectpack
import imageio.v2 as imageio

from PIL import Image
from tqdm import tqdm
from plyfile import PlyData

from renderer import render_headless
from exporter import export_glb

# ------------------------------------------- #

OUTFILL_AMOUNT = 0.0

DEPTH_MIN_QUANTILE = 0.0
DEPTH_MAX_QUANTILE = 0.8
NUM_SLICES = 30

BLOCK_SIZE = 64

ALPHA_REPLACE_THRESHOLD = 0.1
ALPHA_TEST_THRESHOLD = 0.5

DEPTH_INFILL_CUTOFF = 0.1
DEPTH_INFILL_OUTLIER_STD = 2.0

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

UV_PADDING = 0.0

IPD = 0.064

# ------------------------------------------- #

def look_at(eye, target, up):
	f = (target - eye)
	f = f / torch.norm(f)
	u = up / torch.norm(up)
	s = torch.cross(f, u, dim=0)
	s = s / torch.norm(s)
	u = torch.cross(s, f, dim=0)
	
	m = torch.eye(4, dtype=torch.float32)
	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)

	return m

def perspective(fovy, aspect, znear, zfar):
	tan_half_fovy = math.tan(fovy / 2)

	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0

	return m

def load_ply(path, device='cuda'):
	data = PlyData.read(path)
	vertex = data['vertex'].data

	def np_to_torch(name, dim=1):
		arr = np.stack([vertex[n] for n in name], axis=-1) if isinstance(name, (list, tuple)) else vertex[name]
		return torch.tensor(arr, dtype=torch.float32, device=device)

	means = np_to_torch(['x', 'y', 'z'])
	colors = 0.5 + np_to_torch(['f_dc_0', 'f_dc_1', 'f_dc_2']) * 0.28209479177387814 # convert to SH
	opacities = torch.sigmoid(np_to_torch('opacity').unsqueeze(1))
	scales = torch.exp(np_to_torch(['scale_0', 'scale_1', 'scale_2']))
	rotations = np_to_torch(['rot_1', 'rot_2', 'rot_3', 'rot_0'])

	numGaussians = means.shape[0]

	colors = colors.reshape((numGaussians, 1, 3)) # reshape to match expected format

	gaussians = (means, scales, rotations, opacities, colors)
	focalY = data['intrinsic'].data['intrinsic'][0]

	return gaussians, focalY

# ------------------------------------------- #

def slice_t(numSlices, idx):
	return (idx / numSlices) * (idx / numSlices)

def get_slice(gaussians, zMin, zMax, numSlices, idx, includeBehind = False):
	means, scales, rotations, opacities, colors = gaussians

	tMin = slice_t(numSlices, idx)
	tMax = slice_t(numSlices, idx + 1)

	zMinSlice = zMin + tMin * (zMax - zMin)
	zMaxSlice = zMin + tMax * (zMax - zMin)
	
	if (idx == numSlices - 1) or includeBehind:
		where = means[:, 2] >= zMinSlice
	elif idx == 0:
		where = means[:, 2] < zMaxSlice
	else:
		where = (means[:, 2] >= zMinSlice) & (means[:, 2] < zMaxSlice)

	return means[where], scales[where], rotations[where], opacities[where], colors[where]

# ------------------------------------------- #

def extract_blocks(img, blockSize):
	h, w, _ = img.shape
	gw, gh = w // blockSize, h // blockSize

	present = np.zeros((gh, gw), dtype=bool)
	blocks = {}

	for gy in range(gh):
		for gx in range(gw):
			y = gy * blockSize
			x = gx * blockSize
			block = img[y:y+blockSize, x:x+blockSize]

			alpha = block[..., 3]
			if np.any(alpha >= ALPHA_TEST_THRESHOLD * 255.0):
				present[gy, gx] = True
				blocks[(gx, gy)] = block

	return present, blocks

def maximal_rectangles(mask):
	h, w = mask.shape
	heights = np.zeros(w, dtype=int)
	rects = []

	for y in range(h):
		for x in range(w):
			heights[x] = heights[x] + 1 if mask[y, x] else 0

		stack = []
		x = 0
		while x <= w:
			cur = heights[x] if x < w else 0
			if not stack or cur >= heights[stack[-1]]:
				stack.append(x)
				x += 1
			else:
				top = stack.pop()
				width = x if not stack else x - stack[-1] - 1
				height = heights[top]
				if width > 0 and height > 0:
					rects.append((x - width, y - height + 1, width, height))

	return rects

def greedy_mesh(mask):
	mask = mask.copy()
	rectsOut = []

	while np.any(mask):
		rects = maximal_rectangles(mask)
		x, y, w, h = max(rects, key=lambda r: r[2] * r[3])

		rectsOut.append((x, y, w, h))
		
		mask[y:y+h, x:x+w] = False

	return rectsOut

def build_merged_blocks(blocks, rects, blockSize):
	merged = []

	for gx, gy, gw, gh in rects:
		px = gx * blockSize
		py = gy * blockSize
		pw = gw * blockSize
		ph = gh * blockSize

		img = np.zeros((ph, pw, 4), dtype=np.uint8)
		blockCoords = []

		for dy in range(gh):
			for dx in range(gw):
				block = blocks[(gx + dx, gy + dy)]
				img[
					dy*blockSize:(dy+1)*blockSize,
					dx*blockSize:(dx+1)*blockSize
				] = block

				blockCoords.append((gx + dx, gy + dy))

		merged.append((px, py, pw, ph, img, blockCoords))

	return merged

def pack_blocks(mergedBlocks):

	# binary search to find best size:
	# ---------------
	def fits(size):
		packer = rectpack.newPacker(rotation=False)
		for i, (w, h, _, _) in enumerate(mergedBlocks):
			packer.add_rect(w, h, i)

		packer.add_bin(size, size)
		packer.pack()

		return len(packer.rect_list()) == len(mergedBlocks)

	low, high = ATLAS_MIN_SIZE, ATLAS_MAX_SIZE
	bestSize = high

	while (high - low) > ATLAS_MIN_SIZE:
		mid = (low + high) // 2
		if fits(mid):
			bestSize = mid
			high = mid - 1
		else:
			low = mid + 1

	# pack using best size:
	# ---------------
	finalPacker = rectpack.newPacker(rotation=False)
	for i, (w, h, _, _) in enumerate(mergedBlocks):
		finalPacker.add_rect(w, h, i)
	
	finalPacker.add_bin(bestSize, bestSize)
	finalPacker.pack()

	return finalPacker

def generate_block_atlas(slices, blockSize):

	mergedBlocks = []
	placementsMeta = []

	# greedy mesh each slice:
	# ---------------
	for idx, (img, depth) in enumerate(tqdm(slices, desc="Greedy meshing slices", unit="slice")):
		mask, blocks = extract_blocks(img, blockSize)
		if not np.any(mask):
			continue

		rects = greedy_mesh(mask)
		merged = build_merged_blocks(blocks, rects, blockSize)

		for (px, py, pw, ph, imgBlock, asdf) in merged:
			mergedBlocks.append((pw, ph, imgBlock, asdf))
			placementsMeta.append((idx, px, py, pw, ph))

	# pack greedy meshed rects into atlas:
	# ---------------
	print("Packing slices into atlas... ", end='', flush=True)

	packer = pack_blocks(mergedBlocks)

	bin0 = packer.bin_list()[0]
	atlasWidth, atlasHeight = bin0
	atlas = np.zeros((atlasHeight, atlasWidth, 4), dtype=np.uint8)

	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		_, _, img, blockCoords = mergedBlocks[i]
		idx, _, _, _, _ = placementsMeta[i]

		atlas[ay:ay+ah, ax:ax+aw] = img

		for gx, gy in blockCoords:
			bx = gx * blockSize
			by = gy * blockSize

			ox = (bx - placementsMeta[i][1])
			oy = (by - placementsMeta[i][2])

			u0 = (ax + ox + UV_PADDING) / atlasWidth
			v0 = (ay + oy + UV_PADDING) / atlasHeight
			u1 = (ax + ox + blockSize - UV_PADDING) / atlasWidth
			v1 = (ay + oy + blockSize - UV_PADDING) / atlasHeight

			placements.append((idx, bx, by, u0, v1, u1, v0))

	atlas[...][atlas[..., 3] < ALPHA_TEST_THRESHOLD * 255] = [0, 0, 0, 0]
	atlas[..., 3][atlas[..., 3] >= ALPHA_TEST_THRESHOLD * 255] = 255

	print('done')

	return atlas, placements

def generate_block_atlas_fast(slices, blockSize):

	allBlocks = []
	placementsMeta = []

	# collect every individual block:
	# ---------------
	for idx, (img, depth) in enumerate(tqdm(slices, desc="Extracting blocks", unit="slice")):
		mask, blocks = extract_blocks(img, blockSize)
		if not np.any(mask):
			continue

		for (gx, gy), block in blocks.items():
			if not mask[gy, gx]:
				continue

			px = gx * blockSize
			py = gy * blockSize

			allBlocks.append((blockSize, blockSize, block, [(gx, gy)]))
			placementsMeta.append((idx, px, py, blockSize, blockSize))

	# find smallest square atlas that fits all blocks:
	# ---------------
	print("Packing blocks into atlas... ", end='', flush=True)

	n = len(allBlocks)
	blocksPerSide = math.ceil(math.sqrt(n))
	atlasSize = blocksPerSide * blockSize

	atlasWidth = atlasSize
	atlasHeight = atlasSize
	atlas = np.zeros((atlasHeight, atlasWidth, 4), dtype=np.uint8)

	placements = []

	for i, (bw, bh, imgBlock, blockCoords) in enumerate(allBlocks):
		col = i % blocksPerSide
		row = i // blocksPerSide

		ax = col * blockSize
		ay = row * blockSize

		atlas[ay:ay+bh, ax:ax+bw] = imgBlock

		idx, _, _, _, _ = placementsMeta[i]

		u0 = (ax + UV_PADDING) / atlasWidth
		v0 = (ay + UV_PADDING) / atlasHeight
		u1 = (ax + blockSize - UV_PADDING) / atlasWidth
		v1 = (ay + blockSize - UV_PADDING) / atlasHeight

		gx, gy = blockCoords[0]
		bx = gx * blockSize
		by = gy * blockSize

		placements.append((idx, bx, by, u0, v1, u1, v0))

	atlas[...][atlas[..., 3] < ALPHA_TEST_THRESHOLD * 255] = [0, 0, 0, 0]
	atlas[..., 3][atlas[..., 3] >= ALPHA_TEST_THRESHOLD * 255] = 255

	print('done')

	return atlas, placements

# ------------------------------------------- #

def fill_block_depth(depthBlock):

	# find out where depth valid:
	# ---------------
	valid = depthBlock > 0
	if not valid.any():
		return depthBlock

	ys, xs = np.where(valid)
	zs = depthBlock[ys, xs]

	# remove outliers:
	# ---------------
	mean = zs.mean()
	std = zs.std()
	inlierMask = np.abs(zs - mean) <= DEPTH_INFILL_OUTLIER_STD * std
	xs, ys, zs = xs[inlierMask], ys[inlierMask], zs[inlierMask]

	# fit to plane:
	# ---------------
	A = np.stack([xs, ys, np.ones_like(xs)], axis=1)
	coeff, *_ = np.linalg.lstsq(A, zs, rcond=None)
	a, b, c = coeff

	H, W = depthBlock.shape
	yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
	zEst = a * xx + b * yy + c

	# infill:
	# ---------------
	filled = depthBlock.copy()

	if valid.mean() < DEPTH_INFILL_CUTOFF:
		filled[~valid] = mean
	else:
		filled[~valid] = zEst[~valid]

	return filled


def build_geometry(placements, slices, width, height, blockSize, focal, aspect):

	# infill depth for each block:
	# ---------------
	blockDepths = {}
	for (sliceIdx, px, py, _, _, _, _) in tqdm(placements, desc="Filling block depths", unit="block"):
		_, depth = slices[sliceIdx]

		depthBlock = depth[py:py+blockSize, px:px+blockSize, 0]
		depthBlock = fill_block_depth(depthBlock)

		blockDepths[f"{sliceIdx}:{px}:{py}"] = depthBlock

	# define vertex position helper:
	# ---------------
	def get_position(sliceIdx, px, py, checkPrev = True):

		zAccum = 0
		count = 0
		def read_depth(gx, gy):
			nonlocal zAccum, count

			key = f"{sliceIdx}:{px + gx * blockSize}:{py + gy * blockSize}"
			if key in blockDepths:
				block = blockDepths[key]

				zAccum += block[(blockSize + gy) % blockSize, (blockSize + gx) % blockSize]
				count += 1

		read_depth( 0,  0)
		read_depth(-1,  0)
		read_depth( 0, -1)
		read_depth(-1, -1)

		if count > 0:
			z = zAccum / count
		else:
			z = 0

		if checkPrev:
			for i in range(0, sliceIdx):
				_, _, prevZ = get_position(i, px, py, False)
				if z < prevZ:
					z = prevZ

		x = (px - width  * 0.5) * z / focal
		y = (height * 0.5 - py) * z / focal

		return x, y, z

	# construct mesh grid:
	# ---------------
	positions = []
	uvs = []
	indices = []
	idx = 0

	placements = sorted(placements, key=lambda x: x[0])
	for (sliceIdx, px, py, u0, v0, u1, v1) in tqdm(placements, desc="Generating geometry", unit="block"):
		positions += [
			*get_position(sliceIdx, px            , py + blockSize),
			*get_position(sliceIdx, px + blockSize, py + blockSize),
			*get_position(sliceIdx, px + blockSize, py            ),
			*get_position(sliceIdx, px            , py            )
		]

		uvs += [
			u0, v0,
			u1, v0,
			u1, v1,
			u0, v1
		]

		indices += [
			idx, idx+1, idx+2,
			idx, idx+2, idx+3
		]
		idx += 4

	return (
		np.array(positions, dtype=np.float32),
		np.array(uvs, dtype=np.float32),
		np.array(indices, dtype=np.uint32)
	)

# ------------------------------------------- #

def mlsharp_to_spatial_photo(
	orgImagePath, plyPath, 
	outGLB, outStereoImage
	):

	torch.set_default_device('cuda')

	# load original image:
	# ---------------	
	print('Reading original image... ', end='', flush=True)

	orgImage = Image.open(orgImagePath)
	orgWidth = orgImage.width
	orgHeight = orgImage.height

	outfilledWidth  = math.floor((1 + OUTFILL_AMOUNT) * orgWidth)
	outfilledHeight = math.floor((1 + OUTFILL_AMOUNT) * orgHeight)

	outfilledWidth  = (outfilledWidth  // BLOCK_SIZE) * BLOCK_SIZE
	outfilledHeight = (outfilledHeight // BLOCK_SIZE) * BLOCK_SIZE

	aspect = outfilledWidth / outfilledHeight

	print('done')

	# load ply:
	# ---------------	
	print('Reading gaussians... ', end='', flush=True)

	gaussians, focalY = load_ply(plyPath)
	fov = 2 * math.atan(outfilledHeight / (2 * focalY))

	print('done')

	# create render settings:
	# ---------------
	eye    = torch.tensor([0.0, 0.0, 0.0])
	target = torch.tensor([0.0, 0.0, 1.0])
	up     = torch.tensor([0.0, 1.0, 0.0])
	view = look_at(eye, target, up)

	proj = perspective(fov, aspect, 0.1, 1000.0)
	focalX = focalY

	settings = ddgs.Settings(
		width=outfilledWidth, height=outfilledHeight,
		view=view, proj=proj,
		focalX=focalX, focalY=focalY,
		outputs=ddgs.RenderOutputs.COLOR | ddgs.RenderOutputs.ALPHA | ddgs.RenderOutputs.DEPTH,
		debug=False
	)

	# render slices:
	# ---------------
	means = gaussians[0]
	zMin = torch.quantile(means[:, 2], DEPTH_MIN_QUANTILE).item()
	zMax = torch.quantile(means[:, 2], DEPTH_MAX_QUANTILE).item()

	slices = []
	
	for i in tqdm(range(NUM_SLICES), desc='Rendering slices', unit='slice'):
		render = ddgs.render(
			settings,
			*get_slice(
				gaussians, 
				zMin, zMax, 
				NUM_SLICES, i
			)
		)
		renderBehind = ddgs.render(
			settings,
			*get_slice(
				gaussians, 
				zMin, zMax, 
				NUM_SLICES, i, 
				includeBehind=True
			)
		)
	
		color = renderBehind.color.detach().cpu().numpy()
		alpha = render.alpha.detach().cpu().numpy()
		depth = render.depth.detach().cpu().numpy()

		img = np.dstack((color, alpha))
		img = (img * 255).astype(np.uint8)

		depth[depth > zMax] = zMax
		depth[np.logical_and(depth < zMin, depth > 0)] = zMin

		slices.append((img, depth))

	# replace pixels where GT data exists:
	# ---------------
	print('Replacing renders with GT color... ', end='', flush=True)

	orgRGB = np.array(orgImage.convert("RGB"), dtype=np.uint8)
	orgRGB = np.flip(orgRGB, 1)

	alphaStack = np.stack([
		slices[i][0][..., 3]
		for i in range(NUM_SLICES)
	], axis=0)

	hitMask = alphaStack > (ALPHA_REPLACE_THRESHOLD * 255.0)
	hitAny = hitMask.any(axis=0)
	firstHit = np.argmax(hitMask, axis=0)

	offX = (outfilledWidth - orgWidth) // 2
	offY = (outfilledHeight - orgHeight) // 2
	yy, xx = np.meshgrid(
		np.arange(outfilledHeight),
		np.arange(outfilledWidth),
		indexing="ij"
	)

	insideGT = (
		(xx >= offX) & (xx < offX + orgWidth) &
		(yy >= offY) & (yy < offY + orgHeight)
	)

	for s in range(NUM_SLICES):
		mask = (firstHit == s) & hitAny & insideGT

		if not mask.any():
			continue

		orgX = xx[mask] - offX
		orgY = yy[mask] - offY

		slices[s][0][mask, :3] = orgRGB[orgY, orgX]

	print('done')

	# generate geometry:
	# ---------------
	if outGLB is None:
		atlas, placements = generate_block_atlas_fast(
			slices,
			BLOCK_SIZE
		)
	else:
		atlas, placements = generate_block_atlas(
			slices,
			BLOCK_SIZE
		)

	positions, uvs, indices = build_geometry(
		placements,
		slices,
		outfilledWidth,
		outfilledHeight,
		BLOCK_SIZE,
		focalY,
		aspect
	)

	# save as GLB:
	# ---------------
	if outGLB is not None:
		print('Writing GLB... ', end='', flush=True)

		export_glb(atlas, positions, uvs, indices, outGLB)

		print('done')

	# render stereo image:
	# ---------------
	if outStereoImage is not None:
		print('Rendering stereo image... ', end='', flush=True)

		eyeLeft     = torch.tensor([ IPD / 2, 0.0, 0.0])
		targetLeft  = torch.tensor([ IPD / 2, 0.0, 1.0])
		viewLeft = look_at(eyeLeft, targetLeft, up)

		eyeRight    = torch.tensor([-IPD / 2, 0.0, 0.0])
		targetRight = torch.tensor([-IPD / 2, 0.0, 1.0])
		viewRight = look_at(eyeRight, targetRight, up)

		imgLeft = render_headless(
			positions, uvs, indices, atlas, 
			(orgWidth, orgHeight),
			viewLeft.cpu().numpy(), 
			proj.cpu().numpy()
		)
		imgRight = render_headless(
			positions, uvs, indices, atlas, 
			(orgWidth, orgHeight),
			viewRight.cpu().numpy(), 
			proj.cpu().numpy()
		)

		stereo = Image.new(imgLeft.mode, (orgWidth * 2, orgHeight))
		stereo.paste(imgLeft, (0, 0))
		stereo.paste(imgRight, (orgWidth, 0))
		stereo.save(outStereoImage)

		print('done')

# TODO: AI SLOP!!!!!

import os
import re
import argparse
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(
		description="Batch process frame_N.png and frame_N.ply pairs."
	)

	parser.add_argument("--image_dir", required=True,
						help="Directory containing frame_XXX.png images")
	parser.add_argument("--ply_dir", required=True,
						help="Directory containing frame_XXX.ply files")

	parser.add_argument("--out_glb_dir", default=None,
						help="Output directory for GLB files (optional)")
	parser.add_argument("--out_png_dir", default=None,
						help="Output directory for stereo PNG files (optional)")

	args = parser.parse_args()

	image_dir = Path(args.image_dir)
	ply_dir   = Path(args.ply_dir)

	out_glb_dir = Path(args.out_glb_dir) if args.out_glb_dir else None
	out_png_dir = Path(args.out_png_dir) if args.out_png_dir else None

	if out_glb_dir:
		out_glb_dir.mkdir(parents=True, exist_ok=True)

	if out_png_dir:
		out_png_dir.mkdir(parents=True, exist_ok=True)

	pattern = re.compile(r"frame_?(\d+)\.png$")

	image_files = sorted(image_dir.glob("*.png"))

	pairs = []

	for img_path in image_files:
		match = pattern.search(img_path.name)
		if not match:
			continue

		idx = match.group(1)

		# Look for matching PLY
		ply_path = ply_dir / f"frame_{idx}.ply"
		if not ply_path.exists():
			ply_path = ply_dir / f"frame{idx}.ply"

		if ply_path.exists():
			pairs.append((idx, img_path, ply_path))

	total = len(pairs)
	print(f"Found {total} matching frame pairs\n")

	for i, (idx, img_path, ply_path) in enumerate(pairs, 1):

		out_glb = out_glb_dir / f"frame_{idx}.glb" if out_glb_dir else None
		out_png = out_png_dir / f"frame_{idx}.png" if out_png_dir else None

		mlsharp_to_spatial_photo(
			orgImagePath=str(img_path),
			plyPath=str(ply_path),
			outGLB=str(out_glb) if out_glb else None,
			outStereoImage=str(out_png) if out_png else None
		)

		print(f"FINISHED {i}/{total} (frame_{idx})\n")


if __name__ == "__main__":
	main()

	# mlsharp_to_spatial_photo(
	# 	orgImagePath="insidious/clip2/frames/frame_045.png",
	# 	plyPath="insidious/clip2/plys/frame_045.ply",
	# 	outGLB="insidious/clip2/glbs/frame_045.glb",
	# 	outStereoImage="insidious/clip2/stereo/frame_045.png",
	# )
