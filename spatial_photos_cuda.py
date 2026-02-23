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

import renderer
import exporter

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

IPD = 0.016

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
		return torch.tensor(arr, dtype=torch.float32, device='cuda')

	means = np_to_torch(['x', 'y', 'z'])
	colors = 0.5 + np_to_torch(['f_dc_0', 'f_dc_1', 'f_dc_2']) * 0.28209479177387814
	opacities = torch.sigmoid(np_to_torch('opacity').unsqueeze(1))
	scales = torch.exp(np_to_torch(['scale_0', 'scale_1', 'scale_2']))
	rotations = np_to_torch(['rot_1', 'rot_2', 'rot_3', 'rot_0'])

	numGaussians = means.shape[0]

	colors = colors.reshape((numGaussians, 1, 3))

	gaussians = (means, scales, rotations, opacities, colors)
	focalY = data['intrinsic'].data['intrinsic'][0]

	return gaussians, focalY

# ------------------------------------------- #

def slice_t(numSlices, idx):
	return (idx / numSlices) * (idx / numSlices)

def get_slice(gaussians, zMin, zMax, numSlices, idx, includeBehind=False):
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

def pack_blocks(mergedBlockDims):
	
	# binary search to find best size:
	# ---------------
	def fits(size):
		packer = rectpack.newPacker(rotation=False)
		for i, (w, h) in enumerate(mergedBlockDims):
			packer.add_rect(w, h, i)

		packer.add_bin(size, size)
		packer.pack()

		return len(packer.rect_list()) == len(mergedBlockDims)

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
	for i, (w, h) in enumerate(mergedBlockDims):
		finalPacker.add_rect(w, h, i)

	finalPacker.add_bin(bestSize, bestSize)
	finalPacker.pack()

	return finalPacker


def generate_block_atlas(slices, blockSize):
	mergedMeta = []
	mergedDims = []

	# greedy mesh each slice:
	# ---------------
	for idx, (img, _) in enumerate(tqdm(slices, desc="Greedy meshing slices", unit="slice")):
		H, W, _ = img.shape
		GH, GW = H // blockSize, W // blockSize

		alpha = img[:GH*blockSize, :GW*blockSize, 3]
		alpha = alpha.reshape(GH, blockSize, GW, blockSize)

		present = (alpha >= int(ALPHA_TEST_THRESHOLD * 255)).any(dim=1).any(dim=2)
		presentCpu = present.cpu().numpy()

		if not np.any(presentCpu):
			continue

		rects = greedy_mesh(presentCpu)

		for (gx, gy, gw, gh) in rects:
			px, py = gx * blockSize, gy * blockSize
			pw, ph = gw * blockSize, gh * blockSize

			blockCoords = [(gx + dx, gy + dy) for dy in range(gh) for dx in range(gw)]
			mergedMeta.append((idx, px, py, pw, ph, blockCoords))
			mergedDims.append((pw, ph))

	# pack greedy meshed rects:
	# ---------------
	print("Packing slices into atlas... ", end='', flush=True)

	packer = pack_blocks(mergedDims)

	bin0 = packer.bin_list()[0]
	atlasW, atlasH = bin0

	atlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')

	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		sliceIdx, srcPx, srcPy, pw, ph, blockCoords = mergedMeta[i]
		img, _ = slices[sliceIdx]

		srcPatch = img[srcPy:srcPy+ph, srcPx:srcPx+pw]
		atlas[ay:ay+ah, ax:ax+aw] = srcPatch

		for (gx, gy) in blockCoords:
			bx = gx * blockSize
			by = gy * blockSize

			ox = bx - srcPx
			oy = by - srcPy

			u0 = (ax + ox + UV_PADDING) / atlasW
			v0 = (ay + oy + UV_PADDING) / atlasH
			u1 = (ax + ox + blockSize - UV_PADDING) / atlasW
			v1 = (ay + oy + blockSize - UV_PADDING) / atlasH

			placements.append((sliceIdx, bx, by, u0, v1, u1, v0))

	alpha = atlas[..., 3]
	transparent = alpha < int(ALPHA_TEST_THRESHOLD * 255)

	atlas[transparent] = 0
	atlas[..., 3][~transparent] = 255

	print('done')

	return atlas, placements


# ------------------------------------------- #
# Depth infill — fully on CUDA
# ------------------------------------------- #

def fill_block_depth_cuda(depth_block: torch.Tensor) -> torch.Tensor:
	"""
	depth_block: (blockSize, blockSize) float32 CUDA tensor
	Returns filled tensor, same shape and device.
	"""
	valid = depth_block > 0
	if not valid.any():
		return depth_block

	ys, xs = torch.where(valid)
	zs = depth_block[ys, xs]

	mean = zs.mean()
	std = zs.std() if zs.shape[0] >= 2 else torch.tensor(0.0, device=zs.device)
	inlier = (zs - mean).abs() <= DEPTH_INFILL_OUTLIER_STD * std

	if not inlier.any():
		filled = depth_block.clone()
		filled[~valid] = mean
		return filled

	xs_in = xs[inlier].float()
	ys_in = ys[inlier].float()
	zs_in = zs[inlier]

	# Plane fit via least squares on CUDA
	A = torch.stack([xs_in, ys_in, torch.ones_like(xs_in)], dim=1)  # (N, 3)
	sol = torch.linalg.lstsq(A, zs_in.unsqueeze(1)).solution         # (3, 1)
	a, b, c = sol[0, 0], sol[1, 0], sol[2, 0]

	H, W = depth_block.shape
	yy = torch.arange(H, device=depth_block.device, dtype=torch.float32)
	xx = torch.arange(W, device=depth_block.device, dtype=torch.float32)
	yy, xx = torch.meshgrid(yy, xx, indexing='ij')
	z_est = a * xx + b * yy + c

	filled = depth_block.clone()
	invalid = ~valid
	if valid.float().mean() < DEPTH_INFILL_CUTOFF:
		filled[invalid] = mean
	else:
		filled[invalid] = z_est[invalid]

	return filled

def fill_block_depths(placements, slices, blockSize):

	B = blockSize

	# collect unique blocks:
	# ---------------
	seen = {}
	keys = []
	for (sliceIdx, px, py, *_) in placements:
		key = (sliceIdx, px, py)
		if key not in seen:
			seen[key] = len(keys)
			keys.append(key)

	N = len(keys)

	# build list of blocks to process:
	# ---------------
	blocks = torch.zeros((N, B, B), dtype=torch.float32, device='cuda')
	for i, (sliceIdx, px, py) in enumerate(keys):
		_, depth = slices[sliceIdx]
		blocks[i] = depth[py:py+B, px:px+B, 0]

	# get per-block mean and stdev:
	# ---------------
	valid = blocks > 0
	validCount = valid.sum(dim=(1, 2)).float()

	flat     = blocks.reshape(N, B * B)
	validFlat = valid.reshape(N, B * B)

	sumZ  = (flat * validFlat.float()).sum(dim=1)
	meanZ = sumZ / validCount.clamp(min=1)

	diffSq = ((flat - meanZ.unsqueeze(1)) ** 2) * validFlat.float()
	varZ   = diffSq.sum(dim=1) / (validCount - 1).clamp(min=1)
	stdevZ = varZ.sqrt()
	stdevZ[validCount < 2] = 0.0

	inlier = validFlat & (
		(flat - meanZ.unsqueeze(1)).abs() <= DEPTH_INFILL_OUTLIER_STD * stdevZ.unsqueeze(1)
	)
	inlierCount = inlier.sum(dim=1).float()

	# fit to plane:
	# ---------------
	gy = torch.arange(B, device='cuda', dtype=torch.float32)
	gx = torch.arange(B, device='cuda', dtype=torch.float32)

	yy, xx = torch.meshgrid(gy, gx, indexing='ij')
	xxFlat = xx.reshape(B * B)
	yyFlat = yy.reshape(B * B)
	onesFlat = torch.ones(B * B, device='cuda')

	A = torch.stack([xxFlat, yyFlat, onesFlat], dim=1)  # (P, 3)

	inlierF = inlier.float().unsqueeze(2)
	ABatch  = A.unsqueeze(0).expand(N, -1, -1) * inlierF
	bBatch  = (flat * inlier.float()).unsqueeze(2)

	sol = torch.linalg.lstsq(ABatch, bBatch).solution
	a   = sol[:, 0, 0]
	b   = sol[:, 1, 0]
	c   = sol[:, 2, 0]

	zEstFlat = (
		a.unsqueeze(1) * xxFlat.unsqueeze(0) +
		b.unsqueeze(1) * yyFlat.unsqueeze(0) +
		c.unsqueeze(1)
	)

	# infill:
	# ---------------
	filled = flat.clone()
	invalidFlat = ~validFlat

	sparse = (validCount / (B * B)) < DEPTH_INFILL_CUTOFF
	useMean  = sparse.unsqueeze(1) & invalidFlat
	usePlane = (~sparse).unsqueeze(1) & invalidFlat

	filled[useMean]  = meanZ.unsqueeze(1).expand(N, B*B)[useMean]
	filled[usePlane] = zEstFlat[usePlane]

	noData = (validCount == 0).unsqueeze(1).expand(N, B*B)
	filled[noData] = flat[noData]

	noInlier = (inlierCount == 0).unsqueeze(1).expand(N, B*B)
	filled[noInlier & invalidFlat] = meanZ.unsqueeze(1).expand(N, B*B)[noInlier & invalidFlat]

	filled = filled.reshape(N, B, B)

	return {keys[i]: filled[i] for i in range(N)}

def build_geometry(placements, slices, width, height, blockSize, focal, aspect):

	# sort blocks by slice idx:
	# ---------------
	placements = sorted(placements, key=lambda x: x[0])
	N = len(placements)

	# infill depth for each block:
	# ---------------
	print("Filling block depths... ", end='', flush=True)

	blockDepths = fill_block_depths(placements, slices, blockSize)
	
	print('done')

	# average depth at corners, enforce monotonicity:
	# ---------------
	S  = NUM_SLICES
	GW = width  // blockSize
	GH = height // blockSize

	zGrid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device='cuda')
	countGrid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device='cuda')

	CORNER_OFFSETS = [
		(0, 0, 0,             0            ),
		(1, 0, 0,             blockSize - 1),
		(0, 1, blockSize - 1, 0            ),
		(1, 1, blockSize - 1, blockSize - 1),
	]
	for (sliceIdx, px, py, *_) in tqdm(placements, desc="Building Z grid", unit="block"):
		key = (sliceIdx, px, py)
		if key not in blockDepths:
			continue

		block = blockDepths[key]
		gx = px // blockSize
		gy = py // blockSize

		for (dgx, dgy, row, col) in CORNER_OFFSETS:
			cvx = gx + dgx
			cvy = gy + dgy
			if cvx > GW or cvy > GH:
				continue
				
			val = block[row, col]
			zGrid[sliceIdx, cvy, cvx]     += val
			countGrid[sliceIdx, cvy, cvx] += 1.0


	hasData = countGrid > 0
	zGrid[hasData] = zGrid[hasData] / countGrid[hasData]

	zGrid, _ = zGrid.cummax(dim=0)

	# compute worldspace vertex coordinates:
	# ---------------
	gxCoords = torch.arange(GW + 1, device='cuda', dtype=torch.float32) * blockSize
	gyCoords = torch.arange(GH + 1, device='cuda', dtype=torch.float32) * blockSize

	xOffset = gxCoords - width  * 0.5
	yOffset = height * 0.5 - gyCoords

	xOffset = xOffset.unsqueeze(0).unsqueeze(0).expand(S, GH+1, GW+1)
	yOffset = yOffset.unsqueeze(0).unsqueeze(2).expand(S, GH+1, GW+1)

	xWorld = xOffset * zGrid / focal
	yWorld = yOffset * zGrid / focal
	# zWorld is zGrid

	# compute positions and uvs:
	# ---------------
	plSiceIdx  = torch.tensor([p[0] for p in placements], dtype=torch.long,  device='cuda')
	plPx = torch.tensor([p[1] for p in placements], dtype=torch.long,  device='cuda')
	plPy = torch.tensor([p[2] for p in placements], dtype=torch.long,  device='cuda')
	plU0 = torch.tensor([p[3] for p in placements], dtype=torch.float32, device='cuda')
	plV0 = torch.tensor([p[4] for p in placements], dtype=torch.float32, device='cuda')
	plU1 = torch.tensor([p[5] for p in placements], dtype=torch.float32, device='cuda')
	plV1 = torch.tensor([p[6] for p in placements], dtype=torch.float32, device='cuda')

	gx0 = plPx // blockSize
	gy0 = plPy // blockSize
	gx1 = gx0 + 1
	gy1 = gy0 + 1

	cornerGx = torch.stack([gx0, gx1, gx1, gx0], dim=1)
	cornerGy = torch.stack([gy1, gy1, gy0, gy0], dim=1)
	cornerSliceIdx  = plSiceIdx.unsqueeze(1).expand(N, 4)

	cx = xWorld[cornerSliceIdx, cornerGy, cornerGx]
	cy = yWorld[cornerSliceIdx, cornerGy, cornerGx]
	cz = zGrid[cornerSliceIdx, cornerGy, cornerGx]

	positions = torch.stack([cx, cy, cz], dim=2).reshape(N * 4, 3)

	uvCorners = torch.stack([
		torch.stack([plU0, plV0], dim=1),
		torch.stack([plU1, plV0], dim=1),
		torch.stack([plU1, plV1], dim=1),
		torch.stack([plU0, plV1], dim=1),
	], dim=1)
	uvs = uvCorners.reshape(N * 4, 2)

	base = torch.arange(N, device='cuda', dtype=torch.int32) * 4
	tri0 = torch.stack([base, base + 1, base + 2], dim=1)
	tri1 = torch.stack([base, base + 2, base + 3], dim=1)
	indices = torch.cat([tri0, tri1], dim=1).reshape(N * 2, 3)

	return positions, uvs, indices

def replace_gt_color(slices, orgImage, outfilledWidth, outfilledHeight, orgWidth, orgHeight):

	orgRGB = torch.tensor(
		np.flip(np.array(orgImage.convert("RGB"), dtype=np.uint8), axis=1).copy(),
		device='cuda'
	)

	alphaStack = torch.stack([
		slices[i][0][..., 3]
		for i in range(len(slices))
	], dim=0)

	hitMask  = alphaStack > int(ALPHA_REPLACE_THRESHOLD * 255)
	hitAny   = hitMask.any(dim=0)
	firstHit = hitMask.to(torch.int32).argmax(dim=0)

	offX = (outfilledWidth  - orgWidth)  // 2
	offY = (outfilledHeight - orgHeight) // 2
	H, W = outfilledHeight, outfilledWidth

	yy = torch.arange(H, device='cuda').unsqueeze(1).expand(H, W)
	xx = torch.arange(W, device='cuda').unsqueeze(0).expand(H, W)

	insideGT = (xx >= offX) & (xx < offX + orgWidth) & (yy >= offY) & (yy < offY + orgHeight)

	for s in range(len(slices)):
		mask = (firstHit == s) & hitAny & insideGT

		if not mask.any():
			continue

		flatIdx = mask.nonzero(as_tuple=False)
		fy = flatIdx[:, 0]
		fx = flatIdx[:, 1]

		org_x = fx - offX
		org_y = fy - offY

		slices[s][0][fy, fx, :3] = orgRGB[org_y, org_x]

	return slices

# ------------------------------------------- #

def mlsharp_to_spatial_photo(orgImagePath, plyPath, outGLB, outStereoImages):

	torch.set_default_device('cuda')

	# load original image:
	# ---------------
	print('Reading original image... ', end='', flush=True)

	orgImage  = Image.open(orgImagePath)
	orgWidth  = orgImage.width
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
	view   = look_at(eye, target, up)

	proj   = perspective(fov, aspect, 0.1, 1000.0)
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
	zMin  = torch.quantile(means[:, 2], DEPTH_MIN_QUANTILE).item()
	zMax  = torch.quantile(means[:, 2], DEPTH_MAX_QUANTILE).item()

	slices = []

	for i in tqdm(range(NUM_SLICES), desc='Rendering slices', unit='slice'):
		with torch.no_grad():
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

		color = renderBehind.color
		alpha = render.alpha
		depth = render.depth

		img = torch.cat([color, alpha], dim=-1)
		img = (img * 255).to(torch.uint8)

		depth[depth > zMax] = zMax
		depth[(depth < zMin) & (depth > 0)] = zMin

		slices.append([img, depth])

	# replace pixels where GT data exists:
	# ---------------
	print('Replacing renders with GT color... ', end='', flush=True)

	replace_gt_color(slices, orgImage, outfilledWidth, outfilledHeight, orgWidth, orgHeight)
	
	print('done')

	# generate geometry:
	# ---------------
	atlas, placements = generate_block_atlas(slices, BLOCK_SIZE)

	positions, uvs, indices = build_geometry(
		placements, slices,
		outfilledWidth, outfilledHeight,
		BLOCK_SIZE, focalY, aspect
	)

	atlasCpu = atlas.cpu().numpy()
	positionsCpu = positions.cpu().numpy()
	uvsCpu = uvs.cpu().numpy()
	indicesCpu = indices.cpu().numpy()

	# sace as GLB:
	# ---------------
	if outGLB is not None:
		print('Writing GLB... ', end='', flush=True)

		exporter.export_glb(atlasCpu, positions, uvs, indices, outGLB)
		
		print('done')

	# render stereo images:
	# ---------------
	if outStereoImages is not None:
		scene = renderer.upload_scene(
			positionsCpu, uvsCpu, indicesCpu, 
			atlasCpu, (orgWidth, orgHeight)
		)

		try:
			for (ipd, path) in tqdm(outStereoImages, desc='Rendering stereo images', unit='image'):
				eyeLeft     = torch.tensor([ ipd / 2, 0.0, 0.0])
				targetLeft  = torch.tensor([ ipd / 2, 0.0, 1.0])
				viewLeft = look_at(eyeLeft, targetLeft, up)

				eyeRight    = torch.tensor([-ipd / 2, 0.0, 0.0])
				targetRight = torch.tensor([-ipd / 2, 0.0, 1.0])
				viewRight = look_at(eyeRight, targetRight, up)

				imgLeft = renderer.render_view(
					scene,
					viewLeft.cpu().numpy(), 
					proj.cpu().numpy()
				)
				imgRight = renderer.render_view(
					scene,
					viewRight.cpu().numpy(), 
					proj.cpu().numpy()
				)

				stereo = Image.new(imgLeft.mode, (orgWidth * 2, orgHeight))
				stereo.paste(imgLeft, (0, 0))
				stereo.paste(imgRight, (orgWidth, 0))
				stereo.save(path)
		finally:
			renderer.release_scene(scene)

# ------------------------------------------- #

import re
import argparse
from pathlib import Path

def main():
	parser = argparse.ArgumentParser(description="Batch process frame_N.png and frame_N.ply pairs.")
	parser.add_argument("--image_dir", required=True)
	parser.add_argument("--ply_dir",   required=True)
	parser.add_argument("--out_glb_dir", default=None)
	parser.add_argument("--out_png_dir", default=None)
	args = parser.parse_args()

	image_dir   = Path(args.image_dir)
	ply_dir     = Path(args.ply_dir)
	out_glb_dir = Path(args.out_glb_dir) if args.out_glb_dir else None
	out_png_dir = Path(args.out_png_dir) if args.out_png_dir else None

	if out_glb_dir: out_glb_dir.mkdir(parents=True, exist_ok=True)
	if out_png_dir: out_png_dir.mkdir(parents=True, exist_ok=True)

	pattern    = re.compile(r"frame_?(\d+)\.png$")
	image_files = sorted(image_dir.glob("*.png"))
	pairs = []

	for img_path in image_files:
		match = pattern.search(img_path.name)
		if not match:
			continue
		idx = match.group(1)
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
			outGLB=str(out_glb)  if out_glb else None,
			outStereoImage=str(out_png) if out_png else None
		)
		print(f"FINISHED {i}/{total} (frame_{idx})\n")


if __name__ == "__main__":
	mlsharp_to_spatial_photo(
		orgImagePath="insidious/clip2/frames/frame_044.png",
		plyPath     ="insidious/clip2/plys/frame_044.ply",
		outGLB      =None,#"insidious/clip2/glbs/frame_044.glb",
		outStereoImages=[(0.064, "test.png")],
	)
	main()