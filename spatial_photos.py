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

ALPHA_REPLACE_THRESHOLD = 25
ALPHA_TEST_THRESHOLD = 128

DEPTH_INFILL_CUTOFF = 0.1
DEPTH_INFILL_OUTLIER_STD = 2.0

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

UV_PADDING = 0.0

MOTION_THRESHOLD = 10.0
MOTION_THUMB_SIZE = (128, 72)
TEMPORAL_WINDOW = 2

BLEED_K = 3
BLEED_COLOR_TOLERANCE = 20

DEPTH_SMOOTH_ALPHA = 0.3
DEPTH_CROSS_SLICE_REL_TOLERANCE = 0.05

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

def compute_motion_score(imgA, imgB):
	a = imgA.convert('L').resize(MOTION_THUMB_SIZE, Image.BILINEAR)
	b = imgB.convert('L').resize(MOTION_THUMB_SIZE, Image.BILINEAR)

	ta = torch.tensor(np.array(a), dtype=torch.float32)
	tb = torch.tensor(np.array(b), dtype=torch.float32)
	
	return (ta - tb).abs().mean().item()

def temporal_smooth_slices(frameWindow, centerIdx, gtMasks):
	centerSlices = frameWindow[centerIdx]
	centerGtMasks = gtMasks[centerIdx]

	img0 = centerSlices[0][0]
	
	S = len(centerSlices)
	H, W = img0.shape[0], img0.shape[1]

	smoothed = [[img.clone(), depth.clone()] for (img, depth) in centerSlices]
	frameImgs = [
		[frameWindow[f][s][0].float() for s in range(S)]
		for f in range(len(frameWindow))
	]

	for s in range(S):
		centerImg  = smoothed[s][0]
		centerGtMask = centerGtMasks[s]

		accRgba  = torch.zeros((H, W, 4), dtype=torch.float32, device='cuda')
		accCount = torch.zeros((H, W),    dtype=torch.float32, device='cuda')

		for fi in range(len(frameWindow)):
			src = frameImgs[fi][s if s < S else S - 1]

			rgba  = src
			alpha = src[..., 3]

			found = alpha > ALPHA_TEST_THRESHOLD
			bestRgba = rgba.clone()

			accRgba[found]  += bestRgba[found]
			accCount[found] += 1.0

		hasData      = accCount > 0
		smoothedRgba = torch.zeros((H, W, 4), dtype=torch.float32, device='cuda')
		smoothedRgba[hasData] = accRgba[hasData] / accCount[hasData].unsqueeze(-1)

		applyMask = (~centerGtMask) & (centerImg[..., 3] > ALPHA_TEST_THRESHOLD) & hasData

		ys, xs = applyMask.nonzero(as_tuple=True)
		centerImg[ys, xs, 0] = smoothedRgba[ys, xs, 0].to(torch.uint8)
		centerImg[ys, xs, 1] = smoothedRgba[ys, xs, 1].to(torch.uint8)
		centerImg[ys, xs, 2] = smoothedRgba[ys, xs, 2].to(torch.uint8)
		centerImg[ys, xs, 3] = smoothedRgba[ys, xs, 3].to(torch.uint8)

		smoothed[s][0] = centerImg

	return smoothed

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


def generate_block_atlas(slicesGeom, slicesRender=None, gtMasks=None):
	mergedMeta = []
	mergedDims = []

	# greedy mesh each slice:
	# ---------------
	print("Greedy meshing slices...")

	for idx, (img, _) in enumerate(slicesGeom):
		H, W, _ = img.shape
		GH, GW = H // BLOCK_SIZE, W // BLOCK_SIZE

		alpha = img[:GH*BLOCK_SIZE, :GW*BLOCK_SIZE, 3]
		alpha = alpha.reshape(GH, BLOCK_SIZE, GW, BLOCK_SIZE)

		present = (alpha >= ALPHA_TEST_THRESHOLD).any(dim=1).any(dim=2)
		presentCpu = present.cpu().numpy()

		if not np.any(presentCpu):
			continue

		rects = greedy_mesh(presentCpu)

		for (gx, gy, gw, gh) in rects:
			px, py = gx * BLOCK_SIZE, gy * BLOCK_SIZE
			pw, ph = gw * BLOCK_SIZE, gh * BLOCK_SIZE

			blockCoords = [(gx + dx, gy + dy) for dy in range(gh) for dx in range(gw)]
			mergedMeta.append((idx, px, py, pw, ph, blockCoords))
			mergedDims.append((pw, ph))

	# pack greedy meshed rects:
	# ---------------
	print("Packing slices into atlas... ")

	packer = pack_blocks(mergedDims)

	bin0 = packer.bin_list()[0]
	atlasW, atlasH = bin0

	atlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')

	maskAtlas = None
	if gtMasks is not None:
		maskAtlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')

	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		sliceIdx, srcPx, srcPy, pw, ph, blockCoords = mergedMeta[i]
		img, _ = slicesGeom[sliceIdx] if slicesRender is None else slicesRender[sliceIdx]

		srcPatch = img[srcPy:srcPy+ph, srcPx:srcPx+pw]
		atlas[ay:ay+ah, ax:ax+aw] = srcPatch

		if maskAtlas is not None:
			gtPatch = gtMasks[sliceIdx, srcPy:srcPy+ph, srcPx:srcPx+pw]

			maskAtlas[ay:ay+ah, ax:ax+aw, :3] = torch.where(
				gtPatch.unsqueeze(-1), 0, 255
			)

		for (gx, gy) in blockCoords:
			bx = gx * BLOCK_SIZE
			by = gy * BLOCK_SIZE

			ox = bx - srcPx
			oy = by - srcPy

			u0 = (ax + ox + UV_PADDING) / atlasW
			v0 = (ay + oy + UV_PADDING) / atlasH
			u1 = (ax + ox + BLOCK_SIZE - UV_PADDING) / atlasW
			v1 = (ay + oy + BLOCK_SIZE - UV_PADDING) / atlasH

			placements.append((sliceIdx, bx, by, u0, v1, u1, v0))

	if maskAtlas is not None:
		maskAtlas[..., 3] = atlas[..., 3]

	return atlas, placements, maskAtlas

def fill_block_depths(placements, slices):

	B = BLOCK_SIZE

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

def build_geometry(placements, slices, width, height, focal, aspect):

	# sort blocks by slice idx:
	# ---------------
	placements = sorted(placements, key=lambda x: x[0])
	N = len(placements)

	# infill depth for each block:
	# ---------------
	print("Filling block depths...")

	blockDepths = fill_block_depths(placements, slices)

	# average depth at corners, enforce monotonicity:
	# ---------------
	print("Computing vertex depths...")

	S  = NUM_SLICES
	GW = width  // BLOCK_SIZE
	GH = height // BLOCK_SIZE

	zGrid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device='cuda')
	countGrid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device='cuda')

	plSliceIdx = torch.tensor([p[0] for p in placements], dtype=torch.long, device='cuda')
	plGx = torch.tensor([p[1] // BLOCK_SIZE for p in placements], dtype=torch.long, device='cuda')
	plGy = torch.tensor([p[2] // BLOCK_SIZE for p in placements], dtype=torch.long, device='cuda')

	cornerDGx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device='cuda')
	cornerDGy = torch.tensor([0, 0, 1, 1], dtype=torch.long, device='cuda')
	cornerRow = torch.tensor([0, 0, BLOCK_SIZE-1, BLOCK_SIZE-1], dtype=torch.long, device='cuda')
	cornerCol = torch.tensor([0, BLOCK_SIZE-1, 0, BLOCK_SIZE-1], dtype=torch.long, device='cuda')

	cvx = (plGx.unsqueeze(1) + cornerDGx.unsqueeze(0)).clamp(max=GW)
	cvy = (plGy.unsqueeze(1) + cornerDGy.unsqueeze(0)).clamp(max=GH)
	s   = plSliceIdx.unsqueeze(1).expand(N, 4)

	keysList = [(p[0], p[1], p[2]) for p in placements]
	blocksTensor = torch.stack([blockDepths[(si, px, py)] for (si, px, py) in keysList])
	vals = blocksTensor[:, cornerRow, cornerCol]

	validBounds = (plGx.unsqueeze(1) + cornerDGx.unsqueeze(0) <= GW) & \
	              (plGy.unsqueeze(1) + cornerDGy.unsqueeze(0) <= GH)
	validDepth = vals > 0.0
	valid = validBounds & validDepth

	flatIdx = (s * (GH+1) * (GW+1) + cvy * (GW+1) + cvx).reshape(-1)  # (N*4,)
	flatVals = (vals  * valid.float()).reshape(-1)
	latCount = valid.float().reshape(-1)

	zGrid    .reshape(-1).scatter_add_(0, flatIdx, flatVals)
	countGrid.reshape(-1).scatter_add_(0, flatIdx, latCount)

	hasData = countGrid > 0
	zGrid[hasData] = zGrid[hasData] / countGrid[hasData]

	zGrid, _ = zGrid.cummax(dim=0)

	return zGrid, GW, GH

def apply_depth_smoothing(zGrid, depthHistory, motionGated):

	S, GHp1, GWp1 = zGrid.shape
	hasDepth = zGrid > 0.0

	# initialize on first frame:
	# ---------------
	if 'ema' not in depthHistory:
		depthHistory['ema']   = zGrid.clone()
		depthHistory['valid'] = hasDepth.clone()

		return zGrid

	ema   = depthHistory['ema']
	valid = depthHistory['valid']

	# early exit if scene has high motion:
	# ---------------
	if motionGated:
		smoothed = torch.where(valid & hasDepth, ema, zGrid)
		return smoothed

	# fill any region without history:
	# ---------------
	noHistory = hasDepth & ~valid
	if noHistory.any():
		for dSlice in [1, -1]:
			if dSlice == 1:
				neighbValid = torch.cat([valid[1:],  valid[-1:].new_zeros((1, GHp1, GWp1))],  dim=0)
				neighbEma   = torch.cat([ema[1:],    ema[-1:].new_zeros((1, GHp1, GWp1))],    dim=0)
			else:
				neighbValid = torch.cat([valid[:1].new_zeros((1, GHp1, GWp1)), valid[:-1]],   dim=0)
				neighbEma   = torch.cat([ema[:1].new_zeros((1, GHp1, GWp1)),   ema[:-1]],     dim=0)

			depthClose = (
				(neighbEma - zGrid).abs() <
				DEPTH_CROSS_SLICE_REL_TOLERANCE * zGrid.clamp(min=1e-6)
			)

			canInherit = noHistory & neighbValid & depthClose

			ema   = torch.where(canInherit, neighbEma, ema)
			valid = valid | canInherit

			noHistory = noHistory & ~canInherit

	# apply smoothing:
	# ---------------
	alpha = DEPTH_SMOOTH_ALPHA

	hasHistory = valid & hasDepth
	newEma = torch.where(
		hasHistory,
		alpha * zGrid + (1.0 - alpha) * ema,
		ema
	)

	newEma = torch.where(noHistory, zGrid, newEma)
	newValid = valid | hasDepth

	depthHistory['ema']   = newEma
	depthHistory['valid'] = newValid

	smoothed = torch.where(newValid & hasDepth, newEma, zGrid)
	return smoothed


def finish_geometry(smoothedZGrid, placements, width, height, focal, aspect):
	"""
	Given an already-smoothed zGrid, complete the geometry build:
	enforce monotonicity then compute world-space positions and UVs.
	"""
	placements = sorted(placements, key=lambda x: x[0], reverse=True)
	N = len(placements)

	S  = NUM_SLICES
	GW = width  // BLOCK_SIZE
	GH = height // BLOCK_SIZE

	# enforce monotonicity across slices
	zGrid, _ = smoothedZGrid.cummax(dim=0)

	# compute worldspace vertex coordinates:
	# ---------------
	print("Computing vertex coordinates...")

	gxCoords = torch.arange(GW + 1, device='cuda', dtype=torch.float32) * BLOCK_SIZE
	gyCoords = torch.arange(GH + 1, device='cuda', dtype=torch.float32) * BLOCK_SIZE

	xOffset = gxCoords - width  * 0.5
	yOffset = height * 0.5 - gyCoords

	xOffset = xOffset.unsqueeze(0).unsqueeze(0).expand(S, GH+1, GW+1)
	yOffset = yOffset.unsqueeze(0).unsqueeze(2).expand(S, GH+1, GW+1)

	xWorld = xOffset * zGrid / focal
	yWorld = yOffset * zGrid / focal

	# compute positions and uvs:
	# ---------------
	print("Building vertex buffers...")

	plSiceIdx  = torch.tensor([p[0] for p in placements], dtype=torch.long,  device='cuda')
	plPx = torch.tensor([p[1] for p in placements], dtype=torch.long,  device='cuda')
	plPy = torch.tensor([p[2] for p in placements], dtype=torch.long,  device='cuda')
	plU0 = torch.tensor([p[3] for p in placements], dtype=torch.float32, device='cuda')
	plV0 = torch.tensor([p[4] for p in placements], dtype=torch.float32, device='cuda')
	plU1 = torch.tensor([p[5] for p in placements], dtype=torch.float32, device='cuda')
	plV1 = torch.tensor([p[6] for p in placements], dtype=torch.float32, device='cuda')

	gx0 = plPx // BLOCK_SIZE
	gy0 = plPy // BLOCK_SIZE
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

	hitMask  = alphaStack > ALPHA_REPLACE_THRESHOLD
	hitAny   = hitMask.any(dim=0)
	firstHit = hitMask.to(torch.int32).argmax(dim=0)

	offX = (outfilledWidth  - orgWidth)  // 2
	offY = (outfilledHeight - orgHeight) // 2
	H, W = outfilledHeight, outfilledWidth

	yy = torch.arange(H, device='cuda').unsqueeze(1).expand(H, W)
	xx = torch.arange(W, device='cuda').unsqueeze(0).expand(H, W)

	insideGT = (xx >= offX) & (xx < offX + orgWidth) & (yy >= offY) & (yy < offY + orgHeight)

	S = len(slices)
	gtMask = torch.zeros((S, H, W), dtype=torch.bool, device='cuda')

	for s in range(S):
		mask = (firstHit == s) & hitAny & insideGT

		if not mask.any():
			continue

		flatIdx = mask.nonzero(as_tuple=False)
		fy = flatIdx[:, 0]
		fx = flatIdx[:, 1]

		orgX = fx - offX
		orgY = fy - offY

		slices[s][0][fy, fx, :3] = orgRGB[orgY, orgX]
		gtMask[s, fy, fx] = True

		if BLEED_K >= 0 and s + 1 < S:
			transparentMap = (slices[s][0][..., 3] < ALPHA_TEST_THRESHOLD).float()
			transparentMap = transparentMap.unsqueeze(0).unsqueeze(0)
			nearTransparent = torch.nn.functional.max_pool2d(
				transparentMap,
				kernel_size=2 * BLEED_K + 1,
				stride=1,
				padding=BLEED_K,
			).squeeze(0).squeeze(0).bool()

			nextAboveThresh = slices[s + 1][0][..., 3] >= ALPHA_TEST_THRESHOLD

			nextRgb = slices[s + 1][0][..., :3].float()

			gtRGB = torch.zeros((H, W, 3), dtype=torch.float32, device='cuda')
			gtRGB[fy, fx] = orgRGB[orgY, orgX].float()

			colorDiff  = (nextRgb - gtRGB).abs().mean(dim=-1)  # (H, W)
			colorClose = colorDiff <= BLEED_COLOR_TOLERANCE

			bleedMask = mask & nearTransparent & nextAboveThresh & colorClose

			bleedIdx = bleedMask.nonzero(as_tuple=False)
			by = bleedIdx[:, 0]
			bx = bleedIdx[:, 1]

			bOrgX = bx - offX
			bOrgY = by - offY

			gtMask[s + 1, by, bx] = True

	return slices, gtMask

# ------------------------------------------- #

def spatial_photo_sequence(frameIndices, orgImagePathFn, plyPathFn, outGLBFn, outStereoImagesFn):

	torch.set_default_device('cuda')

	# get total frame count:
	# ---------------
	frameIndices = list(frameIndices)
	total = len(frameIndices)

	# get dimensions:
	# ---------------
	firstOrg = Image.open(orgImagePathFn(frameIndices[0]))
	orgWidth, orgHeight = firstOrg.width, firstOrg.height
	firstOrg.close()

	outfilledWidth  = math.floor((1 + OUTFILL_AMOUNT) * orgWidth)
	outfilledHeight = math.floor((1 + OUTFILL_AMOUNT) * orgHeight)
	outfilledWidth  = (outfilledWidth  // BLOCK_SIZE) * BLOCK_SIZE
	outfilledHeight = (outfilledHeight // BLOCK_SIZE) * BLOCK_SIZE
	aspect = outfilledWidth / outfilledHeight

	# init sliding window buffers:
	# ---------------
	orgImageCache   = {}
	slicesCache     = {}
	gtMaskCache     = {}
	projMatrixCache = {}
	settingsCache   = {}
	depthHistory = {}

	def load_frame(pos):
		if pos in slicesCache:
			return

		fi = frameIndices[pos]
		print(f'Loading frame {fi}...')

		# load src
		print(f'    Loading source image and ply...')
		
		orgImage  = Image.open(orgImagePathFn(fi))
		gaussians, focalY = load_ply(plyPathFn(fi))

		orgImageCache[pos] = orgImage.copy().convert('RGB')
		fov = 2 * math.atan(outfilledHeight / (2 * focalY))

		eye    = torch.tensor([0.0, 0.0, 0.0])
		target = torch.tensor([0.0, 0.0, 1.0])
		up     = torch.tensor([0.0, 1.0, 0.0])
		view   = look_at(eye, target, up)
		proj   = perspective(fov, aspect, 0.1, 1000.0)

		settings = ddgs.Settings(
			width=outfilledWidth, height=outfilledHeight,
			view=view, proj=proj,
			focalX=focalY, focalY=focalY,
			outputs=ddgs.RenderOutputs.COLOR | ddgs.RenderOutputs.ALPHA | ddgs.RenderOutputs.DEPTH,
			debug=False
		)

		means = gaussians[0]
		zMin  = torch.quantile(means[:, 2], DEPTH_MIN_QUANTILE).item()
		zMax  = torch.quantile(means[:, 2], DEPTH_MAX_QUANTILE).item()

		# render slices
		print(f'    Rendering slices...')

		slices = []
		for i in range(NUM_SLICES):
			with torch.no_grad():
				render = ddgs.render(
					settings,
					*get_slice(gaussians, zMin, zMax, NUM_SLICES, i)
				)
				renderBehind = ddgs.render(
					settings,
					*get_slice(gaussians, zMin, zMax, NUM_SLICES, i, includeBehind=True)
				)

			color = renderBehind.color
			alpha = render.alpha
			depth = render.depth

			img = torch.cat([color, alpha], dim=-1)
			img = (img * 255).to(torch.uint8)

			depth[depth > zMax] = zMax
			depth[(depth < zMin) & (depth > 0)] = zMin

			slices.append([img, depth])

		# replace with GT
		print(f'    Replacing renders with GT color...')

		slices, gtMask = replace_gt_color(slices, orgImage, outfilledWidth, outfilledHeight, orgWidth, orgHeight)

		slicesCache[pos] = slices
		gtMaskCache[pos] = gtMask
		projMatrixCache[pos] = proj

	def evict_old_frames(centerPos):
		for pos in list(slicesCache.keys()):
			if pos < centerPos - TEMPORAL_WINDOW:
				del slicesCache[pos]
				del gtMaskCache[pos]
				del projMatrixCache[pos]
				del orgImageCache[pos]

	# process each frame:
	# ---------------
	for centerPos in range(total):
		fi = frameIndices[centerPos]
		print(f'\n--- Processing frame {fi} ({centerPos+1}/{total}) ---\n')

		windowStart = max(0, centerPos - TEMPORAL_WINDOW)
		windowEnd   = min(total - 1, centerPos + TEMPORAL_WINDOW)

		# load each frame needed for smoothing
		for pos in range(windowStart, windowEnd + 1):
			load_frame(pos)

		# build smoothing list, exclude frames with high motion
		centerImg = orgImageCache[centerPos]
		windowSlices  = []
		windowGtMasks = []
		excluded = 0
		highMotion = False
		for pos in range(windowStart, windowEnd + 1):
			if pos == centerPos:
				windowSlices.append(slicesCache[pos])
				windowGtMasks.append(gtMaskCache[pos])
			else:
				score = compute_motion_score(centerImg, orgImageCache[pos])
				if score <= MOTION_THRESHOLD:
					windowSlices.append(slicesCache[pos])
					windowGtMasks.append(gtMaskCache[pos])
				else:
					excluded += 1
					highMotion = True
					print(f'Excluding frame pos={pos} from temporal smoothing (score={score:.1f} > threshold={MOTION_THRESHOLD})')

		centerInWindow = sum(
			1 for pos in range(windowStart, centerPos)
			if compute_motion_score(centerImg, orgImageCache[pos]) <= MOTION_THRESHOLD
		)

		# temporal smoothing (color)
		print(f'Applying temporal color smoothing...')

		smoothedSlices = temporal_smooth_slices(
			frameWindow=windowSlices,
			centerIdx=centerInWindow,
			gtMasks=windowGtMasks
		)

		# generate block atlas
		originalSlices = slicesCache[centerPos]
		gtMasks = gtMaskCache[centerPos]

		outStereoImages = outStereoImagesFn(fi) if outStereoImagesFn else None
		needsMaskAtlas = outStereoImages is not None and any(maskPath is not None for (_, _, maskPath) in outStereoImages)

		atlas, placements, maskAtlas = generate_block_atlas(
			originalSlices,
			smoothedSlices,
			gtMasks=gtMasks if needsMaskAtlas else None
		)

		# build geometry (returns raw zGrid before smoothing)
		proj = projMatrixCache[centerPos]
		focal = float(proj[1, 1].item() * outfilledHeight / 2)

		print("Building raw depth grid...")
		zGrid, GW, GH = build_geometry(
			placements, originalSlices,
			outfilledWidth, outfilledHeight,
			focal, aspect
		)

		# temporal smoothing (depth)
		print("Applying temporal depth smoothing...")

		smoothedZGrid = apply_depth_smoothing(zGrid, depthHistory, motionGated=highMotion)

		# finish geometry from smoothed zGrid
		positions, uvs, indices = finish_geometry(
			smoothedZGrid, placements,
			outfilledWidth, outfilledHeight,
			focal, aspect
		)

		atlasCpu     = atlas.cpu().numpy()
		maskAtlasCpu = maskAtlas.cpu().numpy() if maskAtlas is not None else None
		positionsCpu = positions.cpu().numpy()
		uvsCpu       = uvs.cpu().numpy()
		indicesCpu   = indices.cpu().numpy()

		# save GLB
		outGLB = outGLBFn(fi) if outGLBFn else None
		if outGLB is not None:
			print(f'Writing GLB to {outGLB}...')
			exporter.export_glb(atlasCpu, positionsCpu, uvsCpu, indicesCpu, outGLB)

		# render stereo images
		if outStereoImages is not None:
			up = torch.tensor([0.0, 1.0, 0.0])
			scene = renderer.upload_scene(positionsCpu, uvsCpu, indicesCpu, atlasCpu, (orgWidth, orgHeight))

			maskScene = None
			if maskAtlasCpu is not None:
				maskScene = renderer.upload_scene(positionsCpu, uvsCpu, indicesCpu, maskAtlasCpu, (orgWidth, orgHeight))

			try:
				for entry in outStereoImages:
					ipd, path, maskPath = entry

					print(f'Saving stereo render: {path}...')

					eyeLeft    = torch.tensor([ ipd / 2, 0.0, 0.0])
					targetLeft = torch.tensor([ ipd / 2, 0.0, 1.0])
					viewLeft   = look_at(eyeLeft, targetLeft, up)

					eyeRight    = torch.tensor([-ipd / 2, 0.0, 0.0])
					targetRight = torch.tensor([-ipd / 2, 0.0, 1.0])
					viewRight   = look_at(eyeRight, targetRight, up)

					imgLeft  = renderer.render_view(scene, viewLeft.cpu().numpy(),  proj.cpu().numpy())
					imgRight = renderer.render_view(scene, viewRight.cpu().numpy(), proj.cpu().numpy())

					stereo = Image.new(imgLeft.mode, (orgWidth * 2, orgHeight))
					stereo.paste(imgLeft,  (0, 0))
					stereo.paste(imgRight, (orgWidth, 0))
					stereo.save(path, compress_level=1)

					# optionally save mask stereo image
					if maskPath is not None and maskScene is not None:
						print(f'Saving stereo mask: {maskPath}...')

						maskLeft  = renderer.render_view(maskScene, viewLeft.cpu().numpy(),  proj.cpu().numpy())
						maskRight = renderer.render_view(maskScene, viewRight.cpu().numpy(), proj.cpu().numpy())

						maskLeft  = maskLeft.convert('L')
						maskRight = maskRight.convert('L')

						stereoMask = Image.new('L', (orgWidth * 2, orgHeight))
						stereoMask.paste(maskLeft,  (0, 0))
						stereoMask.paste(maskRight, (orgWidth, 0))
						stereoMask.save(maskPath, compress_level=1)

			finally:
				renderer.release_scene(scene)
				if maskScene is not None:
					renderer.release_scene(maskScene)

		evict_old_frames(centerPos)

def spatial_photo(orgImagePath, plyPath, outGLB, outStereoImages):
	spatial_photo_sequence(
	    frameIndices=range(1),
	    orgImagePathFn =lambda i: orgImagePath,
	    plyPathFn      =lambda i: plyPath,
	    outGLBFn       =lambda i: outGLB,
	    outStereoImagesFn=lambda i: outStereoImages
	)

# ------------------------------------------- #

if __name__ == "__main__":
	import argparse
	import glob
	import re

	# setup argparse:
	# ---------------
	parser = argparse.ArgumentParser(description="Generate spatial photos / stereo image sequences from ML-Sharp outputs")

	parser.add_argument("input", help=(
		"Either a single source image (e.g. photo.png) or a directory containing "
		"'frames/' and 'plys/' subdirectories for sequence mode."
	))

	parser.add_argument("--start", type=int, default=None, help="First frame index to process (sequence mode only).")
	parser.add_argument("--end",   type=int, default=None, help="Last frame index to process, inclusive (sequence mode only).")
	parser.add_argument("--frame-digits", type=int, default=3, help="Zero-padding width for frame filenames, e.g. 3 → frame_001.png (default: 3).")
	parser.add_argument("--frame-prefix", type=str, default="frame_", help="Filename prefix for frame/ply files (default: 'frame_').")

	parser.add_argument("--ply", type=str, default=None, help="Path to PLY file (single file mode only; inferred from input path otherwise).")

	parser.add_argument("--out-glb",    type=str, default=None, help="Output GLB path or directory (sequence mode writes per-frame GLBs here).")
	parser.add_argument("--out-stereo", type=str, default=None, help="Output stereo image path or directory.")
	parser.add_argument("--out-mask",   type=str, default=None, help="Output stereo mask image path or directory (requires --out-stereo).")
	parser.add_argument("--ipd", type=int, nargs="+", default=[64], metavar="MM",
		help="One or more interpupillary distances in millimetres (default: 64). Each IPD is rendered into its own ipd_NNN subdirectory.")

	args = parser.parse_args()

	# get paths:
	# ---------------
	inputPath = args.input
	isDir = os.path.isdir(inputPath)

	def out_path(base, fi, digits, prefix, ext):
		os.makedirs(base, exist_ok=True)
		return os.path.join(base, f"{prefix}{fi:0{digits}d}{ext}")

	# sequence mode:
	# ---------------
	if isDir:
		framesDir = os.path.join(inputPath, "frames")
		plysDir   = os.path.join(inputPath, "plys")

		if not os.path.isdir(framesDir):
			parser.error(f"Expected a 'frames/' subdirectory inside '{inputPath}'.")
		if not os.path.isdir(plysDir):
			parser.error(f"Expected a 'plys/' subdirectory inside '{inputPath}'.")

		digits = args.frame_digits
		prefix = args.frame_prefix

		pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
		available = sorted(
			int(m.group(1))
			for f in os.listdir(framesDir)
			if (m := pattern.match(f))
		)

		if not available:
			parser.error(f"No frame files matching '{prefix}<N>.png/jpg' found in '{framesDir}'.")

		start = args.start if args.start is not None else available[0]
		end   = args.end   if args.end   is not None else available[-1]

		frameIndices = [i for i in available if start <= i <= end]
		if not frameIndices:
			parser.error(f"No frames found in the range [{start}, {end}].")

		def frame_ext(fi):
			for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
				p = os.path.join(framesDir, f"{prefix}{fi:0{digits}d}{ext}")
				if os.path.exists(p):
					return ext
			return ".png"

		spatial_photo_sequence(
			frameIndices     = frameIndices,
			orgImagePathFn   = lambda i: os.path.join(framesDir, f"{prefix}{i:0{digits}d}{frame_ext(i)}"),
			plyPathFn        = lambda i: os.path.join(plysDir,   f"{prefix}{i:0{digits}d}.ply"),
			outGLBFn         = (lambda i: out_path(args.out_glb,    i, digits, prefix, ".glb")) if args.out_glb    else None,
			outStereoImagesFn= (lambda i: [
				(
					ipdMM / 1000.0,
					out_path(os.path.join(args.out_stereo, f"ipd_{ipdMM:03d}"), i, digits, prefix, ".png"),
					out_path(os.path.join(args.out_mask,   f"ipd_{ipdMM:03d}"), i, digits, prefix, ".png") if args.out_mask else None,
				)
				for ipdMM in args.ipd
			]) if args.out_stereo else None,
		)

	# single file mode:
	# ---------------
	else:
		if not os.path.isfile(inputPath):
			parser.error(f"'{inputPath}' is not a file or directory.")

		base, imgExt = os.path.splitext(inputPath)
		plyPath = args.ply if args.ply else base + ".ply"

		if not os.path.isfile(plyPath):
			parser.error(f"PLY file not found: '{plyPath}'. Use --ply to specify its path.")

		stereoBase = args.out_stereo if args.out_stereo else os.path.splitext(inputPath)[0] + "_stereo"
		maskBase   = args.out_mask   if args.out_mask   else None

		def single_stereo_entry(ipdMM):
			stereoDir = os.path.join(stereoBase, f"ipd_{ipdMM:03d}")
			os.makedirs(stereoDir, exist_ok=True)
			stereoOut = os.path.join(stereoDir, os.path.basename(base) + "_stereo.png")

			maskOut = None
			if maskBase is not None:
				maskDir = os.path.join(maskBase, f"ipd_{ipdMM:03d}")
				os.makedirs(maskDir, exist_ok=True)
				maskOut = os.path.join(maskDir, os.path.basename(base) + "_mask.png")

			return (ipdMM / 1000.0, stereoOut, maskOut)

		spatial_photo(
			orgImagePath    = inputPath,
			plyPath         = plyPath,
			outGLB          = args.out_glb,
			outStereoImages = [single_stereo_entry(ipdMM) for ipdMM in args.ipd],
		)