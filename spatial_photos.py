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
from concurrent.futures import ThreadPoolExecutor, Future

import renderer
import exporter

# ------------------------------------------- #

OUTFILL_AMOUNT = 0.0

DEPTH_MIN_QUANTILE = 0.0
DEPTH_MAX_QUANTILE = 0.8
NUM_SLICES = 30

BLOCK_SIZE = 64

ALPHA_THRESHOLD = 1
ALPHA_SOLID_THRESHOLD = 128

DEPTH_INFILL_CUTOFF = 0.1
DEPTH_INFILL_OUTLIER_STD = 2.0

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

UV_PADDING = 0.5

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

def slice_t(idx):
	return (idx / NUM_SLICES) * (idx / NUM_SLICES)

def get_slice(gaussians, zMin, zMax, idx):
	means, scales, rotations, opacities, colors = gaussians

	tMin = slice_t(idx)
	tMax = slice_t(idx + 1)

	zMinSlice = zMin + tMin * (zMax - zMin)
	zMaxSlice = zMin + tMax * (zMax - zMin)

	if (idx == NUM_SLICES - 1):
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

def generate_block_atlas(slices, gtMasks=None, opaqueOnly=False):

	B = BLOCK_SIZE

	# collect present blocks:
	# ---------------
	meta = []
	for idx, (img, _, _) in enumerate(slices):
		H, W, _ = img.shape
		GH, GW = H // B, W // B

		alpha = img[:GH*B, :GW*B, 3]
		alpha = alpha.reshape(GH, B, GW, B)

		threshold = ALPHA_SOLID_THRESHOLD if opaqueOnly else ALPHA_THRESHOLD
		present = (alpha >= threshold).any(dim=1).any(dim=2)

		gyIdx, gxIdx = present.nonzero(as_tuple=True)
		for gy, gx in zip(gyIdx.tolist(), gxIdx.tolist()):
			meta.append((idx, gx * B, gy * B))

	N = len(meta)
	if N == 0:
		empty = torch.zeros((B, B, 4), dtype=torch.uint8, device='cuda')
		return empty, [], (torch.zeros((B, B, 4), dtype=torch.uint8, device='cuda') if gtMasks is not None else None)

	# find smallest square atlas:
	# ---------------
	cols = math.ceil(math.sqrt(N))
	rows = math.ceil(N / cols)
	atlasW = cols * B
	atlasH = rows * B

	atlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')
	maskAtlas = None
	if gtMasks is not None:
		maskAtlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')

	# pack atlas:
	# ---------------
	placements = []

	for i, (sliceIdx, srcPx, srcPy) in enumerate(meta):
		ax = (i % cols) * B
		ay = (i // cols) * B

		img, _, _ = slices[sliceIdx]
		atlas[ay:ay+B, ax:ax+B] = img[srcPy:srcPy+B, srcPx:srcPx+B]

		if maskAtlas is not None:
			gtPatch = gtMasks[sliceIdx, srcPy:srcPy+B, srcPx:srcPx+B]
			maskAtlas[ay:ay+B, ax:ax+B, :3] = torch.where(
				gtPatch.unsqueeze(-1), 0, 255
			)

		u0 = (ax     + UV_PADDING) / atlasW
		v0 = (ay     + UV_PADDING) / atlasH
		u1 = (ax + B - UV_PADDING) / atlasW
		v1 = (ay + B - UV_PADDING) / atlasH

		placements.append((sliceIdx, srcPx, srcPy, u0, v1, u1, v0))

	if opaqueOnly:
		opaque = atlas[..., 3] >= ALPHA_SOLID_THRESHOLD

		alpha = atlas[..., 3:4].float() / 255.0
		nonzero = alpha > 0
		mask = opaque.unsqueeze(-1) & nonzero 
		atlas[..., :3] = torch.where(
			mask,
			(atlas[..., :3].float() / alpha).clamp(0, 255),
			atlas[..., :3]
		).to(atlas.dtype)

		atlas[..., 3] = opaque.to(atlas.dtype) * 255

	if maskAtlas is not None:
		maskAtlas[..., 3] = atlas[..., 3]

	return atlas, placements, maskAtlas

def generate_block_atlas_greedy(slices, gtMasks=None, opaqueOnly=False):
	mergedMeta = []
	mergedDims = []

	# greedy mesh each slice:
	# ---------------
	for idx, (img, _, _) in enumerate(slices):
		H, W, _ = img.shape
		GH, GW = H // BLOCK_SIZE, W // BLOCK_SIZE

		alpha = img[:GH*BLOCK_SIZE, :GW*BLOCK_SIZE, 3]
		alpha = alpha.reshape(GH, BLOCK_SIZE, GW, BLOCK_SIZE)

		threshold = ALPHA_SOLID_THRESHOLD if opaqueOnly else ALPHA_THRESHOLD
		present = (alpha >= threshold).any(dim=1).any(dim=2)
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
		img, _, _ = slices[sliceIdx]

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

			padLeft  = UV_PADDING if (ox == 0)               else 0
			padRight = UV_PADDING if (ox + BLOCK_SIZE >= pw) else 0
			padTop   = UV_PADDING if (oy == 0)               else 0
			padBot   = UV_PADDING if (oy + BLOCK_SIZE >= ph) else 0

			u0 = (ax + ox + padLeft) / atlasW
			v0 = (ay + oy + padTop) / atlasH
			u1 = (ax + ox + BLOCK_SIZE - padRight) / atlasW
			v1 = (ay + oy + BLOCK_SIZE - padBot) / atlasH

			placements.append((sliceIdx, bx, by, u0, v1, u1, v0))

	if opaqueOnly:
		opaque = atlas[..., 3] >= ALPHA_SOLID_THRESHOLD

		alpha = atlas[..., 3:4].float() / 255.0
		nonzero = alpha > 0
		mask = opaque.unsqueeze(-1) & nonzero 
		atlas[..., :3] = torch.where(
			mask,
			(atlas[..., :3].float() / alpha).clamp(0, 255),
			atlas[..., :3]
		).to(atlas.dtype)

		atlas[..., 3] = opaque.to(atlas.dtype) * 255

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
	slicePositions = torch.zeros(N, dtype=torch.float32, device='cuda')

	for i, (sliceIdx, px, py) in enumerate(keys):
		_, depth, slicePos = slices[sliceIdx]

		blocks[i] = depth[py:py+B, px:px+B, 0]
		slicePositions[i] = slicePos

	# compute mean depth:
	# ---------------
	valid = blocks > 0
	invalid = ~valid
	validCount = valid.sum(dim=(1, 2)).float()

	sumZ  = (blocks * valid.float()).sum(dim=(1, 2))
	meanZ = sumZ / validCount.clamp(min=1)

	# infill:
	# ---------------
	fillValue = torch.where(
		validCount == 0,
		slicePositions,
		meanZ
	).unsqueeze(-1).unsqueeze(-1).expand([-1, B, B])

	filled = blocks.clone()
	filled[invalid] = fillValue[invalid]

	return {keys[i]: filled[i] for i in range(N)}

def build_geometry(placements, slices, width, height, focal, aspect):

	# sort blocks by slice idx:
	# ---------------
	placements = sorted(placements, key=lambda x: x[0])
	N = len(placements)

	# infill depth for each block:
	# ---------------
	blockDepths = fill_block_depths(placements, slices)

	# average depth at corners, enforce monotonicity:
	# ---------------
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

def finish_geometry(zGrid, placements, width, height, focal, aspect):
	placements = sorted(placements, key=lambda x: x[0], reverse=True)
	N = len(placements)

	S  = NUM_SLICES
	GW = width  // BLOCK_SIZE
	GH = height // BLOCK_SIZE

	# enforce monotonicity across slices:
	# ---------------
	zGrid, _ = zGrid.cummax(dim=0)

	# compute worldspace vertex coordinates:
	# ---------------
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
	plSiceIdx = torch.tensor([p[0] for p in placements], dtype=torch.long,   device='cuda')
	plPx      = torch.tensor([p[1] for p in placements], dtype=torch.long,   device='cuda')
	plPy      = torch.tensor([p[2] for p in placements], dtype=torch.long,   device='cuda')
	plU0      = torch.tensor([p[3] for p in placements], dtype=torch.float32, device='cuda')
	plV0      = torch.tensor([p[4] for p in placements], dtype=torch.float32, device='cuda')
	plU1      = torch.tensor([p[5] for p in placements], dtype=torch.float32, device='cuda')
	plV1      = torch.tensor([p[6] for p in placements], dtype=torch.float32, device='cuda')

	gx0 = plPx // BLOCK_SIZE
	gy0 = plPy // BLOCK_SIZE
	gx1 = gx0 + 1
	gy1 = gy0 + 1

	cornerGx = torch.stack([gx0, gx1, gx1, gx0], dim=1)
	cornerGy = torch.stack([gy1, gy1, gy0, gy0], dim=1)
	cornerSliceIdx = plSiceIdx.unsqueeze(1).expand(N, 4)

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

	hitMask  = alphaStack > ALPHA_THRESHOLD
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

		slices[s][0][fy, fx, :3] = (orgRGB[orgY, orgX].float() * (slices[s][0][fy, fx, 3].unsqueeze(-1).float() / 255.0)).to(dtype=torch.uint8)
		gtMask[s, fy, fx] = True

	return slices, gtMask

def render_stereo_views(positions, uvs, indices, atlasCpu, maskAtlasCpu,
                         orgImage, orgWidth, orgHeight, outfilledWidth, outfilledHeight,
                         focal, outStereoImages):

	up    = torch.tensor([0.0, 1.0, 0.0])
	scene = renderer.upload_scene(positions, uvs, indices, atlasCpu, (orgWidth, orgHeight), orgImage.convert('RGB'))

	maskScene = None
	if maskAtlasCpu is not None:
		maskScene = renderer.upload_scene(positions, uvs, indices, maskAtlasCpu, (orgWidth, orgHeight))

	fovRender  = 2 * math.atan(orgHeight / (2 * focal))
	projRender = perspective(fovRender, outfilledWidth / outfilledHeight, 0.1, 1000.0)

	viewData = []
	for ipd, path, maskPath in outStereoImages:
		eyeLeft    = torch.tensor([ ipd / 2, 0.0, 0.0])
		targetLeft = torch.tensor([ ipd / 2, 0.0, 1.0])
		viewLeft   = look_at(eyeLeft, targetLeft, up)

		eyeRight    = torch.tensor([-ipd / 2, 0.0, 0.0])
		targetRight = torch.tensor([-ipd / 2, 0.0, 1.0])
		viewRight   = look_at(eyeRight, targetRight, up)

		imgLeft  = renderer.render_view(scene, viewLeft.cpu().numpy(),  projRender.cpu().numpy())
		imgRight = renderer.render_view(scene, viewRight.cpu().numpy(), projRender.cpu().numpy())

		stereo = Image.new(imgLeft.mode, (orgWidth * 2, orgHeight))
		stereo.paste(imgLeft,  (0, 0))
		stereo.paste(imgRight, (orgWidth, 0))

		maskStereo = None
		if maskPath is not None and maskScene is not None:
			maskLeft  = renderer.render_view(maskScene, viewLeft.cpu().numpy(),  projRender.cpu().numpy())
			maskRight = renderer.render_view(maskScene, viewRight.cpu().numpy(), projRender.cpu().numpy())

			maskLeft  = maskLeft.convert('L')
			maskRight = maskRight.convert('L')

			maskStereo = Image.new('L', (orgWidth * 2, orgHeight))
			maskStereo.paste(maskLeft,  (0, 0))
			maskStereo.paste(maskRight, (orgWidth, 0))

		viewData.append((path, stereo, maskPath, maskStereo))

	return viewData, scene, maskScene

def save_outputs(outGLB, glbData, viewData):
	if outGLB is not None:
		atlasCpu, positionsCpu, uvsCpu, indicesCpu = glbData
		exporter.export_glb(atlasCpu, positionsCpu, uvsCpu, indicesCpu, outGLB)

	for path, stereo, maskPath, maskStereo in viewData:
		stereo.save(path, compress_level=1)

		if maskPath is not None and maskStereo is not None:
			maskStereo.save(maskPath, compress_level=1)

# ------------------------------------------- #

def spatial_photo(orgImagePath, plyPath, outGLB, outStereoImages, 
                  greedy=False, opaqueOnly=False, skipExisting=False,
                  saveFuture=None, saveExecutor=None):

	torch.set_default_device('cuda')

	# skip if all outputs already exist:
	# ---------------
	if skipExisting:
		paths = []
		if outGLB is not None:
			paths.append(outGLB)
		if outStereoImages is not None:
			for _, stereoPath, maskPath in outStereoImages:
				if stereoPath is not None:
					paths.append(stereoPath)
				if maskPath is not None:
					paths.append(maskPath)

		if paths and all(os.path.isfile(p) for p in paths):
			print('Skipping: all outputs already exist.')
			return None

	# get dimensions:
	# ---------------
	orgImage = Image.open(orgImagePath)
	orgWidth, orgHeight = orgImage.width, orgImage.height

	outfilledWidth  = math.floor((1 + OUTFILL_AMOUNT) * orgWidth)
	outfilledHeight = math.floor((1 + OUTFILL_AMOUNT) * orgHeight)
	outfilledWidth  = ((outfilledWidth  + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
	outfilledHeight = ((outfilledHeight + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
	aspect = outfilledWidth / outfilledHeight

	# load src:
	# ---------------
	print('Loading source image and ply...')

	gaussians, focalY = load_ply(plyPath)

	fov  = 2 * math.atan(outfilledHeight / (2 * focalY))
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

	# render slices:
	# ---------------
	print('Rendering slices...')

	slices = []
	for i in range(NUM_SLICES):
		with torch.no_grad():
			render = ddgs.render(
				settings,
				*get_slice(gaussians, zMin, zMax, i)
			)

		img = torch.cat([render.color, render.alpha], dim=-1)
		img = (img * 255).to(torch.uint8)

		render.depth[render.depth > zMax] = zMax
		render.depth[(render.depth < zMin) & (render.depth > 0)] = zMin

		sliceT = (slice_t(i) + slice_t(i + 1)) * 0.5
		slicePos = sliceT * (zMax - zMin) + zMin

		slices.append([img, render.depth, slicePos])

	# replace with GT:
	# ---------------
	print('Replacing renders with GT color...')

	slices, gtMasks = replace_gt_color(
		slices, orgImage, 
		outfilledWidth, outfilledHeight, orgWidth, orgHeight
	)

	# generate block atlas:
	# ---------------
	print('Generating block atlas...')

	needsMaskAtlas = outStereoImages is not None and any(maskPath is not None for (_, _, maskPath) in outStereoImages)

	if greedy:
		atlas, placements, maskAtlas = generate_block_atlas_greedy(
			slices,
			gtMasks if needsMaskAtlas else None,
			opaqueOnly
		)
	else:
		atlas, placements, maskAtlas = generate_block_atlas(
			slices,
			gtMasks if needsMaskAtlas else None,
			opaqueOnly
		)

	# build geometry:
	# ---------------
	print('Building geometry...')

	focal = float(proj[1, 1].item() * outfilledHeight / 2)

	zGrid, GW, GH = build_geometry(
		placements, slices,
		outfilledWidth, outfilledHeight,
		focal, aspect
	)

	# compute render buffers:
	# ---------------
	print('Computing render buffers...')

	positions, uvs, indices = finish_geometry(
		zGrid, placements,
		outfilledWidth, outfilledHeight,
		focal, aspect
	)

	atlasCpu     = atlas.cpu().numpy()
	maskAtlasCpu = maskAtlas.cpu().numpy() if maskAtlas is not None else None
	positionsCpu = positions.cpu().numpy()
	uvsCpu       = uvs.cpu().numpy()
	indicesCpu   = indices.cpu().numpy()

	# render stereo views:
	# ---------------
	viewData = []
	if outStereoImages is not None:
		print('Rendering stereo views...')

		viewData, scene, maskScene = render_stereo_views(
			positionsCpu, uvsCpu, indicesCpu, atlasCpu, maskAtlasCpu,
			orgImage, orgWidth, orgHeight, outfilledWidth, outfilledHeight,
			focal, outStereoImages
		)

		renderer.release_scene(scene)
		if maskScene is not None:
			renderer.release_scene(maskScene)

	# dispatch saving on background thread:
	# ---------------
	print('Saving outputs...')
	if outGLB is not None:
		print(f'    - GLB: {outGLB}')
	if viewData:
		for path, _, maskPath, _ in viewData:
			print(f'    - Stereo render: {path}')
			if maskPath is not None:
				print(f'    - Stereo mask render: {maskPath}')

	glbData = (atlasCpu, positionsCpu, uvsCpu, indicesCpu) if outGLB is not None else None
	if glbData is None and not viewData:
		return None

	if saveFuture is not None:
		saveFuture.result()

	ownExecutor = saveExecutor is None
	executor    = saveExecutor or ThreadPoolExecutor(max_workers=1)
	try:
		newFuture = executor.submit(save_outputs, outGLB, glbData, viewData)
		if ownExecutor:
			newFuture.result()
			newFuture = None

	finally:
		if ownExecutor:
			executor.shutdown(wait=False)

	return newFuture

# ------------------------------------------- #

if __name__ == "__main__":
	import argparse
	import re

	# setup argparse:
	# ---------------
	parser = argparse.ArgumentParser(description="Generate spatial photos / stereo image sequences from ML-Sharp outputs")

	parser.add_argument("img", help="Source image file, or a directory containing 'frames/' and 'plys/' subdirectories for sequence mode.")
	parser.add_argument("ply", nargs="?", default=None, help="PLY file (single file mode only; inferred from img path if omitted).")

	parser.add_argument("--start", type=int, default=None, help="First frame index to process, inclusive (sequence mode only).")
	parser.add_argument("--end",   type=int, default=None, help="Last frame index to process, inclusive (sequence mode only).")

	parser.add_argument("--out-glb",    type=str, default=None, help="Output GLB path or directory (sequence mode writes per-frame GLBs here).")
	parser.add_argument("--out-stereo", type=str, default=None, help="Output stereo image path or directory.")
	parser.add_argument("--out-mask",   type=str, default=None, help="Output stereo mask image path or directory (requires --out-stereo).")
	parser.add_argument("--ipd", type=int, nargs="+", default=[56], metavar="MM",
		help="One or more interpupillary distances in millimetres (default: 56). Each IPD is rendered into its own ipd_NNN subdirectory.")
	parser.add_argument("--greedy", action="store_true",
		help="Use greedy-meshed atlas packing instead of the default square atlas.")
	parser.add_argument("--opaque-only", action="store_true",
		help="Whether to only include opaque pixels in the atlas.")
	parser.add_argument("--resume", action="store_true",
		help="Skip frames whose output files already exist on disk.")

	args = parser.parse_args()

	# sequence mode:
	# ---------------
	if os.path.isdir(args.img):
		inputDir  = args.img
		framesDir = os.path.join(inputDir, "frames")
		plysDir   = os.path.join(inputDir, "plys")

		if not os.path.isdir(framesDir):
			parser.error(f"Expected a 'frames/' subdirectory inside '{inputDir}'.")
		if not os.path.isdir(plysDir):
			parser.error(f"Expected a 'plys/' subdirectory inside '{inputDir}'.")

		IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff'}
		frameFiles = sorted(
			f for f in os.listdir(framesDir)
			if os.path.splitext(f)[1].lower() in IMAGE_EXTS
		)

		if not frameFiles:
			parser.error(f"No image files found in '{framesDir}'.")

		if args.start is not None:
			frameFiles = frameFiles[args.start:]
		if args.end is not None:
			frameFiles = frameFiles[:args.end - (args.start or 0) + 1]

		if not frameFiles:
			parser.error(f"No frames remain after applying --start/--end range.")

		with ThreadPoolExecutor(max_workers=1) as saveExecutor:
			failedFiles = []
			pendingSave = None

			for i, frameFile in enumerate(frameFiles):
				stem = os.path.splitext(frameFile)[0]
				orgImagePath = os.path.join(framesDir, frameFile)
				plyPath      = os.path.join(plysDir, stem + ".ply")

				if not os.path.isfile(plyPath):
					print(f"Warning: PLY not found for '{frameFile}', skipping.")
					failedFiles.append(frameFile)

					continue

				print(f'\n--- Processing {frameFile} ({i + 1}/{len(frameFiles)}) ---\n')

				def out_path(base, ext):
					os.makedirs(base, exist_ok=True)
					return os.path.join(base, stem + ext)

				outGLB = out_path(args.out_glb, ".glb") if args.out_glb else None
				outStereoImages = None
				if args.out_stereo:
					outStereoImages = [
						(
							ipdMM / 1000.0,
							out_path(os.path.join(args.out_stereo, f"ipd_{ipdMM:03d}"), ".png"),
							out_path(os.path.join(args.out_mask,   f"ipd_{ipdMM:03d}"), ".png") if args.out_mask else None,
						)
						for ipdMM in args.ipd
					]

				try:
					pendingSave = spatial_photo(
						orgImagePath    = orgImagePath,
						plyPath         = plyPath,
						outGLB          = outGLB,
						outStereoImages = outStereoImages,
						greedy          = args.greedy,
						opaqueOnly      = args.opaque_only,
						skipExisting    = args.resume,
						saveFuture      = pendingSave,
						saveExecutor    = saveExecutor,
					)
				except Exception as e:
					print(f'Failed with exception: {e}')
					failedFiles.append(frameFile)

			if pendingSave is not None:
				pendingSave.result()

			print('\n--- Finished Processing ---\n')
			if failedFiles:
				print(f'Processing failed on entries: {failedFiles}')

	# single file mode:
	# ---------------
	else:
		if not os.path.isfile(args.img):
			parser.error(f"'{args.img}' is not a file or directory.")

		plyPath = args.ply if args.ply else os.path.splitext(args.img)[0] + ".ply"

		if not os.path.isfile(plyPath):
			parser.error(f"PLY file not found: '{plyPath}'. Pass it as a positional argument.")

		base       = os.path.splitext(args.img)[0]
		stereoBase = args.out_stereo if args.out_stereo else base + "_stereo"
		maskBase   = args.out_mask

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
			orgImagePath    = args.img,
			plyPath         = plyPath,
			outGLB          = args.out_glb,
			outStereoImages = [single_stereo_entry(ipdMM) for ipdMM in args.ipd] if args.out_stereo else None,
			greedy          = args.greedy,
			opaqueOnly      = args.opaque_only,
			skipExisting    = args.resume,
		)