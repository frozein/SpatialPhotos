import math
import numpy as np
import torch
import rectpack

from sharp.utils.gaussians import Gaussians3D

if torch.cuda.is_available():
	import ddgs
else:
	import ddgs_cpu as ddgs

import exporter

# ------------------------------------------- #

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEPTH_MIN_QUANTILE = 0.0
DEPTH_MAX_QUANTILE = 0.8

ALPHA_THRESHOLD = 1
ALPHA_SOLID_THRESHOLD = 128

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

UV_PADDING = 1

# ------------------------------------------- #

def look_at(
	eye: torch.Tensor, 
	target: torch.Tensor, 
	up: torch.Tensor
) -> torch.Tensor:
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

def perspective(
	fovy: float, 
	aspect: float, 
	znear: float, 
	zfar: float
) -> torch.Tensor:
	tan_half_fovy = math.tan(fovy / 2)

	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0

	return m

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
	x = x.clamp(0, 1)
	return torch.where(
		x <= 0.0031308,
		12.92 * x,
		1.055 * torch.pow(x, 1 / 2.4) - 0.055,
	)

# ------------------------------------------- #

def slice_t(idx: int, numSlices: int) -> int:
	return (idx / numSlices) * (idx / numSlices)

def get_slice(
	gaussians: Gaussians3D, 
	zMin: float, zMax: float, 
	idx: int, numSlices: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	means = gaussians.mean_vectors.flatten(0, 1)
	scales = gaussians.singular_values.flatten(0, 1)
	rotations = gaussians.quaternions.flatten(0, 1)
	colors = gaussians.colors.flatten(0, 1).unsqueeze(-2)
	opacities = gaussians.opacities.flatten(0, 1).unsqueeze(-1)

	tMin = slice_t(idx, numSlices)
	tMax = slice_t(idx + 1, numSlices)

	zMinSlice = zMin + tMin * (zMax - zMin)
	zMaxSlice = zMin + tMax * (zMax - zMin)

	if (idx == numSlices - 1):
		where = means[:, 2] >= zMinSlice
	elif idx == 0:
		where = means[:, 2] < zMaxSlice
	else:
		where = (means[:, 2] >= zMinSlice) & (means[:, 2] < zMaxSlice)

	return (
		means[where], 
		scales[where], 
		rotations[where], 
		opacities[where], 
		colors[where]
	)

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

def generate_block_atlas(
	slices, 
	blockSize: int, 
	opaqueOnly: bool
):
	mergedMeta = []
	mergedDims = []

	# greedy mesh each slice:
	# ---------------
	for idx, (img, _, _) in enumerate(slices):
		H, W, _ = img.shape
		GH, GW = H // blockSize, W // blockSize

		alpha = img[:GH*blockSize, :GW*blockSize, 3]
		alpha = alpha.reshape(GH, blockSize, GW, blockSize)

		threshold = ALPHA_SOLID_THRESHOLD if opaqueOnly else ALPHA_THRESHOLD
		present = (alpha >= threshold).any(dim=1).any(dim=2)
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
	packer = pack_blocks(mergedDims)

	bin0 = packer.bin_list()[0]
	atlasW, atlasH = bin0

	atlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device=DEVICE)

	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		sliceIdx, srcPx, srcPy, pw, ph, blockCoords = mergedMeta[i]
		img, _, _ = slices[sliceIdx]

		srcPatch = img[srcPy:srcPy+ph, srcPx:srcPx+pw]
		atlas[ay:ay+ah, ax:ax+aw] = srcPatch

		for (gx, gy) in blockCoords:
			bx = gx * blockSize
			by = gy * blockSize

			ox = bx - srcPx
			oy = by - srcPy

			padLeft  = UV_PADDING if (ox == 0)              else 0
			padRight = UV_PADDING if (ox + blockSize >= pw) else 0
			padTop   = UV_PADDING if (oy == 0)              else 0
			padBot   = UV_PADDING if (oy + blockSize >= ph) else 0

			u0 = (ax + ox + padLeft) / atlasW
			v0 = (ay + oy + padTop) / atlasH
			u1 = (ax + ox + blockSize - padRight) / atlasW
			v1 = (ay + oy + blockSize - padBot) / atlasH

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

	return atlas, placements

def fill_block_depths(
	placements, 
	slices, 
	blockSize: int
):

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
	blocks = torch.zeros((N, blockSize, blockSize), dtype=torch.float32, device=DEVICE)
	slicePositions = torch.zeros(N, dtype=torch.float32, device=DEVICE)

	for i, (sliceIdx, px, py) in enumerate(keys):
		_, depth, slicePos = slices[sliceIdx]

		blocks[i] = depth[py:py+blockSize, px:px+blockSize, 0]
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
	).unsqueeze(-1).unsqueeze(-1).expand([-1, blockSize, blockSize])

	filled = blocks.clone()
	filled[invalid] = fillValue[invalid]

	return {keys[i]: filled[i] for i in range(N)}

def build_geometry(placements, slices, width, height, focal, aspect, numSlices: int, blockSize: int):

	# sort blocks by slice idx:
	# ---------------
	placements = sorted(placements, key=lambda x: x[0])
	N = len(placements)

	# infill depth for each block:
	# ---------------
	blockDepths = fill_block_depths(placements, slices, blockSize)

	# average depth at corners, enforce monotonicity:
	# ---------------
	GW = width  // blockSize
	GH = height // blockSize

	zGrid = torch.zeros((numSlices, GH + 1, GW + 1), dtype=torch.float32, device=DEVICE)
	countGrid = torch.zeros((numSlices, GH + 1, GW + 1), dtype=torch.float32, device=DEVICE)

	plSliceIdx = torch.tensor([p[0] for p in placements], dtype=torch.long, device=DEVICE)
	plGx = torch.tensor([p[1] // blockSize for p in placements], dtype=torch.long, device=DEVICE)
	plGy = torch.tensor([p[2] // blockSize for p in placements], dtype=torch.long, device=DEVICE)

	cornerDGx = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
	cornerDGy = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)
	cornerRow = torch.tensor([0, 0, blockSize-1, blockSize-1], dtype=torch.long, device=DEVICE)
	cornerCol = torch.tensor([0, blockSize-1, 0, blockSize-1], dtype=torch.long, device=DEVICE)

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

	return zGrid

def finish_geometry(zGrid, placements, width, height, focal, aspect, numSlices: int, blockSize: int):
	placements = sorted(placements, key=lambda x: x[0], reverse=True)
	N = len(placements)

	GW = width  // blockSize
	GH = height // blockSize

	# enforce monotonicity across slices:
	# ---------------
	zGrid, _ = zGrid.cummax(dim=0)

	# compute worldspace vertex coordinates:
	# ---------------
	gxCoords = torch.arange(GW + 1, device=DEVICE, dtype=torch.float32) * blockSize
	gyCoords = torch.arange(GH + 1, device=DEVICE, dtype=torch.float32) * blockSize

	xOffset = gxCoords - width  * 0.5
	yOffset = height * 0.5 - gyCoords

	xOffset = xOffset.unsqueeze(0).unsqueeze(0).expand(numSlices, GH+1, GW+1)
	yOffset = yOffset.unsqueeze(0).unsqueeze(2).expand(numSlices, GH+1, GW+1)

	xWorld = xOffset * zGrid / focal
	yWorld = yOffset * zGrid / focal

	# compute positions and uvs:
	# ---------------
	plSiceIdx = torch.tensor([p[0] for p in placements], dtype=torch.long,   device=DEVICE)
	plPx      = torch.tensor([p[1] for p in placements], dtype=torch.long,   device=DEVICE)
	plPy      = torch.tensor([p[2] for p in placements], dtype=torch.long,   device=DEVICE)
	plU0      = torch.tensor([p[3] for p in placements], dtype=torch.float32, device=DEVICE)
	plV0      = torch.tensor([p[4] for p in placements], dtype=torch.float32, device=DEVICE)
	plU1      = torch.tensor([p[5] for p in placements], dtype=torch.float32, device=DEVICE)
	plV1      = torch.tensor([p[6] for p in placements], dtype=torch.float32, device=DEVICE)

	gx0 = plPx // blockSize
	gy0 = plPy // blockSize
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

	base = torch.arange(N, device=DEVICE, dtype=torch.int32) * 4
	tri0 = torch.stack([base, base + 1, base + 2], dim=1)
	tri1 = torch.stack([base, base + 2, base + 3], dim=1)
	indices = torch.cat([tri0, tri1], dim=1).reshape(N * 2, 3)

	return positions, uvs, indices

def replace_gt_color(slices, image, outfilledWidth, outfilledHeight, orgWidth, orgHeight):

	orgRGB = torch.tensor(
		np.flip(image, axis=1).copy(),
		device=DEVICE
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

	yy = torch.arange(H, device=DEVICE).unsqueeze(1).expand(H, W)
	xx = torch.arange(W, device=DEVICE).unsqueeze(0).expand(H, W)

	insideGT = (xx >= offX) & (xx < offX + orgWidth) & (yy >= offY) & (yy < offY + orgHeight)

	S = len(slices)
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

	return slices

def save_outputs(outGLB, glbData):
	if outGLB is not None:
		atlasCpu, positionsCpu, uvsCpu, indicesCpu = glbData
		exporter.export_glb(atlasCpu, positionsCpu, uvsCpu, indicesCpu, outGLB)

# ------------------------------------------- #

def spatial_photo(
	image: np.ndarray, 
	gaussians: Gaussians3D, 
	focalY: float, 
	outGLB, 
	numSlices: int = 30,
	blockSize: int = 64,
	outfillAmount: float = 0.0,
	opaqueOnly: bool = False
):

	torch.set_default_device(DEVICE)

	# compute dimensions + settings:
	# ---------------
	orgHeight, orgWidth = image.shape[:2]

	outfilledWidth  = math.floor((1 + outfillAmount) * orgWidth)
	outfilledHeight = math.floor((1 + outfillAmount) * orgHeight)
	outfilledWidth  = ((outfilledWidth  + blockSize - 1) // blockSize) * blockSize
	outfilledHeight = ((outfilledHeight + blockSize - 1) // blockSize) * blockSize
	aspect = outfilledWidth / outfilledHeight

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

	means = gaussians.mean_vectors.flatten(0, 1)
	zMin  = torch.quantile(means[:, 2], DEPTH_MIN_QUANTILE).item()
	zMax  = torch.quantile(means[:, 2], DEPTH_MAX_QUANTILE).item()

	# render slices:
	# ---------------
	print('Rendering slices...')

	slices = []
	for i in range(numSlices):
		with torch.no_grad():
			render = ddgs.render(
				settings,
				*get_slice(
					gaussians=gaussians, 
					zMin=zMin, zMax=zMax, 
					idx=i, numSlices=numSlices
				)
			)

		img = torch.cat([render.color, render.alpha], dim=-1)
		img = (linear_to_srgb(img) * 255).to(torch.uint8)

		render.depth[render.depth > zMax] = zMax
		render.depth[(render.depth < zMin) & (render.depth > 0)] = zMin

		sliceT = (slice_t(i, numSlices) + slice_t(i + 1, numSlices)) * 0.5
		slicePos = sliceT * (zMax - zMin) + zMin

		slices.append([img, render.depth, slicePos])

	# replace with GT:
	# ---------------
	print('Replacing renders with GT color...')

	slices = replace_gt_color(
		slices, image, 
		outfilledWidth, outfilledHeight, orgWidth, orgHeight
	)

	# generate block atlas:
	# ---------------
	print('Generating block atlas...')

	atlas, placements = generate_block_atlas(
		slices, blockSize, opaqueOnly
	)

	# build geometry:
	# ---------------
	print('Building geometry...')

	zGrid = build_geometry(
		placements, slices,
		outfilledWidth, outfilledHeight,
		focalY, aspect, numSlices, blockSize
	)

	# compute render buffers:
	# ---------------
	print('Computing render buffers...')

	positions, uvs, indices = finish_geometry(
		zGrid, placements,
		outfilledWidth, outfilledHeight,
		focalY, aspect, numSlices, blockSize
	)

	atlasCpu     = atlas.cpu().numpy()
	positionsCpu = positions.cpu().numpy()
	uvsCpu       = uvs.cpu().numpy()
	indicesCpu   = indices.cpu().numpy()

	# dispatch saving on background thread:
	# ---------------
	print('Saving outputs...')
	if outGLB is not None:
		print(f'    - GLB: {outGLB}')

	glbData = (atlasCpu, positionsCpu, uvsCpu, indicesCpu) if outGLB is not None else None
	save_outputs(
		outGLB, glbData
	)