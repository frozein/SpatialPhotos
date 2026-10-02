import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch
import rectpack
from rectpack.packer import PackerBBF

from sharp.utils.gaussians import Gaussians3D

if torch.cuda.is_available():
	import ddgs
else:
	import ddgs_cpu as ddgs

# ------------------------------------------- #

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEPTH_MIN_QUANTILE = 0.0
DEPTH_MAX_QUANTILE = 0.8

ALPHA_THRESHOLD = 1
ALPHA_SOLID_THRESHOLD = 128

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

Rectangle: TypeAlias = tuple[int, int, int, int]
PatchDimensions: TypeAlias = tuple[int, int]
SpatialSlice: TypeAlias = tuple[torch.Tensor, torch.Tensor, float]
BlockPlacement: TypeAlias = tuple[int, int, int, int, int]
VertexFields: TypeAlias = tuple[
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
]
GaussianSlice: TypeAlias = tuple[
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
	torch.Tensor,
]

# ------------------------------------------- #

def look_at(
	eye: torch.Tensor, 
	target: torch.Tensor, 
	up: torch.Tensor
) -> torch.Tensor:
	f = (target - eye)
	forwardNorm = torch.norm(f)
	upNorm = torch.norm(up)
	f = f / forwardNorm
	u = up / upNorm
	s = torch.cross(f, u, dim=0)
	sideNorm = torch.norm(s)
	s = s / sideNorm
	u = torch.cross(s, f, dim=0)

	m = torch.eye(4, dtype=torch.float32, device=eye.device)
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

def slice_t(idx: int, numSlices: int) -> float:
	return (idx / numSlices) * (idx / numSlices)

def get_slice(
	gaussians: Gaussians3D, 
	zMin: float, zMax: float, 
	idx: int, numSlices: int
) -> GaussianSlice:
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

def maximal_rectangles(mask: np.ndarray) -> list[Rectangle]:
	h, w = mask.shape
	heights = np.zeros(w, dtype=int)
	rects: list[Rectangle] = []

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

def greedy_mesh(mask: np.ndarray) -> list[Rectangle]:
	mask = mask.copy()
	rectsOut: list[Rectangle] = []

	while np.any(mask):
		rects = maximal_rectangles(mask)
		x, y, w, h = max(rects, key=lambda r: r[2] * r[3])

		rectsOut.append((x, y, w, h))

		mask[y:y+h, x:x+w] = False

	return rectsOut

def pack_blocks(
	patchDims: Sequence[PatchDimensions],
	minSize: int = ATLAS_MIN_SIZE,
	maxSize: int = ATLAS_MAX_SIZE,
) -> PackerBBF:
	minRequiredSize = max(
		max(max(width, height) for width, height in patchDims),
		math.ceil(math.sqrt(sum(width * height for width, height in patchDims))),
	)

	if minRequiredSize > maxSize:
		raise ValueError("Block atlas exceeds the atlas size limit")

	# binary search to find best size:
	# ---------------
	def fits(size: int) -> bool:
		packer = rectpack.newPacker(rotation=False)
		for i, (w, h) in enumerate(patchDims):
			packer.add_rect(w, h, i)

		packer.add_bin(size, size)
		packer.pack()

		return len(packer.rect_list()) == len(patchDims)

	low = max(minSize, minRequiredSize)
	high = maxSize
	bestSize = None

	while low <= high:
		mid = (low + high) // 2
		if fits(mid):
			bestSize = mid
			high = mid - 1
		else:
			low = mid + 1

	if bestSize is None:
		raise ValueError("Could not pack blocks within the atlas size limit")

	# pack using best size:
	# ---------------
	finalPacker = rectpack.newPacker(rotation=False)
	for i, (w, h) in enumerate(patchDims):
		finalPacker.add_rect(w, h, i)

	finalPacker.add_bin(bestSize, bestSize)
	finalPacker.pack()

	return finalPacker

def generate_block_atlas(
	slices: Sequence[SpatialSlice],
	blockSize: int, 
	opaqueOnly: bool,
	atlasBlockLimit: int | None = None,
) -> tuple[torch.Tensor, list[BlockPlacement]]:
	sourceOrigins: list[tuple[int, int, int]] = []
	patchDims: list[PatchDimensions] = []

	# greedy mesh each slice:
	# ---------------
	for sliceIdx, (img, _, _) in enumerate(slices):
		H, W, _ = img.shape
		GH, GW = H // blockSize, W // blockSize

		threshold = ALPHA_SOLID_THRESHOLD if opaqueOnly else ALPHA_THRESHOLD
		alpha = img[:GH*blockSize, :GW*blockSize, 3]
		alphaBlocks = alpha.reshape(GH, blockSize, GW, blockSize)
		presentCpu = (alphaBlocks >= threshold).any(dim=(1, 3)).cpu().numpy()

		if not np.any(presentCpu):
			continue

		for gx, gy, gw, gh in greedy_mesh(presentCpu):
			sourceOrigins.append((sliceIdx, gx * blockSize, gy * blockSize))
			patchDims.append((gw, gh))

	if not patchDims:
		raise ValueError("No visible blocks were found in the rendered slices")

	# pack greedy meshed rects:
	# ---------------
	maxBlocks = ATLAS_MAX_SIZE // blockSize
	if atlasBlockLimit is not None:
		maxBlocks = min(maxBlocks, atlasBlockLimit)
	packer = pack_blocks(
		patchDims,
		minSize=math.ceil(ATLAS_MIN_SIZE / blockSize),
		maxSize=maxBlocks,
	)

	bin0 = packer.bin_list()[0]
	atlasW, atlasH = (size * blockSize for size in bin0)

	atlas = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device=DEVICE)

	blocks: list[BlockPlacement] = []

	for rect in packer.rect_list():
		_, atlasX, atlasY, patchWidth, patchHeight, i = rect
		atlasX, atlasY, patchWidth, patchHeight = (
			value * blockSize for value in (atlasX, atlasY, patchWidth, patchHeight)
		)

		sliceIdx, sourceX, sourceY = sourceOrigins[i]
		img, _, _ = slices[sliceIdx]

		atlas[atlasY:atlasY+patchHeight, atlasX:atlasX+patchWidth] = \
			img[sourceY:sourceY+patchHeight, sourceX:sourceX+patchWidth]

		for blockY in range(0, patchHeight, blockSize):
			for blockX in range(0, patchWidth, blockSize):
				blocks.append((
					sliceIdx,
					sourceX + blockX,
					sourceY + blockY,
					atlasX + blockX,
					atlasY + blockY,
				))

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

	return atlas, blocks

def fill_block_depths(
	blocks: Sequence[BlockPlacement],
	slices: Sequence[SpatialSlice],
	blockSize: int
) -> torch.Tensor:
	N = len(blocks)

	# build list of blocks to process:
	# ---------------
	blockDepths = torch.zeros((N, blockSize, blockSize), dtype=torch.float32, device=DEVICE)
	slicePositions = torch.zeros(N, dtype=torch.float32, device=DEVICE)

	for i, (sliceIdx, sourceX, sourceY, *_) in enumerate(blocks):
		_, depth, slicePos = slices[sliceIdx]

		blockDepths[i] = depth[sourceY:sourceY+blockSize, sourceX:sourceX+blockSize, 0]
		slicePositions[i] = slicePos

	# compute mean depth:
	# ---------------
	valid = blockDepths > 0
	invalid = ~valid
	validCount = valid.sum(dim=(1, 2)).float()

	sumZ  = (blockDepths * valid.float()).sum(dim=(1, 2))
	meanZ = sumZ / validCount.clamp(min=1)

	# infill:
	# ---------------
	fillValue = torch.where(
		validCount == 0,
		slicePositions,
		meanZ
	).unsqueeze(-1).unsqueeze(-1).expand([-1, blockSize, blockSize])

	filled = blockDepths.clone()
	filled[invalid] = fillValue[invalid]

	return filled

def build_depth_grid(
	blocks: Sequence[BlockPlacement],
	slices: Sequence[SpatialSlice],
	imageWidth: int,
	imageHeight: int,
	numSlices: int,
	blockSize: int,
) -> torch.Tensor:
	N = len(blocks)
	gridWidth = imageWidth // blockSize
	gridHeight = imageHeight // blockSize

	# infill depth for each block:
	# ---------------
	blockDepths = fill_block_depths(blocks, slices, blockSize)

	# average depth at corners, enforce monotonicity:
	# ---------------
	zGrid = torch.zeros(
		(numSlices, gridHeight + 1, gridWidth + 1),
		dtype=torch.float32,
		device=DEVICE,
	)
	countGrid = torch.zeros_like(zGrid)

	blockSliceIdx = torch.tensor([block[0] for block in blocks], dtype=torch.long, device=DEVICE)
	blockX = torch.tensor([block[1] // blockSize for block in blocks], dtype=torch.long, device=DEVICE)
	blockY = torch.tensor([block[2] // blockSize for block in blocks], dtype=torch.long, device=DEVICE)

	cornerDX = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=DEVICE)
	cornerDY = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)
	cornerRow = torch.tensor([0, 0, blockSize-1, blockSize-1], dtype=torch.long, device=DEVICE)
	cornerCol = torch.tensor([0, blockSize-1, 0, blockSize-1], dtype=torch.long, device=DEVICE)

	cornerX = blockX.unsqueeze(1) + cornerDX.unsqueeze(0)
	cornerY = blockY.unsqueeze(1) + cornerDY.unsqueeze(0)
	cornerSliceIdx = blockSliceIdx.unsqueeze(1).expand(N, 4)

	values = blockDepths[:, cornerRow, cornerCol]
	valid = values > 0.0

	flatIdx = (
		cornerSliceIdx * (gridHeight + 1) * (gridWidth + 1)
		+ cornerY * (gridWidth + 1)
		+ cornerX
	).reshape(-1)
	flatValues = (values * valid.float()).reshape(-1)
	flatCount = valid.float().reshape(-1)

	zGrid.reshape(-1).scatter_add_(0, flatIdx, flatValues)
	countGrid.reshape(-1).scatter_add_(0, flatIdx, flatCount)

	hasData = countGrid > 0
	zGrid[hasData] = zGrid[hasData] / countGrid[hasData]

	zGrid, _ = zGrid.cummax(dim=0)
	depths = zGrid.reshape(-1)[flatIdx]
	if not torch.all(torch.isfinite(depths) & (depths > 0)):
		raise ValueError("Depth grid generation produced nonpositive or nonfinite depths")

	return zGrid

def build_vertices(
	zGrid: torch.Tensor,
	blocks: Sequence[BlockPlacement],
	blockSize: int,
) -> VertexFields:
	blocks = sorted(blocks, key=lambda block: block[0], reverse=True)
	N = len(blocks)

	sliceIdx = torch.tensor([block[0] for block in blocks], dtype=torch.long, device=DEVICE)
	sourceX = torch.tensor([block[1] for block in blocks], dtype=torch.long, device=DEVICE)
	sourceY = torch.tensor([block[2] for block in blocks], dtype=torch.long, device=DEVICE)
	atlasX = torch.tensor([block[3] for block in blocks], dtype=torch.long, device=DEVICE)
	atlasY = torch.tensor([block[4] for block in blocks], dtype=torch.long, device=DEVICE)

	sourceGridX = sourceX // blockSize
	sourceGridY = sourceY // blockSize
	cornerGridX = torch.stack([sourceGridX, sourceGridX + 1, sourceGridX + 1, sourceGridX], dim=1)
	cornerGridY = torch.stack([sourceGridY + 1, sourceGridY + 1, sourceGridY, sourceGridY], dim=1)
	cornerSliceIdx = sliceIdx.unsqueeze(1).expand(N, 4)
	cornerAtlasX = torch.stack([atlasX, atlasX + blockSize, atlasX + blockSize, atlasX], dim=1)
	cornerAtlasY = torch.stack([atlasY + blockSize, atlasY + blockSize, atlasY, atlasY], dim=1)

	depth = zGrid[cornerSliceIdx, cornerGridY, cornerGridX]
	cornerSourceX = cornerGridX * blockSize
	cornerSourceY = cornerGridY * blockSize

	return (
		cornerSliceIdx.reshape(N * 4).to(torch.uint16),
		depth.reshape(N * 4),
		cornerSourceX.reshape(N * 4).to(torch.uint16),
		cornerSourceY.reshape(N * 4).to(torch.uint16),
		cornerAtlasX.reshape(N * 4).to(torch.uint16),
		cornerAtlasY.reshape(N * 4).to(torch.uint16),
	)

def replace_gt_color(
	slices: list[SpatialSlice],
	image: np.ndarray,
	outfilledWidth: int,
	outfilledHeight: int,
	orgWidth: int,
	orgHeight: int,
) -> list[SpatialSlice]:
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

# ------------------------------------------- #

def spatial_photo(
	image: np.ndarray, 
	gaussians: Gaussians3D, 
	focalY: float,
	outputWidth: int,
	outputHeight: int,
	numSlices: int = 30,
	blockSize: int = 64,
	opaqueOnly: bool = False,
	atlasBlockLimit: int | None = None,
) -> tuple[torch.Tensor, VertexFields]:
	
	torch.set_default_device(DEVICE)

	# compute dimensions + settings:
	# ---------------
	orgHeight, orgWidth = image.shape[:2]
	aspect = outputWidth / outputHeight

	fov  = 2 * math.atan(outputHeight / (2 * focalY))
	eye    = torch.tensor([0.0, 0.0, 0.0])
	target = torch.tensor([0.0, 0.0, 1.0])
	up     = torch.tensor([0.0, 1.0, 0.0])
	view   = look_at(eye, target, up)
	proj   = perspective(fov, aspect, 0.1, 1000.0)

	settings = ddgs.Settings(
		width=outputWidth, height=outputHeight,
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

	slices: list[SpatialSlice] = []
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

		alpha = render.alpha
		rgb = linear_to_srgb(render.color / alpha.clamp_min(1e-8)) * alpha
		img = torch.cat([rgb, alpha], dim=-1)
		img = (img * 255).to(torch.uint8)

		render.depth[render.depth > zMax] = zMax
		render.depth[(render.depth < zMin) & (render.depth > 0)] = zMin

		sliceT = (slice_t(i, numSlices) + slice_t(i + 1, numSlices)) * 0.5
		slicePos = sliceT * (zMax - zMin) + zMin

		slices.append((img, render.depth, slicePos))

	# replace with GT:
	# ---------------
	print('Replacing renders with GT color...')

	slices = replace_gt_color(
		slices, image, 
		outputWidth, outputHeight, orgWidth, orgHeight
	)

	# generate block atlas:
	# ---------------
	print('Generating block atlas...')

	atlas, blocks = generate_block_atlas(
		slices, blockSize, opaqueOnly, atlasBlockLimit
	)

	# build depth grid:
	# ---------------
	print('Building depth grid...')

	zGrid = build_depth_grid(
		blocks, slices,
		outputWidth, outputHeight,
		numSlices, blockSize
	)

	# build format-independent vertices:
	# ---------------
	print('Building vertices...')

	vertices = build_vertices(zGrid, blocks, blockSize)

	return atlas, vertices
