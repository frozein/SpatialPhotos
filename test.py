import os
import io
import math
import ddgs
import numpy as np
import torch
import base64
import rectpack
import pygltflib as gltf
import imageio.v2 as imageio

from PIL import Image
from tqdm import tqdm
from plyfile import PlyData

# ------------------------------------------- #

OUTFILL_AMOUNT = 0.0

SLICE_MIN_QUANTILE = 0.0
SLICE_MAX_QUANTILE = 0.9
NUM_SLICES = 10

BLOCK_SIZE = 128

REPLACE_ALPHA_THRESHOLD = 0.1
ALPHA_TEST_THRESHOLD = 0.5

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

def get_slice(gaussians, numSlices, idx):
	means, scales, rotations, opacities, colors = gaussians

	zMin = torch.quantile(means[:, 2], SLICE_MIN_QUANTILE)
	zMax = torch.quantile(means[:, 2], SLICE_MAX_QUANTILE)

	sliceSize = (zMax - zMin) / numSlices

	zMinSlice = idx * sliceSize
	zMaxSlice = (idx + 1) * sliceSize

	
	if idx == 0:
		where = means[:, 2] < zMaxSlice
	if idx == numSlices - 1:
		where = means[:, 2] >= zMinSlice
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

		for (px, py, pw, ph, img_block, asdf) in merged:
			mergedBlocks.append((pw, ph, img_block, asdf))
			placementsMeta.append((idx, px, py, pw, ph))

	# pack greedy meshed rects into atlas:
	# ---------------
	print("Packing slices into atlas... ", end='', flush=True)

	packer = rectpack.newPacker(rotation=False)
	for i, (w, h, _, _) in enumerate(mergedBlocks):
		packer.add_rect(w, h, i)

	packer.add_bin(6144, 6144) # TODO: ladder up in size, find smallest size that fits
	packer.pack()

	bin0 = packer.bin_list()[0]
	atlasWidth, atlasHeight = bin0
	atlas = np.zeros((atlasHeight, atlasWidth, 4), dtype=np.uint8)

	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		_, _, img, block_coords = mergedBlocks[i]
		idx, _, _, _, _ = placementsMeta[i]

		atlas[ay:ay+ah, ax:ax+aw] = img

		for gx, gy in block_coords:
			bx = gx * blockSize
			by = gy * blockSize

			ox = (bx - placementsMeta[i][1])
			oy = (by - placementsMeta[i][2])

			u0 = (ax + ox) / atlasWidth
			v0 = (ay + oy) / atlasHeight
			u1 = (ax + ox + blockSize) / atlasWidth
			v1 = (ay + oy + blockSize) / atlasHeight

			placements.append((idx, bx, by, u0, v1, u1, v0))

	atlas[...][atlas[..., 3] < ALPHA_TEST_THRESHOLD * 255] = [0, 0, 0, 0]
	atlas[..., 3][atlas[..., 3] >= ALPHA_TEST_THRESHOLD * 255] = 255

	print('done')

	return atlas, placements

"""
def generate_block_atlas_naive(slices, blockSize, alphaThreshold):

	# get all blocks:
	# ---------------
	blocks = []

	for idx, (img, depth) in enumerate(tqdm(slices, desc="Generating blocks", unit="slice")):
		height, width, _ = img.shape
		for y in range(0, height, blockSize):
			for x in range(0, width, blockSize):
				block = img[y:y+blockSize, x:x+blockSize]
				if block.shape[0] != blockSize or block.shape[1] != blockSize:
					continue
				
				alpha = block[..., 3]
				if np.all(alpha < alphaThreshold * 255.0):
					continue

				blocks.append((idx, x, y, np.flip(block, axis=0)))

	# pack atlas:
	# ---------------
	numCols = int(math.ceil(math.sqrt(len(blocks))))
	numRows = int(math.ceil(len(blocks) / numCols))

	atlasWidth = numCols * blockSize
	atlasHeight = numRows * blockSize
	atlas = np.zeros((atlasHeight, atlasWidth, 4), dtype=np.uint8)

	placements = []

	for i, (idx, x, y, block) in enumerate(tqdm(blocks, desc="Packing block atlas", unit="block")):
		cx = (i % numCols) * blockSize
		cy = (i // numCols) * blockSize
		atlas[cy:cy+blockSize, cx:cx+blockSize] = block

		u0 = cx / atlasWidth
		v0 = cy / atlasHeight
		u1 = (cx + blockSize) / atlasWidth
		v1 = (cy + blockSize) / atlasHeight

		placements.append((idx, x, y, u0, v0, u1, v1))

	atlas[...][atlas[..., 3] < ALPHA_TEST_THRESHOLD * 255] = [0, 0, 0, 0]
	atlas[..., 3][atlas[..., 3] >= ALPHA_TEST_THRESHOLD * 255] = 255

	return atlas, placements
"""

# ------------------------------------------- #

def fill_block_depth(depthBlock, outlierStd=2.0):

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
	inlierMask = np.abs(zs - mean) <= outlierStd * std
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
	filled[~valid] = zEst[~valid]

	return filled


def build_geometry(placements, slices, width, height, blockSize, vFOV, aspect):

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
	def get_position(sliceIdx, px, py):

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

		z = zAccum / count

		frustumHeight = 2 * z * math.tan(vFOV / 2)
		frustumWidth  = frustumHeight * aspect

		x = (px / width - 0.5) * frustumWidth
		y = (0.5 - py / height) * frustumHeight

		return x, y, z

	# construct mesh grid:
	# ---------------
	positions = []
	uvs = []
	indices = []
	idx = 0

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

def save_glb(atlas, positions, uvs, indices, out_path):
	atlas_img = Image.fromarray(atlas)
	img_bytes = io.BytesIO()
	atlas_img.save(img_bytes, format="WEBP", quality=90)
	atlas_img.save("atlas_asdf2.webp")
	img_bytes = img_bytes.getvalue()

	bin_blob = (
		positions.tobytes() +
		uvs.tobytes() +
		indices.tobytes()
	)

	model = gltf.GLTF2(
		asset=gltf.Asset(version="2.0"),
		buffers=[gltf.Buffer(byteLength=len(bin_blob))],
		bufferViews=[
			gltf.BufferView(buffer=0, byteOffset=0, byteLength=positions.nbytes, target=gltf.ARRAY_BUFFER),
			gltf.BufferView(buffer=0, byteOffset=positions.nbytes, byteLength=uvs.nbytes, target=gltf.ARRAY_BUFFER),
			gltf.BufferView(buffer=0, byteOffset=positions.nbytes + uvs.nbytes, byteLength=indices.nbytes, target=gltf.ELEMENT_ARRAY_BUFFER)
		],
		accessors=[
			gltf.Accessor(bufferView=0, componentType=gltf.FLOAT, count=len(positions)//3, type="VEC3"),
			gltf.Accessor(bufferView=1, componentType=gltf.FLOAT, count=len(uvs)//2, type="VEC2"),
			gltf.Accessor(bufferView=2, componentType=gltf.UNSIGNED_INT, count=len(indices), type="SCALAR")
		],
		images=[gltf.Image(uri="data:image/webp;base64," + base64.b64encode(img_bytes).decode())],
		textures=[gltf.Texture(source=0)],
		materials=[gltf.Material(
			pbrMetallicRoughness=gltf.PbrMetallicRoughness(
				baseColorTexture=gltf.TextureInfo(index=0),
				metallicFactor=0.0,
				roughnessFactor=1.0
			),
			alphaMode="BLEND",
			doubleSided=True
		)],
		meshes=[gltf.Mesh(primitives=[gltf.Primitive(
			attributes={"POSITION": 0, "TEXCOORD_0": 1},
			indices=2,
			material=0
		)])],
		nodes=[gltf.Node(mesh=0)],
		scenes=[gltf.Scene(nodes=[0])],
		scene=0
	)

	model.set_binary_blob(bin_blob)
	model.save_binary(out_path)

# ------------------------------------------- #

def write_slices(orgImagePath, plyPath, outPath):
	torch.set_default_device('cuda')

	# load original image:
	# ---------------	
	print('Reading original image... ', end='', flush=True)

	orgImage = Image.open(orgImagePath)
	width = orgImage.width
	height = orgImage.height
	aspect = width / height

	outfilledWidth  = math.floor((1 + OUTFILL_AMOUNT) * width)
	outfilledHeight = math.floor((1 + OUTFILL_AMOUNT) * height)

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
		width=width, height=height,
		view=view, proj=proj,
		focalX=focalX, focalY=focalY,
		outputs=ddgs.RenderOutputs.COLOR | ddgs.RenderOutputs.ALPHA | ddgs.RenderOutputs.DEPTH,
		debug=False
	)

	# render slices:
	# ---------------
	slices = []
	
	for i in tqdm(range(NUM_SLICES), desc='Rendering slices', unit='slice'):
		means, scales, rotations, opacities, colors = get_slice(gaussians, NUM_SLICES, i)

		render = ddgs.render(
			settings,
			means, scales, rotations, opacities, colors
		)
	
		color = render.color.detach().cpu().numpy()
		alpha = render.alpha.detach().cpu().numpy()
		depth = render.depth.detach().cpu().numpy()

		img = np.dstack((color, alpha))
		img = (img * 255).astype(np.uint8)

		img = np.flip(img, 1)
		depth = np.flip(depth, 1)

		slices.append((img, depth))

	# replace pixels where GT data exists:
	# ---------------
	print('Replacing renders with GT color... ', end='', flush=True)

	orgRGB = np.array(orgImage.convert("RGB"), dtype=np.uint8)

	alphaStack = np.stack([
		slices[i][0][..., 3]
		for i in range(NUM_SLICES)
	], axis=0)

	hitMask = alphaStack > (REPLACE_ALPHA_THRESHOLD * 255.0)
	hitAny = hitMask.any(axis=0)
	firstHit = np.argmax(hitMask, axis=0)

	for s in range(NUM_SLICES):
		mask = (firstHit == s) & hitAny
		slices[s][0][mask, :3] = orgRGB[mask]

	print('done')

	# generate geometry:
	# ---------------
	atlas, placements = generate_block_atlas(
		slices,
		BLOCK_SIZE
	)

	positions, uvs, indices = build_geometry(
		placements,
		slices,
		width,
		height,
		BLOCK_SIZE,
		fov,
		aspect
	)

	# save as GLB:
	# ---------------
	print('Writing GLB... ', end='', flush=True)

	save_glb(atlas, positions, uvs, indices, "slices.glb")

	print('done')

def main():
	write_slices("test/input/t3d.png", "test/output/test.ply", "slices/")

if __name__ == "__main__":
	main()