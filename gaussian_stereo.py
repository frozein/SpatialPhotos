import os
import math
import ddgs
import numpy as np
import torch
import argparse

from PIL import Image
from plyfile import PlyData
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------- #

DEPTH_DISOCCLUSION_THRESHOLD = 0.05

# ------------------------------------------- #

def look_at(eye, target, up):
	f = (target - eye)
	f = f / torch.norm(f)
	u = up / torch.norm(up)
	s = torch.cross(u, f, dim=0)
	s = s / torch.norm(s)
	u = torch.cross(f, s, dim=0)

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

def load_ply(path):
	data = PlyData.read(path)
	vertex = data['vertex'].data

	def np_to_torch(name):
		arr = np.stack([vertex[n] for n in name], axis=-1) if isinstance(name, (list, tuple)) else vertex[name]
		return torch.tensor(arr, dtype=torch.float32, device='cuda')

	means     = np_to_torch(['x', 'y', 'z'])
	colors    = 0.5 + np_to_torch(['f_dc_0', 'f_dc_1', 'f_dc_2']) * 0.28209479177387814
	opacities = torch.sigmoid(np_to_torch('opacity').unsqueeze(1))
	scales    = torch.exp(np_to_torch(['scale_0', 'scale_1', 'scale_2']))
	rotations = np_to_torch(['rot_1', 'rot_2', 'rot_3', 'rot_0'])

	numGaussians = means.shape[0]
	colors = colors.reshape((numGaussians, 1, 3))

	gaussians = (means, scales, rotations, opacities, colors)
	focalY = data['intrinsic'].data['intrinsic'][0]

	return gaussians, focalY

def save_image(path, img):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	img.save(path, compress_level=1)

# ------------------------------------------- #

def compute_mask(
	width, height, focalX, focalY,
	viewSource, viewNovel,
	colorSource, colorNovel,
	depthSource, depthNovel,
	eyeSide='right'
):
	cx = width  / 2.0
	cy = height / 2.0

	# transform novel -> source:
	# ---------------
	u = torch.arange(width,  dtype=torch.float32, device='cuda')
	v = torch.arange(height, dtype=torch.float32, device='cuda')
	uu, vv = torch.meshgrid(u, v, indexing='xy')

	z = depthNovel
	x = (uu - cx) / focalX * z
	y = (vv - cy) / focalY * z
	ones = torch.ones_like(z)

	ptsNovel = torch.stack([x, y, z, ones], dim=0).reshape(4, -1)

	novelToSource = viewSource @ torch.inverse(viewNovel)
	ptrSorce = novelToSource @ ptsNovel

	xSrc = ptrSorce[0]
	ySrc = ptrSorce[1]
	zSrc = ptrSorce[2]
	zSrcSafe = zSrc.abs().clamp(min=1e-6)

	# project to pixel space coordinates:
	# ---------------
	uSrcPx = (xSrc / zSrcSafe * focalX + cx).round().long()
	uSrcPy = (ySrc / zSrcSafe * focalY + cy).round().long()
	inBounds = (
		(uSrcPx >= 0) & (uSrcPx < width) &
		(uSrcPy >= 0) & (uSrcPy < height)
	)

	# disocclusion test:
	# ---------------
	disoccluded = torch.ones(width * height, dtype=torch.bool, device='cuda')

	validIdx = inBounds.nonzero(as_tuple=False).squeeze(1)
	uValid = uSrcPx[validIdx]
	vValid = uSrcPy[validIdx]
	zValid = zSrc[validIdx]

	depthSrcSampled = depthSource[vValid, uValid]
	visibleDepth = zValid <= depthSrcSampled * (1.0 + DEPTH_DISOCCLUSION_THRESHOLD)
	disoccluded[validIdx[visibleDepth]] = False

	# combine and create mask:
	# ---------------
	masked = disoccluded.reshape(height, width)
	return masked.to(torch.uint8) * 255

# ------------------------------------------- #

def render_stereo(orgImagePath, plyPath, outRenders, saveFutures=None, saveExecutor=None, stereoMode='center'):

	torch.set_default_device('cuda')

	# load source:
	# ---------------
	print('Loading source image and PLY...')

	orgImage = Image.open(orgImagePath).convert('RGB')
	orgWidth, orgHeight = orgImage.width, orgImage.height
	aspect = orgWidth / orgHeight

	gaussians, focalY = load_ply(plyPath)
	focalX = focalY # square pixels

	fov  = 2 * math.atan(orgHeight / (2 * focalY))
	proj = perspective(fov, aspect, 0.1, 1000.0)

	up = torch.tensor([0.0, 1.0, 0.0])

	eyeSource    = torch.tensor([0.0, 0.0, 0.0])
	targetSource = torch.tensor([0.0, 0.0, 1.0])
	viewSource   = look_at(eyeSource, targetSource, up)
	settingsBase = dict(
		width=orgWidth,
		height=orgHeight,
		proj=proj,
		focalX=focalX,
		focalY=focalY,
		debug=False,
	)

	# render source depth:
	# ---------------
	print('Rendering source depth...')

	settingsSource = ddgs.Settings(
		view    = viewSource,
		outputs = ddgs.RenderOutputs.DEPTH,
		**settingsBase,
	)

	with torch.no_grad():
		renderSource = ddgs.render(settingsSource, *gaussians)

	colorSource = torch.tensor(np.array(orgImage), dtype=torch.uint8, device='cuda')
	depthSource = renderSource.depth.squeeze(-1)

	# render stereo:
	# ---------------
	toSave = []
	for entry in outRenders:
		ipd, renderPath, maskPath = entry
		print(f'Rendering IPD = {ipd*1000:.0f}mm...')

		eyeDist = ipd / 2 if stereoMode == 'center' else ipd
		sides = [
			('left',  torch.tensor([-eyeDist, 0.0, 0.0]), torch.tensor([-eyeDist, 0.0, 1.0])),
			('right', torch.tensor([ eyeDist, 0.0, 0.0]), torch.tensor([ eyeDist, 0.0, 1.0])),
		]
		fixedIdx = {'left': 1, 'right': 0, 'center': None}[stereoMode]
		novelIdx = {'left': 0, 'right': 1, 'center': [0, 1]}[stereoMode]

		# render each side
		eyeSideList  = []
		eyeView      = []
		eyeColor     = []
		eyeDepth     = []
		for i, (side, eye, target) in enumerate(sides):
			view = look_at(eye, target, up)
			eyeSideList.append(side)
			eyeView.append(view)

			if i == fixedIdx:
				eyeColor.append(colorSource)
				eyeDepth.append(depthSource)
			else:
				outputFlags = ddgs.RenderOutputs.COLOR | ddgs.RenderOutputs.DEPTH
				settings = ddgs.Settings(
					view    = view,
					outputs = outputFlags,
					**settingsBase,
				)
				with torch.no_grad():
					render = ddgs.render(settings, *gaussians)

				color = (render.color * 255).to(torch.uint8)
				depth = render.depth.squeeze(-1)

				eyeColor.append(color)
				eyeDepth.append(depth)

		# combine with GT input
		eyeMask     = []
		for i in range(2):
			isFixed = (i == fixedIdx)

			if isFixed or maskPath is None:
				eyeMask.append(None)
			else:
				mask = compute_mask(
					orgWidth, orgHeight, focalX, focalY,
					viewSource, eyeView[i],
					colorSource, eyeColor[i],
					depthSource, eyeDepth[i],
					eyeSide=eyeSideList[i]
				)
				eyeMask.append(mask)

		# stitch into SBS if needed
		if stereoMode == 'center':
			render = Image.new('RGB', (orgWidth * 2, orgHeight))
			render.paste(Image.fromarray(eyeColor[0].cpu().numpy(), 'RGB'), (0,        0))
			render.paste(Image.fromarray(eyeColor[1].cpu().numpy(), 'RGB'), (orgWidth, 0))
		else:
			render = Image.fromarray(eyeColor[novelIdx].cpu().numpy(), 'RGB')

		mask = None
		if maskPath is not None and any(m is not None for m in eyeMask):
			if stereoMode == 'center':
				mask = Image.new('L', (orgWidth * 2, orgHeight))
				mask.paste(Image.fromarray(eyeMask[0].cpu().numpy(), 'L'), (0,        0))
				mask.paste(Image.fromarray(eyeMask[1].cpu().numpy(), 'L'), (orgWidth, 0))
			else:
				mask = Image.fromarray(eyeMask[novelIdx].cpu().numpy(), 'L')

		toSave.append((renderPath, render, maskPath, mask))

	# wait for previous frame's saves:
	# ---------------
	if saveFutures is not None:
		for f in saveFutures:
			f.result()

	# dispatch saving:
	# ---------------
	print('Saving outputs...')

	futuresArgs = []
	for renderPath, render, maskPath, mask in toSave:
		print(f'    - Render:    {renderPath}')
		futuresArgs.append((renderPath, render))

		if maskPath is not None and mask is not None:
			print(f'    - Mask:      {maskPath}')
			futuresArgs.append((maskPath, mask))

	ownExecutor = saveExecutor is None
	executor    = saveExecutor or ThreadPoolExecutor(max_workers=len(futuresArgs))
	try:
		newFutures = [executor.submit(save_image, path, img) for path, img in futuresArgs]
		if ownExecutor:
			for f in newFutures:
				f.result()
			newFutures = None
	finally:
		if ownExecutor:
			executor.shutdown(wait=False)

	return newFutures

# ------------------------------------------- #

if __name__ == "__main__":

	# setup argparse:
	# ---------------
	parser = argparse.ArgumentParser(description="Render Gaussian splats directly at stereo eye positions")

	parser.add_argument("img", help="Source image file, or a directory containing 'frames/' and 'plys/' subdirectories for sequence mode.")
	parser.add_argument("ply", nargs="?", default=None, help="PLY file (single file mode only; inferred from img path if omitted).")

	parser.add_argument("--start", type=int, default=None, help="First frame index to process, inclusive (sequence mode only).")
	parser.add_argument("--end",   type=int, default=None, help="Last frame index to process, inclusive (sequence mode only).")

	parser.add_argument("--out-renders", type=str, default=None, help="Output render directory. Default: <input>/renders/")
	parser.add_argument("--out-masks",   type=str, default=None, help="Output mask directory. Default: <input>/masks/")

	parser.add_argument("--ipd", type=int, nargs="+", default=[56], metavar="MM",
		help="One or more IPDs in millimetres (default: 56).")
	parser.add_argument("--resume", action="store_true",
		help="Skip frames whose output files already exist on disk.")
	parser.add_argument("--stereo-mode", choices=['center', 'left', 'right'], default='center',
		help="Stereo rendering mode: 'center' renders both eyes as novel views (default); "
		     "'left' uses the original image for the right eye and renders only the left eye as a novel view; "
		     "'right' uses the original image for the left eye and renders only the right eye as a novel view.")

	args = parser.parse_args()

	def make_out_paths(renderBase, maskBase, ipdMM, stem):
		renderDir = os.path.join(renderBase, f"ipd_{ipdMM:03d}")
		os.makedirs(renderDir, exist_ok=True)
		renderOut = os.path.join(renderDir, stem + ".png")

		if args.out_masks is None:
			return (ipdMM / 1000.0, renderOut, None, None)

		maskDir = os.path.join(maskBase, f"ipd_{ipdMM:03d}")
		os.makedirs(maskDir, exist_ok=True)
		maskOut = os.path.join(maskDir, stem + ".png")

		return (ipdMM / 1000.0, renderOut, maskOut)

	# sequence mode
	# ---------------
	if os.path.isdir(args.img):
		inputDir  = args.img
		framesDir = os.path.join(inputDir, "frames")
		plysDir   = os.path.join(inputDir, "plys")

		if not os.path.isdir(framesDir):
			parser.error(f"Expected a 'frames/' subdirectory inside '{inputDir}'.")
		if not os.path.isdir(plysDir):
			parser.error(f"Expected a 'plys/' subdirectory inside '{inputDir}'.")

		IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}
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

		renderBase = args.out_renders or os.path.join(inputDir, "renders")
		maskBase   = args.out_masks   or os.path.join(inputDir, "masks")

		with ThreadPoolExecutor(max_workers=len(args.ipd) * 3) as saveExecutor:
			failedFiles    = []
			pendingFutures = None

			for i, frameFile in enumerate(frameFiles):
				stem         = os.path.splitext(frameFile)[0]
				orgImagePath = os.path.join(framesDir, frameFile)
				plyPath      = os.path.join(plysDir, stem + ".ply")

				if not os.path.isfile(plyPath):
					print(f"Warning: PLY not found for '{frameFile}', skipping.")
					failedFiles.append(frameFile)
					continue

				outRenders = [
					make_out_paths(renderBase, maskBase, ipdMM, stem)
					for ipdMM in args.ipd
				]

				if args.resume and all(os.path.isfile(p) for (_, p, _, _) in outRenders):
					print(f'Skipping {frameFile} (outputs exist).')
					continue

				print(f'\n--- Processing {frameFile} ({i + 1}/{len(frameFiles)}) ---\n')

				try:
					pendingFutures = render_stereo(
						orgImagePath = orgImagePath,
						plyPath      = plyPath,
						outRenders   = outRenders,
						saveFutures  = pendingFutures,
						saveExecutor = saveExecutor,
						stereoMode   = args.stereo_mode,
					)
				except Exception as e:
					print(f'Failed with exception: {e}')
					failedFiles.append(frameFile)

			if pendingFutures is not None:
				for f in pendingFutures:
					f.result()

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
			parser.error(f"PLY file not found: '{plyPath}'.")

		base       = os.path.splitext(args.img)[0]
		renderBase = args.out_renders or os.path.join(os.path.dirname(args.img), "render")
		maskBase   = args.out_masks   or os.path.join(os.path.dirname(args.img), "mask")

		outRenders = [
			make_out_paths(renderBase, maskBase, ipdMM, os.path.basename(base))
			for ipdMM in args.ipd
		]

		render_stereo(
			orgImagePath = args.img,
			plyPath      = plyPath,
			outRenders   = outRenders,
			stereoMode   = args.stereo_mode,
		)