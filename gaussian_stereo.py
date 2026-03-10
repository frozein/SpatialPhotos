import os
import math
import ddgs
import numpy as np
import torch

from PIL import Image
from plyfile import PlyData
from concurrent.futures import ThreadPoolExecutor

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

def save_stereo(stereoPath, stereo):
	os.makedirs(os.path.dirname(os.path.abspath(stereoPath)), exist_ok=True)
	stereo.save(stereoPath, compress_level=1)

# ------------------------------------------- #

def render_stereo(orgImagePath, plyPath, outStereoImages, saveFutures=None, saveExecutor=None):

	torch.set_default_device('cuda')

	# load src:
	# ---------------
	print('Loading source image and PLY...')

	orgImage = Image.open(orgImagePath)
	orgWidth, orgHeight = orgImage.width, orgImage.height
	aspect = orgWidth / orgHeight

	gaussians, focalY = load_ply(plyPath)

	fov  = 2 * math.atan(orgHeight / (2 * focalY))
	proj = perspective(fov, aspect, 0.1, 1000.0)

	up = torch.tensor([0.0, 1.0, 0.0])

	settings_base = dict(
		width=orgWidth,
		height=orgHeight,
		proj=proj,
		focalX=focalY,
		focalY=focalY,
		outputs=ddgs.RenderOutputs.COLOR,
		debug=False,
	)

	# render:
	# ---------------
	toSave = []
	for ipd, stereoPath, maskPath in outStereoImages:
		print(f'Rendering IPD={ipd*1000:.0f}mm...')

		eyeLeft    = torch.tensor([-ipd / 2, 0.0, 0.0])
		targetLeft = torch.tensor([-ipd / 2, 0.0, 1.0])
		viewLeft   = look_at(eyeLeft, targetLeft, up)

		eyeRight    = torch.tensor([ipd / 2, 0.0, 0.0])
		targetRight = torch.tensor([ipd / 2, 0.0, 1.0])
		viewRight   = look_at(eyeRight, targetRight, up)

		settingsLeft = ddgs.Settings(view=viewLeft, **settings_base)
		with torch.no_grad():
			renderLeft = ddgs.render(settingsLeft, *gaussians)
		imgLeft = (renderLeft.color * 255).to(torch.uint8)
		imgLeft  = torch.flip(imgLeft,  dims=[1])

		settingsRight = ddgs.Settings(view=viewRight, **settings_base)
		with torch.no_grad():
			renderRight = ddgs.render(settingsRight, *gaussians)
		imgRight = (renderRight.color * 255).to(torch.uint8)
		imgRight = torch.flip(imgRight, dims=[1])

		left  = Image.fromarray(imgLeft .cpu().numpy(), 'RGB')
		right = Image.fromarray(imgRight.cpu().numpy(), 'RGB')

		stereo = Image.new('RGB', (orgWidth * 2, orgHeight))
		stereo.paste(left,  (0, 0))
		stereo.paste(right, (orgWidth, 0))

		toSave.append((stereoPath, stereo))

	# wait for the previous frame's saves to finish before dispatching new ones:
	# ---------------
	if saveFutures is not None:
		for f in saveFutures:
			f.result()

	# dispatch saving:
	# ---------------
	print('Saving outputs...')
	for _, path, _ in outStereoImages:
		print(f'    - Stereo render: {path}')

	ownExecutor = saveExecutor is None
	executor    = saveExecutor or ThreadPoolExecutor(max_workers=len(toSave))
	try:
		newFutures = [executor.submit(save_stereo, path, img) for path, img in toSave]
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
	import argparse

	parser = argparse.ArgumentParser(description="Render Gaussian splats directly at stereo eye positions")

	parser.add_argument("img", help="Source image file, or a directory containing 'frames/' and 'plys/' subdirectories for sequence mode.")
	parser.add_argument("ply", nargs="?", default=None, help="PLY file (single file mode only; inferred from img path if omitted).")

	parser.add_argument("--start", type=int, default=None, help="First frame index to process, inclusive (sequence mode only).")
	parser.add_argument("--end",   type=int, default=None, help="Last frame index to process, inclusive (sequence mode only).")

	parser.add_argument("--out-stereo", type=str, default=None, help="Output stereo image path or directory.")
	parser.add_argument("--ipd", type=int, nargs="+", default=[56], metavar="MM",
		help="One or more interpupillary distances in millimetres (default: 56).")
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

		stereoBase = args.out_stereo if args.out_stereo else os.path.join(inputDir, "stereo")

		with ThreadPoolExecutor(max_workers=len(args.ipd)) as saveExecutor:
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

				outStereoImages = []
				for ipdMM in args.ipd:
					stereoDir = os.path.join(stereoBase, f"ipd_{ipdMM:03d}")
					os.makedirs(stereoDir, exist_ok=True)
					stereoOut = os.path.join(stereoDir, stem + ".png")
					outStereoImages.append((ipdMM / 1000.0, stereoOut, None))

				if args.resume and all(os.path.isfile(p) for (_, p, _) in outStereoImages):
					print(f'Skipping {frameFile} (outputs exist).')
					continue

				print(f'\n--- Processing {frameFile} ({i + 1}/{len(frameFiles)}) ---\n')

				try:
					pendingFutures = render_stereo(
						orgImagePath    = orgImagePath,
						plyPath         = plyPath,
						outStereoImages = outStereoImages,
						saveFutures     = pendingFutures,
						saveExecutor    = saveExecutor,
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
		stereoBase = args.out_stereo if args.out_stereo else base + "_stereo"

		outStereoImages = []
		for ipdMM in args.ipd:
			stereoDir = os.path.join(stereoBase, f"ipd_{ipdMM:03d}")
			os.makedirs(stereoDir, exist_ok=True)
			stereoOut = os.path.join(stereoDir, os.path.basename(base) + "_stereo.png")
			outStereoImages.append((ipdMM / 1000.0, stereoOut, None))

		render_stereo(
			orgImagePath    = args.img,
			plyPath         = plyPath,
			outStereoImages = outStereoImages,
		)