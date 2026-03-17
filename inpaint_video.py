import argparse
import os
import sys
import subprocess
import tempfile

# ------------------------------------------- #

PROPAINTER_SCRIPT = "/home/al/ProPainter/inference_propainter.py"

EYE_WIDTH  = 1280
EYE_HEIGHT = 720

PROCESSING_FRAMERATE = 24

# ------------------------------------------- #

def run_ffmpeg(args: list[str], label: str = "ffmpeg"):
	cmd = ["ffmpeg", "-y"] + args
	print(f"[{label}] Running: {' '.join(cmd)}")

	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"[{label}] ffmpeg failed:\n{result.stderr}")

# ------------------------------------------- #

def inpaint_video(render: str, mask: str, outPath: str):
	outputDir = os.path.dirname(os.path.abspath(outPath))
	stem      = os.path.basename(render.rstrip("/"))

	cmd = [
		"micromamba", "run", "-n", "propainter",
		"python", PROPAINTER_SCRIPT,
		"--video",           render,
		"--mask",            mask,
		"--output",          outputDir,
		"--width",           str(EYE_WIDTH),
		"--height",          str(EYE_HEIGHT),
		"--mode",            "video_inpainting",
		"--fp16",
		"--raft_iter",       "20",
		"--ref_stride",      "10",
		"--neighbor_length", "10",
		"--subvideo_length", "80",
		"--mask_dilation",   "0",
		"--save_fps",        str(PROCESSING_FRAMERATE),
	]

	print(f"[propainter] Running: {' '.join(cmd)}")
	result = subprocess.run(cmd, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"[propainter] Inference failed for {stem}")

	inpaintOut = os.path.join(outputDir, stem, "inpaint_out.mp4")
	if not os.path.exists(inpaintOut):
		raise FileNotFoundError(
			f"Expected ProPainter output not found: {inpaintOut}\n"
			f"Check the output directory for the actual filename."
		)

	os.replace(inpaintOut, outPath)
	return outPath

def extract_eyes(renderPattern: str, maskPattern: str, tmpDir: str, startFrame: int = 1):

	paths = {}
	for eye, xOffset in [("left", 0), ("right", 1)]:
		renderDir = os.path.join(tmpDir, f"color_{eye}")
		maskDir   = os.path.join(tmpDir, f"mask_{eye}")
		os.makedirs(renderDir, exist_ok=True)
		os.makedirs(maskDir  , exist_ok=True)

		renderOut = os.path.join(renderDir, "%06d.png") 
		maskOut   = os.path.join(maskDir, "%06d.png")

		crop = f"crop=iw/2:ih:{xOffset}*iw/2:0"

		run_ffmpeg([
			"-start_number", str(startFrame),
			"-i", renderPattern,
			"-vf", crop,
			renderOut,
		], label=f"extract color {eye}")

		run_ffmpeg([
			"-start_number", str(startFrame),
			"-i", maskPattern,
			"-vf", crop,
			maskOut ,
		], label=f"extract binary mask {eye}")

		paths[eye] = (renderOut, maskOut)

	return (
		paths["left"][0],  paths["right"][0],
		paths["left"][1],  paths["right"][1],
	)

# ------------------------------------------- #

def composite(
	renderPattern:  str,
	maskPattern:    str,
	leftInpainted:  str | None,
	rightInpainted: str | None,
	width:          int,
	height:         int,
	stereoMode:     str,
	startFrame:     int,
	outPattern:     str,
):
	eyeWidth = width // 2
	eyeHeight = height

	# add ffmpeg inputs:
	# ---------------
	ffmpegInputs = [
		"-start_number", str(startFrame), 
		"-framerate", str(PROCESSING_FRAMERATE),
		"-i", renderPattern,

		"-start_number", str(startFrame), 
		"-framerate", str(PROCESSING_FRAMERATE),
		"-i", maskPattern
	]

	if leftInpainted:
		ffmpegInputs += [ "-i", leftInpainted ]
	if rightInpainted:
		ffmpegInputs += [ "-i", rightInpainted ]

	# build filter:
	# ---------------
	filters = [ f"[1:v]format=gray[mask]" ]

	if stereoMode == 'center':
		filters += [
			f"[2:v]scale={eyeWidth}:{eyeHeight}:flags=lanczos[inp_left]",
			f"[3:v]scale={eyeWidth}:{eyeHeight}:flags=lanczos[inp_right]",
			f"[inp_left][inp_right]hstack=inputs=2[inp]"
		]
	else:
		filters += [
			f"[2:v]scale={eyeWidth}:{eyeHeight}:flags=lanczos[inp]"
		]
	
	filters += [ 
		f"[inp][mask]alphamerge[inp_a]",
		f"[0:v][inp_a]overlay[out]"
	]

	# run command:
	# ---------------
	run_ffmpeg([
		*ffmpegInputs,
		"-filter_complex", ";".join(filters),
		"-map", "[out]",
		outPattern,
	], label="composite")

# ------------------------------------------- #

def main():

	# setup argparse:
	# ---------------
	parser = argparse.ArgumentParser(description="Inpaint side-by-side stereo renders")

	parser.add_argument("render", help="printf-style pattern for the newly-rendered stereo view frames")
	parser.add_argument("mask",   help="printf-style pattern for the newly rendered stereo view disocclusion masks")

	parser.add_argument("--width",       type=int, default=7680, help="Full SBS frame width (default: 7680 for 2x 3840)")
	parser.add_argument("--height",      type=int, default=2160, help="Full SBS frame height (default: 2160)")
	parser.add_argument("--start-frame", type=int, default=1, help="First frame number in the input sequences (default: 1)")

	parser.add_argument("--output", "-o", default="output.mp4", help="printf-style pattern for the final output (default: output.mp4)")
	
	parser.add_argument("--tmp-dir",  default=None, help="Temp directory for intermediate files (default: auto)")
	parser.add_argument("--keep-tmp", action="store_true", help="Keep intermediate files after completion")
	
	parser.add_argument("--stereo-mode", choices=["center", "left", "right"], default="center",
		help="Stereo rendering mode: 'center' inpaints both eyes (default); "
		     "'left' treats the right eye as ground-truth and only inpaints the left; "
		     "'right' treats the left eye as ground-truth and only inpaints the right.")

	args = parser.parse_args()

	# get temporary directory:
	# ---------------
	autoTmp = args.tmp_dir is None
	tmpDir  = args.tmp_dir or tempfile.mkdtemp(prefix="inpaint_stereo_")

	os.makedirs(tmpDir, exist_ok=True)
	print(f"Using temporary directory: {tmpDir}")

	try:

		# extract per-eye sequences:
		# ---------------
		if args.stereo_mode == 'center':
			print("Extracing eyes...")

			leftRender, rightRender, leftMask, rightMask = extract_eyes(
				renderPattern = args.render,
				maskPattern   = args.mask,
				tmpDir        = tmpDir,
				startFrame    = args.start_frame,
			)
		elif args.stereo_mode == 'left':
			leftRender = args.render
			leftMask = args.mask

			rightRender = None
			rightMask = None
		else:
			leftRender = None
			leftMask = None

			rightRender = args.render
			rightMask = args.mask

		# run inpainting:
		# ---------------
		leftInpainted  = os.path.join(tmpDir, "inpainted_left.mp4")  if leftRender  is not None else None
		rightInpainted = os.path.join(tmpDir, "inpainted_right.mp4") if rightRender is not None else None

		novelEyes = []
		if leftRender  is not None: novelEyes.append(("left",  leftRender,  leftMask,  leftInpainted))
		if rightRender is not None: novelEyes.append(("right", rightRender, rightMask, rightInpainted))

		for eye, render, mask, output in novelEyes:
			print(f"Inpainting {eye} eye...")
			inpaint_video(os.path.dirname(render), os.path.dirname(mask), output)

		# composite final output sequence:
		# ---------------
		print("Compositing final output sequence...")

		os.makedirs(os.path.dirname(args.output), exist_ok=True)
		composite(
			renderPattern  = args.render,
			maskPattern    = args.mask,
			leftInpainted  = leftInpainted,
			rightInpainted = rightInpainted,
			width          = args.width,
			height         = args.height,
			stereoMode     = args.stereo_mode,
			startFrame     = args.start_frame,
			outPattern     = args.output,
		)

		print(f"Done. Output: {args.output}")

	finally:
		if not args.keep_tmp and autoTmp:
			import shutil
			shutil.rmtree(tmpDir, ignore_errors=True)

if __name__ == "__main__":
	main()