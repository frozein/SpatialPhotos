import argparse
import os
import sys
import subprocess
import tempfile
import threading
import replicate
import fal_client

# ------------------------------------------- #

MODEL = "jd7h/propainter:e5ea7ae04e97c96a0e14c70d8e4cb899abdf326a377c01f1c10966ccd6c6bae4"

EYE_WIDTH  = 1280
EYE_HEIGHT = 720

# ------------------------------------------- #

def run_ffmpeg(args: list[str], label: str = "ffmpeg"):
	cmd = ["ffmpeg", "-y"] + args
	print(f"[{label}] Running: {' '.join(cmd)}")
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(f"[{label}] ffmpeg failed:\n{result.stderr}")

def upload_file(src_path: str) -> str:
	print(f"Uploading {src_path}...")
	return fal_client.upload_file(src_path)

def inpaint_video(source_video_path: str, mask_video_path: str, output_path: str):
	for path, label in [(source_video_path, "Source"), (mask_video_path, "Mask")]:
		if not os.path.exists(path):
			raise FileNotFoundError(f"{label} video not found: {path}")

	print(f"Creating prediction for {os.path.basename(output_path)}...")
	prediction = replicate.predictions.create(
		version=MODEL.split(":")[1],
		input={
			"video":              upload_file(source_video_path),
			"mask":               upload_file(mask_video_path),
			"mode":               "video_inpainting",
			"fp16":               True,
			"width":              -1,
			"height":             -1,
			"scale_h":            1,
			"scale_w":            1,
			"save_fps":           24,
			"raft_iter":          20,
			"ref_stride":         10,
			"resize_ratio":       1,
			"mask_dilation":      0,
			"neighbor_length":    10,
			"subvideo_length":    80,
			"return_input_video": False,
		},
	)

	print(f"Prediction ID: {prediction.id} — waiting for result...")
	prediction.wait()

	if prediction.status == "failed":
		raise RuntimeError(f"Prediction failed: {prediction.error}")

	output = prediction.output
	if not output:
		raise ValueError("Replicate returned an empty output.")

	print(f"Inpainting complete. Saving to {output_path}...")
	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

	result = output[0]
	with open(output_path, "wb") as f:
		if hasattr(result, "read"):
			f.write(result.read())
		else:
			import urllib.request
			urllib.request.urlretrieve(str(result), output_path)

	print(f"Saved: {output_path}")
	return output_path

# ------------------------------------------- #

def extract_eyes(color_pattern: str, mask_pattern: str, tmp_dir: str, fps: int, stereo_mode: str):
	"""
	Extract per-eye videos from the SBS color and mask frame sequences.
	When stereo_mode is 'left' or 'right', the ground-truth eye is skipped —
	its color/mask paths are returned as None so the caller can bypass inpainting.
	"""
	# Which eye index is the ground-truth (fixed) eye, if any.
	# fixedEye = None means both eyes are novel (center mode).
	fixedEye = {'center': None, 'left': 'left', 'right': 'right'}[stereo_mode]

	paths = {}
	for eye, x_offset in [("left", 0), ("right", 1)]:
		if eye == fixedEye:
			# Ground-truth eye — no extraction needed.
			paths[eye] = (None, None)
			continue

		color_out = os.path.join(tmp_dir, f"color_{eye}.mp4")
		mask_out  = os.path.join(tmp_dir, f"mask_{eye}.mp4")

		crop_color = f"crop=iw/2:ih:{x_offset}*iw/2:0,scale={EYE_WIDTH}:{EYE_HEIGHT}:flags=lanczos"
		crop_mask  = f"crop=iw/2:ih:{x_offset}*iw/2:0,scale={EYE_WIDTH}:{EYE_HEIGHT}"

		run_ffmpeg([
			"-framerate", str(fps), "-i", color_pattern,
			"-vf", crop_color,
			"-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
			color_out,
		], label=f"extract color {eye}")

		run_ffmpeg([
			"-framerate", str(fps), "-i", mask_pattern,
			"-vf", crop_mask,
			"-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
			mask_out,
		], label=f"extract binary mask {eye}")

		paths[eye] = (color_out, mask_out)

	return (
		paths["left"][0],  paths["right"][0],   # color  (None if GT eye)
		paths["left"][1],  paths["right"][1],   # mask   (None if GT eye)
	)

# ------------------------------------------- #

def composite_v2(
	color_pattern:   str,
	mask_pattern:    str,
	left_inpainted:  str | None,
	right_inpainted: str | None,
	fps:             int,
	output_path:     str,
	full_width:      int,
	full_height:     int,
	stereo_mode:     str,
):
	"""
	Composite both eyes back into an SBS output.

	For ground-truth eyes (left_inpainted / right_inpainted is None) the
	original source crop is used directly — no inpainted overlay is applied
	and the corresponding mask half is ignored.
	"""
	eye_w = full_width  // 2
	eye_h = full_height

	# Build inputs list and index map dynamically so GT eyes don't need a
	# dummy inpainted video.
	# Input 0: color frames  (always)
	# Input 1: mask frames   (always — needed for novel eyes)
	# Input 2+: inpainted eye videos (only for novel eyes)

	ffmpeg_inputs = [
		"-framerate", str(fps), "-i", color_pattern,
		"-framerate", str(fps), "-i", mask_pattern,
	]

	inpainted_idx = {}  # eye -> ffmpeg input index
	next_idx = 2
	for eye, path in [("left", left_inpainted), ("right", right_inpainted)]:
		if path is not None:
			ffmpeg_inputs += ["-i", path]
			inpainted_idx[eye] = next_idx
			next_idx += 1

	# Build filter_complex
	parts = []

	for eye, x_offset in [("left", 0), ("right", 1)]:
		src_crop   = f"crop={eye_w}:{eye_h}:{x_offset * eye_w}:0"
		mask_crop  = f"crop={eye_w}:{eye_h}:{x_offset * eye_w}:0"

		if eye not in inpainted_idx:
			# Ground-truth eye: pass source through untouched.
			parts.append(f"[0:v]{src_crop}[eye_{eye}]")
		else:
			inp_idx = inpainted_idx[eye]
			parts += [
				f"[0:v]{src_crop}[src_{eye}]",
				f"[{inp_idx}:v]scale={eye_w}:{eye_h}:flags=lanczos[inp_{eye}]",
				f"[1:v]{mask_crop},scale={eye_w}:{eye_h}:flags=lanczos,format=gray[mask_{eye}]",
				f"[inp_{eye}][mask_{eye}]alphamerge[inp_{eye}a]",
				f"[src_{eye}][inp_{eye}a]overlay[eye_{eye}]",
			]

	parts.append("[eye_left][eye_right]hstack=inputs=2[out]")

	filter_complex = ";".join(parts)

	run_ffmpeg([
		*ffmpeg_inputs,
		"-filter_complex", filter_complex,
		"-map", "[out]",
		"-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
		output_path,
	], label="composite")

# ------------------------------------------- #

def main():
	parser = argparse.ArgumentParser(
		description="Per-eye stereo video inpainting with ProPainter + composite ffmpeg pass"
	)
	parser.add_argument("--color",       "-c", required=True, help="printf-style pattern for SBS color frames, e.g. frames/%%06d.png")
	parser.add_argument("--mask",        "-b", required=True, help="printf-style pattern for SBS binary mask frames (used for inpainting)")
	parser.add_argument("--output",      "-o", default="output_sbs.mp4", help="Final SBS output path (default: output_sbs.mp4)")
	parser.add_argument("--fps",               type=float, default=23.976, help="Frame rate (default: 24)")
	parser.add_argument("--width",             type=int, default=7680, help="Full SBS frame width (default: 7680 for 2x 3840)")
	parser.add_argument("--height",            type=int, default=2160, help="Full SBS frame height (default: 2160)")
	parser.add_argument("--tmp-dir",           default=None, help="Temp directory for intermediate files (default: auto)")
	parser.add_argument("--keep-tmp",          action="store_true", help="Keep intermediate files after completion")
	parser.add_argument("--stereo-mode", choices=["center", "left", "right"], default="center",
		help="Stereo rendering mode: 'center' inpaints both eyes (default); "
		     "'left' treats the left eye as ground-truth and only inpaints the right; "
		     "'right' treats the right eye as ground-truth and only inpaints the left.")
	args = parser.parse_args()

	# set up temp dir
	auto_tmp = args.tmp_dir is None
	tmp_dir  = args.tmp_dir or tempfile.mkdtemp(prefix="inpaint_stereo_")
	os.makedirs(tmp_dir, exist_ok=True)
	print(f"Using temp directory: {tmp_dir}")

	try:
		# 1. Extract per-eye videos at 720p (GT eye is skipped)
		# ---------------
		print("\n--- Extracting eyes ---\n")
		left_color, right_color, left_mask, right_mask = extract_eyes(
			color_pattern = args.color,
			mask_pattern  = args.mask,
			tmp_dir       = tmp_dir,
			fps           = args.fps,
			stereo_mode   = args.stereo_mode,
		)

		# 2. Run inpainting — only for novel eyes, in parallel where both needed
		# ---------------
		print("\n--- Running inpainting ---\n")
		left_inpainted  = os.path.join(tmp_dir, "inpainted_left.mp4")  if left_color  is not None else None
		right_inpainted = os.path.join(tmp_dir, "inpainted_right.mp4") if right_color is not None else None

		novel_eyes = []
		if left_color  is not None: novel_eyes.append(("left",  left_color,  left_mask,  left_inpainted))
		if right_color is not None: novel_eyes.append(("right", right_color, right_mask, right_inpainted))

		if len(novel_eyes) == 0:
			print("Nothing to inpaint (both eyes are ground-truth — check --stereo-mode).")
			sys.exit(1)

		errors = {}

		def run_eye(eye, source, mask, output):
			try:
				inpaint_video(source, mask, output)
			except Exception as e:
				errors[eye] = e

		if len(novel_eyes) == 1:
			eye, source, mask, output = novel_eyes[0]
			run_eye(eye, source, mask, output)
		else:
			threads = [threading.Thread(target=run_eye, args=entry) for entry in novel_eyes]
			for t in threads: t.start()
			for t in threads: t.join()

		if errors:
			for eye, err in errors.items():
				print(f"Error inpainting {eye} eye: {err}")
			sys.exit(1)

		# 3. Composite both eyes back into SBS output
		# ---------------
		print("\n--- Compositing final output ---\n")
		composite_v2(
			color_pattern   = args.color,
			mask_pattern    = args.mask,
			left_inpainted  = left_inpainted,
			right_inpainted = right_inpainted,
			fps             = args.fps,
			output_path     = args.output,
			full_width      = args.width,
			full_height     = args.height,
			stereo_mode     = args.stereo_mode,
		)

		print(f"\n--- Done. Output: {args.output} ---\n")

	finally:
		if not args.keep_tmp and auto_tmp:
			import shutil
			shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
	main()