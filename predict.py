import math
import argparse
import torch
from pathlib import Path

from sharp.utils import io
from sharp.models import (
    PredictorParams,
    create_predictor,
)
from sharp.cli.predict import predict_image

import exporter
from spatial_photos import ATLAS_MAX_SIZE, spatial_photo

# ------------------------------------------- #

CHECKPOINT_PATH = Path("/Users/daniel/Downloads/sharp.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------- #

def log_error(msg: str) -> None:
	print(msg)

def log_info(msg: str) -> None:
	print(msg)

# ------------------------------------------- #

def predict(
	inputPath: Path,
	outputPath: Path,
	numSlices: int = 30,
	blockSize: int = 64,
	outfillAmount: float = 0.0,
	opaqueOnly: bool = False,
	outputFormat: str = "SPM",
	uvPadding: int = 1,
	checkpointPath: Path = CHECKPOINT_PATH,
) -> None:

	# validate:
	# ---------------
	outputFormat = outputFormat.upper()

	if not inputPath.exists():
		raise FileNotFoundError(f"Input path does not exist: {inputPath}")
	if not 0 < numSlices <= 65536:
		raise ValueError("numSlices must be between 1 and 65536")
	if not 0 < blockSize <= ATLAS_MAX_SIZE:
		raise ValueError(f"blockSize must be between 1 and {ATLAS_MAX_SIZE}")
	if not math.isfinite(outfillAmount) or outfillAmount < 0:
		raise ValueError("outfillAmount must be nonnegative and finite")
	if outputFormat not in ("SPM", "GLB"):
		raise ValueError("outputFormat must be SPM or GLB")
	if not math.isfinite(uvPadding) or uvPadding < 0 or uvPadding * 2 >= blockSize:
		raise ValueError("uvPadding must be nonnegative and less than half the block size")

	# get list of input images:
	# ---------------
	inputIsFile = inputPath.is_file()
	imagePaths: list[Path] = []
	if inputIsFile:
		if inputPath.suffix in io.get_supported_image_extensions():
			imagePaths = [inputPath]
	else:
		if outputPath.is_file() or outputPath.suffix.lower() in (".glb", ".spm"):
			log_error(f"Output path must be a directory when the input is a directory. Input was {inputPath} and output was {outputPath}")
			return

		for ext in io.get_supported_image_extensions():
			imagePaths.extend(list(inputPath.glob(f"**/*{ext}")))

	if len(imagePaths) == 0:
		log_error(f"No valid images found. Input was {inputPath}.")
		return

	log_info(f"Processing {len(imagePaths)} valid image files.")

	# create ml-sharp predictor:
	# ---------------
	log_info(f"Loading checkpoint from {checkpointPath}")
	stateDict = torch.load(checkpointPath, weights_only=True)

	gaussianPredictor = create_predictor(PredictorParams())
	gaussianPredictor.load_state_dict(stateDict)
	gaussianPredictor.eval()
	gaussianPredictor.to(DEVICE)

	# process each image:
	# ---------------
	singleOutputFile = inputIsFile and outputPath.suffix.lower() in (".glb", ".spm")
	if singleOutputFile:
		outputFormat = outputPath.suffix[1:].upper()
	outputDirectory = outputPath.parent if singleOutputFile else outputPath
	outputDirectory.mkdir(exist_ok=True, parents=True)

	for imagePath in imagePaths:
		log_info(f"Predicting gaussians for {imagePath}")

		image, _, focalY = io.load_rgb(imagePath)
		height, width = image.shape[:2]
		if not math.isfinite(focalY) or focalY <= 0:
			raise ValueError(f"Focal length must be positive and finite for {imagePath}")

		outputWidth = math.floor((1 + outfillAmount) * width)
		outputHeight = math.floor((1 + outfillAmount) * height)
		outputWidth = ((outputWidth + blockSize - 1) // blockSize) * blockSize
		outputHeight = ((outputHeight + blockSize - 1) // blockSize) * blockSize
		if max(outputWidth, outputHeight) > 65535:
			raise ValueError("Output image coordinates must fit in uint16")

		gaussians = predict_image(gaussianPredictor, image, focalY, DEVICE)
		if gaussians.mean_vectors.numel() == 0:
			raise ValueError(f"Prediction produced no gaussians for {imagePath}")

		log_info(f"Generating Spatial Photo for {imagePath}")

		atlas, vertices = spatial_photo(
			image=image, 
			gaussians=gaussians, 
			focalY=focalY,
			outputWidth=outputWidth,
			outputHeight=outputHeight,
			numSlices=numSlices,
			blockSize=blockSize,
			opaqueOnly=opaqueOnly,
			atlasBlockLimit=256 if outputFormat == "SPM" else None,
		)

		outFile = outputPath if singleOutputFile else outputPath / f"{imagePath.stem}.{outputFormat.lower()}"
		log_info(f"Saving Spatial Photo to {outFile}")

		exportFunction = exporter.export_spm if outputFormat == "SPM" else exporter.export_glb
		exportFunction(
			atlas=atlas,
			vertices=vertices,
			imageWidth=outputWidth,
			imageHeight=outputHeight,
			focal=focalY,
			blockSize=blockSize,
			outPath=outFile,
			uvPadding=uvPadding,
			**({"numSlices": numSlices, "opaqueOnly": opaqueOnly} if outputFormat == "SPM" else {}),
		)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Convert ML-SHARP predictions into sparse SPM meshes or GLBs.")
	parser.add_argument("input", type=Path, help="Input image or image directory")
	parser.add_argument("output", type=Path, help="Output .spm/.glb file or directory")
	parser.add_argument("--format", choices=("spm", "glb"), default="spm", help="Directory output format (default: spm)")
	parser.add_argument("--slices", type=int, default=30)
	parser.add_argument("--block-size", type=int, default=64)
	parser.add_argument("--outfill", type=float, default=0)
	parser.add_argument("--uv-padding", type=int, default=1)
	parser.add_argument("--opaque-only", action="store_true")
	parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
	args = parser.parse_args()
	
	predict(
		args.input, args.output, numSlices=args.slices, blockSize=args.block_size,
		outfillAmount=args.outfill, opaqueOnly=args.opaque_only, outputFormat=args.format,
		uvPadding=args.uv_padding, checkpointPath=args.checkpoint,
	)
