import math
import torch
from pathlib import Path

from sharp.utils import io
from sharp.models import (
    PredictorParams,
    create_predictor,
)
from sharp.cli.predict import predict_image

import exporter
from spatial_photos import spatial_photo

# ------------------------------------------- #

CHECKPOINT_PATH = "/Users/daniel/Downloads/sharp.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------- #

def log_error(msg: str):
	print(msg)

def log_info(msg: str):
	print(msg)

# ------------------------------------------- #

def predict(
	inputPath: Path,
	outputPath: Path,
	numSlices: int = 30,
	blockSize: int = 64,
	outfillAmount: float = 0.0,
	opaqueOnly: bool = False
):
	# get list of input images:
	# ---------------
	inputIsFile = inputPath.is_file()
	imagePaths = []
	if inputIsFile:
		if inputPath.suffix in io.get_supported_image_extensions():
			imagePaths = [inputPath]
	else:
		if outputPath.is_file() or outputPath.suffix.lower() == ".glb":
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
	log_info(f"Loading checkpoint from {CHECKPOINT_PATH}")
	stateDict = torch.load(CHECKPOINT_PATH, weights_only=True)

	gaussianPredictor = create_predictor(PredictorParams())
	gaussianPredictor.load_state_dict(stateDict)
	gaussianPredictor.eval()
	gaussianPredictor.to(DEVICE)

	# process each image:
	# ---------------
	singleOutputFile = inputIsFile and outputPath.suffix.lower() == ".glb"
	outputDirectory = outputPath.parent if singleOutputFile else outputPath
	outputDirectory.mkdir(exist_ok=True, parents=True)

	for imagePath in imagePaths:
		log_info(f"Predicting gaussians for {imagePath}")

		image, _, focalY = io.load_rgb(imagePath)
		height, width = image.shape[:2]
		gaussians = predict_image(gaussianPredictor, image, focalY, DEVICE)

		log_info(f"Generating Spatial Photo for {imagePath}")

		outputWidth = math.floor((1 + outfillAmount) * width)
		outputHeight = math.floor((1 + outfillAmount) * height)
		outputWidth = ((outputWidth + blockSize - 1) // blockSize) * blockSize
		outputHeight = ((outputHeight + blockSize - 1) // blockSize) * blockSize

		atlas, vertices = spatial_photo(
			image=image, 
			gaussians=gaussians, 
			focalY=focalY,
			outputWidth=outputWidth,
			outputHeight=outputHeight,
			numSlices=numSlices,
			blockSize=blockSize,
			opaqueOnly=opaqueOnly,
		)

		outGLB = outputPath if singleOutputFile else outputPath / f"{imagePath.stem}.glb"
		log_info(f"Saving Spatial Photo to {outGLB}")

		exporter.export_glb(
			atlas=atlas,
			vertices=vertices,
			imageWidth=outputWidth,
			imageHeight=outputHeight,
			focal=focalY,
			blockSize=blockSize,
			outPath=outGLB,
		)

predict(
	Path("/Users/daniel/Downloads/test_scaled.png"),
	Path("/Users/daniel/Downloads")
)
