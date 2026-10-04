import io
import struct
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from PIL import Image

# ------------------------------------------- #

VertexFields: TypeAlias = tuple[Any, Any, Any, Any, Any, Any]
NumpyVertexFields: TypeAlias = tuple[
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
]

SPM_MAGIC = b"SPM\x04"
SPM_HEADER = struct.Struct("<4s4If3If5I")

# ------------------------------------------- #

def to_numpy(array: Any, dtype: Any | None = None) -> np.ndarray:
	if hasattr(array, "detach"):
		array = array.detach().cpu().numpy()

	return np.ascontiguousarray(array, dtype=dtype)

def get_vertex_fields(vertices: VertexFields) -> NumpyVertexFields:
	dtypes = (np.uint16, np.float32, np.uint16, np.uint16, np.uint16, np.uint16)
	fields = tuple(to_numpy(field, dtype) for field, dtype in zip(vertices, dtypes))

	return fields

# ------------------------------------------- #

def split_atlas(atlas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	alpha = atlas[..., 3]
	color = np.zeros(atlas.shape[:2] + (3,), dtype=np.float32)
	np.divide(atlas[..., :3].astype(np.float32) * 255, alpha[..., None],
		out=color, where=alpha[..., None] > 0)

	return np.rint(color).clip(0, 255).astype(np.uint8), alpha

def encode_color(color: np.ndarray, quality: int = 75) -> bytes:
	buffer = io.BytesIO()
	Image.fromarray(color).save(buffer, format="JPEG", quality=quality,
		subsampling=0, optimize=True, progressive=False)
	return buffer.getvalue()

def encode_alpha(alpha: np.ndarray, quality: int = 75) -> bytes:
	buffer = io.BytesIO()
	Image.fromarray(alpha).save(buffer, format="JPEG", quality=quality,
		optimize=True, progressive=False)
	return buffer.getvalue()

def export_spm(
	atlas: Any,
	vertices: VertexFields,
	imageWidth: int,
	imageHeight: int,
	focal: float,
	blockSize: int,
	outPath: str | PathLike[str],
	uvPadding: float = 1,
	numSlices: int | None = None,
	opaqueOnly: bool = False,
	quality: int = 75,
) -> None:

	# unpack fields:
	# ---------------
	atlas = to_numpy(atlas, np.uint8)
	atlasHeight, atlasWidth = atlas.shape[:2]
	sliceIdx, depth, sourceX, sourceY, atlasX, atlasY = get_vertex_fields(vertices)
	sliceIdx, sourceX, sourceY, atlasX, atlasY = [
		field.astype(np.int64).reshape(-1, 4) for field in (sliceIdx, sourceX, sourceY, atlasX, atlasY)
	]
	if numSlices is None:
		numSlices = int(sliceIdx.max()) + 1 if sliceIdx.size else 1

	# compute metadata:
	# ---------------
	gridWidth, gridHeight = imageWidth // blockSize, imageHeight // blockSize
	cornerSlots = (gridWidth + 1) * (gridHeight + 1)
	blockSlots = gridWidth * gridHeight
	cornerKeys = (sliceIdx * cornerSlots + sourceY // blockSize * (gridWidth + 1) + sourceX // blockSize).reshape(-1)
	uniqueKeys, first = np.unique(cornerKeys, return_index=True)
	cornerDepths = depth[first]
	blockKeys = sliceIdx[:, 3] * blockSlots + sourceY[:, 3] // blockSize * gridWidth + sourceX[:, 3] // blockSize
	order = np.argsort(blockKeys)
	blockKeys = blockKeys[order]
	atlasCoords = np.column_stack((atlasX[:, 3], atlasY[:, 3]))[order] // blockSize
	
	# compute bitmasks, write vertex + block lists:
	# ---------------
	chunks: list[bytes] = []
	for s in range(numSlices):
		v0, v1 = np.searchsorted(uniqueKeys, [s * cornerSlots, (s + 1) * cornerSlots])
		b0, b1 = np.searchsorted(blockKeys, [s * blockSlots, (s + 1) * blockSlots])
		cornerMask = np.zeros(cornerSlots, dtype=np.uint8)
		cornerMask[uniqueKeys[v0:v1] - s * cornerSlots] = 1
		blockMask = np.zeros(blockSlots, dtype=np.uint8)
		blockMask[blockKeys[b0:b1] - s * blockSlots] = 1
		chunks.extend((
			struct.pack("<II", v1 - v0, b1 - b0),
			np.packbits(cornerMask, bitorder="little").tobytes(),
			cornerDepths[v0:v1].astype("<f4").tobytes(),
			np.packbits(blockMask, bitorder="little").tobytes(),
			atlasCoords[b0:b1].astype(np.uint8).tobytes(),
		))

	# pack everything together:
	# ---------------
	geometry = b"".join(chunks)

	color, alpha = split_atlas(atlas)
	colorBytes = encode_color(color, quality)
	alphaBytes = encode_alpha(alpha, quality)

	flags = int(bool(opaqueOnly))
	header = SPM_HEADER.pack(
		SPM_MAGIC, imageWidth, imageHeight, numSlices, blockSize,
		focal, atlasWidth, atlasHeight, flags, uvPadding,
		len(uniqueKeys), len(blockKeys), len(geometry), len(colorBytes), len(alphaBytes),
	)

	Path(outPath).write_bytes(header + geometry + colorBytes + alphaBytes)
