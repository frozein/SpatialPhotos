import io
from os import PathLike
from typing import Any, TypeAlias

import numpy as np
import pygltflib as gltf
from PIL import Image

# ------------------------------------------- #

IMAGE_OPTIONS = {
	"WEBP": dict(lossless=True, quality=75, method=4, exact=True),
	"PNG":  dict(optimize=True),
}
MIME_TYPES = {
	"WEBP": "image/webp",
	"PNG":  "image/png",
	"JPEG": "image/jpeg",
}
WEBP_EXTENSION = "EXT_texture_webp"

VertexFields: TypeAlias = tuple[Any, Any, Any, Any, Any, Any]
NumpyVertexFields: TypeAlias = tuple[
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
]
RenderBuffers: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]

# ------------------------------------------- #

def to_numpy(array: Any, dtype: Any | None = None) -> np.ndarray:
	if hasattr(array, "detach"):
		array = array.detach().cpu().numpy()

	return np.ascontiguousarray(array, dtype=dtype)

def get_vertex_fields(vertices: VertexFields) -> NumpyVertexFields:
	dtypes = (np.uint16, np.float32, np.uint16, np.uint16, np.uint16, np.uint16)
	fields = tuple(to_numpy(field, dtype) for field, dtype in zip(vertices, dtypes))

	return fields

def get_padded_atlas_coordinates(
	sliceIdx: np.ndarray,
	sourceX: np.ndarray,
	sourceY: np.ndarray,
	atlasX: np.ndarray,
	atlasY: np.ndarray,
	blockSize: int,
	uvPadding: int,
) -> np.ndarray:
	atlasCoordinates = np.empty((len(atlasX), 2), dtype=np.float32)
	atlasCoordinates[:, 0] = atlasX
	atlasCoordinates[:, 1] = atlasY
	atlasCoordinates = atlasCoordinates.reshape(-1, 4, 2)

	if uvPadding == 0:
		return atlasCoordinates.reshape(-1, 2)

	blockKeys = np.column_stack([
		sliceIdx.reshape(-1, 4)[:, 0],
		sourceX.reshape(-1, 4)[:, 3],
		sourceY.reshape(-1, 4)[:, 3],
		atlasX.reshape(-1, 4)[:, 3],
		atlasY.reshape(-1, 4)[:, 3],
	]).astype(np.int64)
	blockKeySet = {tuple(key) for key in blockKeys}

	for i, (sliceIdx, sourceX, sourceY, atlasX, atlasY) in enumerate(blockKeys):
		if (
			sliceIdx, sourceX - blockSize, sourceY, atlasX - blockSize, atlasY
		) not in blockKeySet:
			atlasCoordinates[i, [0, 3], 0] += uvPadding
		if (
			sliceIdx, sourceX + blockSize, sourceY, atlasX + blockSize, atlasY
		) not in blockKeySet:
			atlasCoordinates[i, [1, 2], 0] -= uvPadding
		if (
			sliceIdx, sourceX, sourceY - blockSize, atlasX, atlasY - blockSize
		) not in blockKeySet:
			atlasCoordinates[i, [2, 3], 1] += uvPadding
		if (
			sliceIdx, sourceX, sourceY + blockSize, atlasX, atlasY + blockSize
		) not in blockKeySet:
			atlasCoordinates[i, [0, 1], 1] -= uvPadding

	return atlasCoordinates.reshape(-1, 2)

def build_render_buffers(
	atlas: Any,
	vertices: VertexFields,
	imageWidth: int,
	imageHeight: int,
	focal: float,
	blockSize: int,
	uvPadding: int,
) -> RenderBuffers:

	# unpack vertex fields:
	# ---------------
	sliceIdx, depth, sourceX, sourceY, atlasX, atlasY = get_vertex_fields(vertices)

	# convert src pixel coordinates -> NDC:
	# ---------------
	positions = np.empty((len(depth), 3), dtype=np.float32)
	positions[:, 0] = sourceX
	positions[:, 0] -= imageWidth * 0.5
	positions[:, 0] *= depth
	positions[:, 0] /= focal
	positions[:, 1] = imageHeight * 0.5
	positions[:, 1] -= sourceY
	positions[:, 1] *= depth
	positions[:, 1] /= focal
	positions[:, 2] = depth

	# pad atlas coords:
	# ---------------
	atlasHeight, atlasWidth = atlas.shape[:2]
	uvs = get_padded_atlas_coordinates(
		sliceIdx,
		sourceX,
		sourceY,
		atlasX,
		atlasY,
		blockSize,
		uvPadding,
	)
	uvs[:, 0] /= atlasWidth
	uvs[:, 1] /= atlasHeight

	# create index buf:
	# ---------------
	quadBase = np.arange(len(depth) // 4, dtype=np.uint32) * 4
	indices = np.stack([
		quadBase,
		quadBase + 1,
		quadBase + 2,
		quadBase,
		quadBase + 2,
		quadBase + 3,
	], axis=1).reshape(-1, 3)

	return positions, uvs, indices

# ------------------------------------------- #

def export_glb(
	atlas: Any,
	vertices: VertexFields,
	imageWidth: int,
	imageHeight: int,
	focal: float,
	blockSize: int,
	outPath: str | PathLike[str],
	uvPadding: int = 1,
	imageFormat: str = "WEBP"
) -> None:

	# validate:
	# ---------------
	if min(imageWidth, imageHeight, blockSize) <= 0:
		raise ValueError("image dimensions and blockSize must be positive")
	if not np.isfinite(focal) or focal <= 0:
		raise ValueError("focal must be positive and finite")
	if uvPadding < 0 or uvPadding * 2 >= blockSize:
		raise ValueError("uvPadding must be nonnegative and less than half the block size")

	# build render buffers:
	# ---------------
	atlas = to_numpy(atlas, np.uint8)
	positions, uvs, indices = build_render_buffers(
		atlas,
		vertices,
		imageWidth,
		imageHeight,
		focal,
		blockSize,
		uvPadding,
	)

	# encode image:
	# ---------------
	imgBytes = io.BytesIO()

	atlasImg = Image.fromarray(atlas)
	atlasImg.save(imgBytes, format=imageFormat, **IMAGE_OPTIONS.get(imageFormat, {}))
	imgBytes = imgBytes.getvalue()

	# define GLTF structure:
	# ---------------
	imgOffset = positions.nbytes + uvs.nbytes + indices.nbytes
	imgPadding = -imgOffset % 4

	binBlob = (
		positions.tobytes() +
		uvs.tobytes() +
		indices.tobytes() +
		b"\x00" * imgPadding +
		imgBytes
	)

	isWebp = imageFormat == "WEBP"

	model = gltf.GLTF2(
		asset=gltf.Asset(version="2.0"),
		extensionsUsed=[WEBP_EXTENSION] if isWebp else [],
		extensionsRequired=[WEBP_EXTENSION] if isWebp else [],
		buffers=[gltf.Buffer(byteLength=len(binBlob))],
		bufferViews=[
			gltf.BufferView(buffer=0, byteOffset=0, byteLength=positions.nbytes, target=gltf.ARRAY_BUFFER),
			gltf.BufferView(buffer=0, byteOffset=positions.nbytes, byteLength=uvs.nbytes, target=gltf.ARRAY_BUFFER),
			gltf.BufferView(buffer=0, byteOffset=positions.nbytes + uvs.nbytes, byteLength=indices.nbytes, target=gltf.ELEMENT_ARRAY_BUFFER),
			gltf.BufferView(buffer=0, byteOffset=imgOffset + imgPadding, byteLength=len(imgBytes))
		],
		accessors=[
			gltf.Accessor(bufferView=0, componentType=gltf.FLOAT, count=len(positions), type="VEC3"),
			gltf.Accessor(bufferView=1, componentType=gltf.FLOAT, count=len(uvs), type="VEC2"),
			gltf.Accessor(bufferView=2, componentType=gltf.UNSIGNED_INT, count=len(indices.flat), type="SCALAR")
		],
		images=[gltf.Image(bufferView=3, mimeType=MIME_TYPES[imageFormat])],
		textures=[
			gltf.Texture(extensions={WEBP_EXTENSION: {"source": 0}}) if isWebp else gltf.Texture(source=0)
		],
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

	model.set_binary_blob(binBlob)

	# save:
	# ---------------
	model.save_binary(outPath)
