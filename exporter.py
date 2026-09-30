from PIL import Image
import pygltflib as gltf
import io

# ------------------------------------------- #

IMAGE_FORMAT = "WEBP"

# encoder settings per format. lossless webp is ~30% smaller than png on these
# atlases, method 4 costs a few seconds and is within 1% of method 6.
# exact keeps the rgb under fully transparent texels instead of letting libwebp
# pick whatever compresses best - the atlas is premultiplied, so those texels are
# already black and preserving them is (marginally) smaller as well as exact
IMAGE_OPTIONS = {
	"WEBP": dict(lossless=True, quality=75, method=4, exact=True),
	"PNG":  dict(optimize=True),
}

MIME_TYPES = {
	"WEBP": "image/webp",
	"PNG":  "image/png",
	"JPEG": "image/jpeg",
}

# gltf core only allows png and jpeg, webp textures need this extension. it is
# declared as required, so a viewer without it fails loudly instead of showing
# an untextured mesh
WEBP_EXTENSION = "EXT_texture_webp"

# ------------------------------------------- #

def export_glb(atlas, positions, uvs, indices, outPath):

	# encode image:
	# ---------------
	imgBytes = io.BytesIO()

	atlasImg = Image.fromarray(atlas)
	atlasImg.save(imgBytes, format=IMAGE_FORMAT, **IMAGE_OPTIONS.get(IMAGE_FORMAT, {}))
	imgBytes = imgBytes.getvalue()

	# define GLTF structure:
	# ---------------
	# the image lives in the binary chunk rather than a base64 data uri, which
	# would cost 33% on top of whatever the encoder achieved. bufferViews start
	# on 4 byte boundaries
	imgOffset = positions.nbytes + uvs.nbytes + indices.nbytes
	imgPadding = -imgOffset % 4

	binBlob = (
		positions.tobytes() +
		uvs.tobytes() +
		indices.tobytes() +
		b"\x00" * imgPadding +
		imgBytes
	)

	isWebp = IMAGE_FORMAT == "WEBP"

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
		images=[gltf.Image(bufferView=3, mimeType=MIME_TYPES[IMAGE_FORMAT])],
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
