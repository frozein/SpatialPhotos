from PIL import Image
import pygltflib as gltf
import io
import base64

# ------------------------------------------- #

def export_glb(atlas, positions, uvs, indices, outPath):

	# encode image to WEBP:
	# ---------------
	imgBytes = io.BytesIO()

	atlasImg = Image.fromarray(atlas)
	atlasImg.save(imgBytes, format="PNG")
	imgBytes = imgBytes.getvalue()

	# define GLTF structure:
	# ---------------
	binBlob = (
		positions.tobytes() +
		uvs.tobytes() +
		indices.tobytes()
	)

	model = gltf.GLTF2(
		asset=gltf.Asset(version="2.0"),
		buffers=[gltf.Buffer(byteLength=len(binBlob))],
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
		images=[gltf.Image(uri="data:image/png;base64," + base64.b64encode(imgBytes).decode())],
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

	model.set_binary_blob(binBlob)
	
	# save:
	# ---------------
	model.save_binary(outPath)