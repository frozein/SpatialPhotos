import moderngl
import numpy as np
import math
from PIL import Image

# ------------------------------------------- #

ctx = moderngl.create_context(standalone=True)
ctx.enable(moderngl.DEPTH_TEST)

prog = ctx.program(
	vertex_shader='''
		#version 330
		uniform mat4 m_proj;
		uniform mat4 m_view;
		in vec3 in_vert;
		in vec2 in_text;
		out vec2 v_text;
		void main() {
			gl_Position = m_proj * m_view * vec4(in_vert, 1.0);
			v_text = in_text;
		}
	''',
	fragment_shader='''
		#version 330
		uniform sampler2D Texture;
		in vec2 v_text;
		out vec4 f_color;
		void main() {
			f_color = texture(Texture, v_text);
			if(f_color.a < 1.0)
				discard;
		}
	''',
)

# ------------------------------------------- #

def upload_scene(positions, uvs, indices, atlasImg, output_size):
	width, height = output_size

	if not atlasImg.flags['C_CONTIGUOUS']:
		atlasImg = np.ascontiguousarray(atlasImg)
	
	texture = ctx.texture((atlasImg.shape[1], atlasImg.shape[0]), 4, atlasImg.tobytes())
	texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
	texture.use(0)

	vertexData = np.hstack([positions.reshape(-1, 3), uvs.reshape(-1, 2)]).astype('f4')
	vbo = ctx.buffer(vertexData.tobytes())
	ibo = ctx.buffer(indices.astype('i4').tobytes())

	vao = ctx.vertex_array(prog, [
		(vbo, '3f 2f', 'in_vert', 'in_text')
	], index_buffer=ibo)

	fbo = ctx.simple_framebuffer((width, height))

	return {
		'texture': texture,
		'vbo':     vbo,
		'ibo':     ibo,
		'vao':     vao,
		'fbo':     fbo,
		'size':    output_size,
	}

def release_scene(scene):
	scene['vao'].release()
	scene['vbo'].release()
	scene['ibo'].release()
	scene['texture'].release()
	scene['fbo'].release()

def render_view(scene, view, proj):
	view = view.T.astype('f4').copy()
	proj = proj.T.astype('f4').copy()

	prog['m_view'].write(view.tobytes())
	prog['m_proj'].write(proj.tobytes())

	scene['fbo'].use()
	scene['fbo'].clear(0.0, 0.0, 0.0, 1.0)
	
	scene['vao'].render(moderngl.TRIANGLES)

	width, height = scene['size']

	rawImg = scene['fbo'].read(components=4, dtype='f1')
	img = Image.frombytes('RGBA', (width, height), rawImg)
	img = img.transpose(Image.FLIP_TOP_BOTTOM)

	return img