import moderngl
import numpy as np
import math
from PIL import Image

# ------------------------------------------- #

ctx = moderngl.create_context(standalone=True)
ctx.enable(moderngl.BLEND)
ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

program = ctx.program(
	vertex_shader='''
		#version 330

		uniform mat4 m_proj;
		uniform mat4 m_view;
		
		in vec3 i_pos;
		in vec2 i_uv;
		out vec2 v_uv;
		
		void main() 
		{
			gl_Position = m_proj * m_view * vec4(i_pos, 1.0);
			v_uv = i_uv;
		}
	''',
	fragment_shader='''
		#version 330

		uniform sampler2D image;
		
		in vec2 v_uv;
		out vec4 o_color;
		
		void main() {
			vec4 color = texture(image, v_uv);
			if(color.a < 1.0 / 255.0)
				discard;

			o_color = vec4(color.rgb * color.a, color.a);
		}
	''',
)

bgProgram = ctx.program(
	vertex_shader='''
		#version 330

		in vec2 i_pos;
		out vec2 v_uv;

		void main() 
		{
			gl_Position = vec4(i_pos, 0.0, 1.0);

			v_uv = i_pos * 0.5 + 0.5;
			v_uv.y = 1.0 - v_uv.y;
		}
	''',
	fragment_shader='''
		#version 330

		uniform sampler2D image;

		in vec2 v_uv;
		out vec4 o_color;

		void main() 
		{
			vec3 color = texture(image, v_uv).rgb;
			o_color = vec4(color, 1.0);
		}
	''',
)

bgQuadVBO = ctx.buffer(np.array([
	-1.0, -1.0,
	 1.0, -1.0,
	-1.0,  1.0,
	 1.0, -1.0,
	 1.0,  1.0,
	-1.0,  1.0,
], dtype='f4').tobytes())
bgQuadVAO = ctx.simple_vertex_array(bgProgram, bgQuadVBO, 'i_pos')

# ------------------------------------------- #

def upload_scene(positions, uvs, indices, atlasImg, output_size, bgImage=None):
	width, height = output_size

	if not atlasImg.flags['C_CONTIGUOUS']:
		atlasImg = np.ascontiguousarray(atlasImg)
	
	texture = ctx.texture((atlasImg.shape[1], atlasImg.shape[0]), 4, atlasImg.tobytes())
	texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
	texture.use(0)

	vertexData = np.hstack([positions.reshape(-1, 3), uvs.reshape(-1, 2)]).astype('f4')
	vbo = ctx.buffer(vertexData.tobytes())
	ibo = ctx.buffer(indices.astype('i4').tobytes())

	vao = ctx.vertex_array(program, [
		(vbo, '3f 2f', 'i_pos', 'i_uv')
	], index_buffer=ibo)

	fbo = ctx.simple_framebuffer((width, height))

	bgTex = None
	if bgImage is not None:
		bg_array = np.array(bgImage.convert('RGB'))
		if not bg_array.flags['C_CONTIGUOUS']:
			bg_array = np.ascontiguousarray(bg_array)

		bgTex = ctx.texture((bg_array.shape[1], bg_array.shape[0]), 3, bg_array.tobytes())
		bgTex.filter = (moderngl.LINEAR, moderngl.LINEAR)

	return {
		'texture': texture,
		'bgTex':  bgTex,
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
	if scene['bgTex'] is not None:
		scene['bgTex'].release()

def render_view(scene, view, proj):
	view = view.T.astype('f4').copy()
	proj = proj.T.astype('f4').copy()

	scene['fbo'].use()
	scene['fbo'].clear(0.0, 0.0, 0.0, 0.0)

	if scene['bgTex'] is not None:
		scene['bgTex'].use(0)
		bgQuadVAO.render(moderngl.TRIANGLES)

	scene['texture'].use(0)

	program['m_view'].write(view.tobytes())
	program['m_proj'].write(proj.tobytes())
	scene['vao'].render(moderngl.TRIANGLES)

	width, height = scene['size']

	rawImg = scene['fbo'].read(components=4, dtype='f1')
	img = Image.frombytes('RGBA', (width, height), rawImg)
	img = img.transpose(Image.FLIP_TOP_BOTTOM)

	return img