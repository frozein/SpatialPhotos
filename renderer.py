import moderngl
import numpy as np
import math
from PIL import Image

ctx = moderngl.create_context(standalone=True)
ctx.enable(moderngl.DEPTH_TEST)

# ... [Shader Code remains the same] ...
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

def render_headless(positions, uvs, indices, atlas_img, output_size, view, proj):
	width, height = output_size

	# ... [Texture Code remains the same] ...
	if not atlas_img.flags['C_CONTIGUOUS']:
		atlas_img = np.ascontiguousarray(atlas_img)
	texture = ctx.texture((atlas_img.shape[1], atlas_img.shape[0]), 4, atlas_img.tobytes())
	texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
	texture.use(0)

	# ---------------------------------------------------------
	# FIX 1: Interleave Data Correctly
	# ---------------------------------------------------------
	# Your input arrays are flat 1D arrays. We must reshape them 
	# to (N, 3) and (N, 2) so hstack combines them per-vertex.
	pos_reshaped = positions.reshape(-1, 3)
	uv_reshaped = uvs.reshape(-1, 2)
	
	# Now this creates shape (N, 5) -> [x, y, z, u, v] per row
	vertex_data = np.hstack([pos_reshaped, uv_reshaped]).astype('f4')
	
	vbo = ctx.buffer(vertex_data.tobytes())
	ibo = ctx.buffer(indices.astype('i4').tobytes())

	vao = ctx.vertex_array(prog, [
		(vbo, '3f 2f', 'in_vert', 'in_text')
	], index_buffer=ibo)

	# ---------------------------------------------------------
	# FIX 2: Matrix Transposition
	# ---------------------------------------------------------
	# Numpy/Torch store matrices in Row-Major order.
	# OpenGL expects Column-Major order. 
	# If you don't transpose, your translation vector [tx, ty, tz] 
	# ends up in the bottom row (projection perspective terms) 
	# causing severe w-warping distortion.
	
	# Ensure they are contiguous float32 before writing
	m_view = view.T.astype('f4').copy() # Transpose!
	m_proj = proj.T.astype('f4').copy() # Transpose!

	prog['m_view'].write(m_view.tobytes())
	prog['m_proj'].write(m_proj.tobytes())

	# ... [Rendering Code remains the same] ...
	fbo = ctx.simple_framebuffer((width, height))
	fbo.use()
	fbo.clear(0.0, 0.0, 0.0, 1.0)
	
	vao.render(moderngl.TRIANGLES)
	
	raw_data = fbo.read(components=4, dtype='f1')
	img = Image.frombytes('RGBA', (width, height), raw_data)
	img = img.transpose(Image.FLIP_TOP_BOTTOM)

	vao.release()
	vbo.release()
	ibo.release()
	texture.release()
	fbo.release()

	return img