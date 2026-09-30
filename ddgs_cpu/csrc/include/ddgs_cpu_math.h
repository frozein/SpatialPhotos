/* ddgs_cpu_math.h
 *
 * minimal vector / matrix math for the CPU renderer
 *
 * the conventions here are copied from QuickMath (which the CUDA renderer
 * uses), so that the transcribed splatting math stays identical:
 *   - matrices are column-major, indexed m[column][row]
 *   - dc_matN_mult(a, b) is the standard matrix product a * b
 *   - dc_quat_to_mat3() returns the transpose of the textbook rotation
 *     matrix, exactly like qm_quaternion_to_mat4() does
 */

#ifndef DDGS_CPU_MATH_H
#define DDGS_CPU_MATH_H

#include <cmath>

//-------------------------------------------//

struct DCvec2
{
	float x, y;
};

struct DCvec3
{
	float x, y, z;
};

struct DCvec4
{
	float x, y, z, w;
};

//column-major: m[column][row]
struct DCmat3
{
	float m[3][3];
};

struct DCmat4
{
	float m[4][4];
};

//-------------------------------------------//
//loading:

static inline DCvec3 dc_vec3_load(const float* in)
{
	return { in[0], in[1], in[2] };
}

static inline DCvec4 dc_vec4_load(const float* in)
{
	return { in[0], in[1], in[2], in[3] };
}

//loads a 4x4 matrix stored in row-major order (the layout torch gives us)
static inline DCmat4 dc_mat4_load_row_major(const float* in)
{
	DCmat4 result;

	for(int row = 0; row < 4; row++)
	for(int col = 0; col < 4; col++)
		result.m[col][row] = in[row * 4 + col];

	return result;
}

//-------------------------------------------//
//vectors:

static inline float dc_vec2_length(DCvec2 v)
{
	return std::sqrt(v.x * v.x + v.y * v.y);
}

//-------------------------------------------//
//matrices:

static inline DCmat3 dc_mat3_scale(DCvec3 s)
{
	DCmat3 result = {};

	result.m[0][0] = s.x;
	result.m[1][1] = s.y;
	result.m[2][2] = s.z;

	return result;
}

static inline DCmat3 dc_quat_to_mat3(DCvec4 q)
{
	DCmat3 result;

	float x2  = q.y + q.y;
	float y2  = q.z + q.z;
	float z2  = q.w + q.w;
	float xx2 = q.y * x2;
	float xy2 = q.y * y2;
	float xz2 = q.y * z2;
	float yy2 = q.z * y2;
	float yz2 = q.z * z2;
	float zz2 = q.w * z2;
	float sx2 = q.x * x2;
	float sy2 = q.x * y2;
	float sz2 = q.x * z2;

	result.m[0][0] = 1.0f - (yy2 + zz2);
	result.m[0][1] = xy2 - sz2;
	result.m[0][2] = xz2 + sy2;
	result.m[1][0] = xy2 + sz2;
	result.m[1][1] = 1.0f - (xx2 + zz2);
	result.m[1][2] = yz2 - sx2;
	result.m[2][0] = xz2 - sy2;
	result.m[2][1] = yz2 + sx2;
	result.m[2][2] = 1.0f - (xx2 + yy2);

	return result;
}

static inline DCmat3 dc_mat4_top_left(const DCmat4& m)
{
	DCmat3 result;

	for(int col = 0; col < 3; col++)
	for(int row = 0; row < 3; row++)
		result.m[col][row] = m.m[col][row];

	return result;
}

static inline DCmat3 dc_mat3_transpose(const DCmat3& m)
{
	DCmat3 result;

	for(int col = 0; col < 3; col++)
	for(int row = 0; row < 3; row++)
		result.m[col][row] = m.m[row][col];

	return result;
}

static inline DCmat3 dc_mat3_mult(const DCmat3& m1, const DCmat3& m2)
{
	DCmat3 result;

	for(int col = 0; col < 3; col++)
	for(int row = 0; row < 3; row++)
		result.m[col][row] = m1.m[0][row] * m2.m[col][0] +
		                     m1.m[1][row] * m2.m[col][1] +
		                     m1.m[2][row] * m2.m[col][2];

	return result;
}

static inline DCvec4 dc_mat4_mult_vec4(const DCmat4& m, DCvec4 v)
{
	DCvec4 result;

	result.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0] * v.w;
	result.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1] * v.w;
	result.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2] * v.w;
	result.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3] * v.w;

	return result;
}

#endif //#ifndef DDGS_CPU_MATH_H
