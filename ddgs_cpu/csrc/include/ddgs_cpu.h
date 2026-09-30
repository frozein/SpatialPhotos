/* ddgs_cpu.h
 *
 * multithreaded CPU rasterizer for 3D gaussian splats, forward pass only
 *
 * the math is a transcription of the DDGS CUDA renderer (see DDGS/csrc):
 * same covariance projection, same alpha clamps, same front-to-back
 * compositing and the same "depth of the 0.95 transmittance crossing"
 * depth output. the parallel structure is not a transcription - tiles are
 * handed out to worker threads instead of thread blocks, and the per-tile
 * gaussian lists are built and sorted per tile rather than with a global
 * radix sort.
 *
 * nothing needed only by a backward pass is computed or retained.
 */

#ifndef DDGS_CPU_H
#define DDGS_CPU_H

#include <cstdint>

#include "ddgs_cpu_math.h"

//-------------------------------------------//

#define DDGS_CPU_TILE_SIZE 16
#define DDGS_CPU_TILE_LEN (DDGS_CPU_TILE_SIZE * DDGS_CPU_TILE_SIZE)

#define DDGS_CPU_MAX_ALPHA 0.99f
#define DDGS_CPU_MIN_ALPHA (1.0f / 255.0f)
#define DDGS_CPU_TRANSMITTANCE_CUTOFF 0.00001f
#define DDGS_CPU_DEPTH_TRANSMITTANCE_CUTOFF 0.95f

//-------------------------------------------//

struct DDGSCPUgaussians
{
	uint32_t count;

	const float* means;     //(count, 3)
	const float* scales;    //(count, 3)
	const float* rotations; //(count, 4), xyzw
	const float* opacities; //(count, 1)
	const float* harmonics; //(count, k, 3), only the first 3 floats per gaussian are read, as plain rgb
};

struct DDGSCPUsettings
{
	uint32_t width;
	uint32_t height;

	DCmat4 view;
	DCmat4 proj;

	float focalX;
	float focalY;

	bool debug;          //prints per-stage timings
	uint32_t numThreads; //0 = one per hardware thread
};

//-------------------------------------------//

/* renders into caller-owned, zero-initialized buffers:
 *   outColor: (height, width, 3), premultiplied by alpha
 *   outAlpha: (height, width, 1)
 *   outDepth: (height, width, 1), positive metric depth, 0 where nothing was hit
 *
 * returns the number of (gaussian, tile) pairs that were rasterized
 */
uint64_t ddgs_cpu_forward(const DDGSCPUsettings& settings, const DDGSCPUgaussians& gaussians,
                          float* outColor, float* outAlpha, float* outDepth);

#endif //#ifndef DDGS_CPU_H
