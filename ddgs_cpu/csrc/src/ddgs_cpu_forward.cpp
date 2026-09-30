/* ddgs_cpu_forward.cpp
 *
 * the forward rasterization pass, see ddgs_cpu.h
 */

#include "ddgs_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

//-------------------------------------------//

namespace {

//per-gaussian data produced by the preprocess pass, everything the
//rasterizer needs and nothing else
struct DCsplat
{
	float centerX, centerY;
	float conicX, conicY, conicZ;
	float opacity;
	float colorR, colorG, colorB;
	float depth;         //view space z, negative in front of the camera
	uint32_t depthBits;  //raw bits of depth, used as the sort key
};

//tile bounds, half open. a culled gaussian gets an empty range
struct DCtileRange
{
	uint32_t minX, minY, maxX, maxY;
};

//-------------------------------------------//

static inline uint32_t _ddgs_cpu_ceildivide32(uint32_t a, uint32_t b)
{
	return (a + b - 1) / b;
}

static inline uint32_t _ddgs_cpu_thread_count(uint32_t requested, uint64_t numChunks)
{
	uint32_t threads = requested;
	if(threads == 0)
	{
		threads = std::thread::hardware_concurrency();
		if(threads == 0)
			threads = 1;
	}

	return (uint32_t)std::min<uint64_t>(threads, numChunks == 0 ? 1 : numChunks);
}

/* calls body(begin, end) over chunks of [0, count), handing chunks out
 * dynamically so that uneven work (tiles hold wildly different numbers of
 * gaussians) still spreads evenly across threads
 */
template<typename F>
static void _ddgs_cpu_parallel(uint64_t count, uint64_t chunkSize, uint32_t requestedThreads, F&& body)
{
	if(count == 0)
		return;

	uint64_t numChunks = (count + chunkSize - 1) / chunkSize;
	uint32_t threads = _ddgs_cpu_thread_count(requestedThreads, numChunks);

	if(threads <= 1)
	{
		body(0, count);
		return;
	}

	std::atomic<uint64_t> nextChunk{ 0 };

	auto worker = [&]() {
		while(true)
		{
			uint64_t chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
			if(chunk >= numChunks)
				break;

			uint64_t begin = chunk * chunkSize;
			uint64_t end = std::min(begin + chunkSize, count);

			body(begin, end);
		}
	};

	std::vector<std::thread> pool;
	pool.reserve(threads - 1);

	for(uint32_t i = 0; i < threads - 1; i++)
		pool.emplace_back(worker);

	worker();

	for(std::thread& thread : pool)
		thread.join();
}

//-------------------------------------------//

/* the tile range a gaussian's bounding box covers, transcribed from
 * _ddgs_get_tile_bounds(), including the truncating float to int casts
 */
static inline void _ddgs_cpu_get_tile_bounds(uint32_t width, uint32_t height, DCvec2 pixCenter, float pixRadius,
                                             DCtileRange& outTiles)
{
	int32_t tilesWidth  = (int32_t)_ddgs_cpu_ceildivide32(width , DDGS_CPU_TILE_SIZE);
	int32_t tilesHeight = (int32_t)_ddgs_cpu_ceildivide32(height, DDGS_CPU_TILE_SIZE);

	outTiles.minX = (uint32_t)std::min(std::max((int32_t)((pixCenter.x - pixRadius) / DDGS_CPU_TILE_SIZE), 0), tilesWidth );
	outTiles.minY = (uint32_t)std::min(std::max((int32_t)((pixCenter.y - pixRadius) / DDGS_CPU_TILE_SIZE), 0), tilesHeight);
	outTiles.maxX = (uint32_t)std::min(std::max((int32_t)((pixCenter.x + pixRadius + DDGS_CPU_TILE_SIZE - 1) / DDGS_CPU_TILE_SIZE), 0), tilesWidth );
	outTiles.maxY = (uint32_t)std::min(std::max((int32_t)((pixCenter.y + pixRadius + DDGS_CPU_TILE_SIZE - 1) / DDGS_CPU_TILE_SIZE), 0), tilesHeight);
}

/* projects one gaussian to screen space, returns false if it was culled
 * (transcribed from _ddgs_foward_preprocess_kernel)
 */
static bool _ddgs_cpu_preprocess_one(const DDGSCPUsettings& settings, const DDGSCPUgaussians& gaussians, uint32_t idx,
                                     DCsplat& outSplat, DCtileRange& outTiles)
{
	//find view and clip pos:
	//---------------
	DCvec3 mean = dc_vec3_load(&gaussians.means[idx * 3]);

	DCvec4 camPos = dc_mat4_mult_vec4(
		settings.view,
		DCvec4{ mean.x, mean.y, mean.z, 1.0f }
	);
	DCvec4 clipPos = dc_mat4_mult_vec4(
		settings.proj, camPos
	);

	//cull gaussians out of view:
	//---------------
	float clip = (float)(1.2 * clipPos.w); //double literal, as in the CUDA renderer
	if(clipPos.x >  clip || clipPos.y >  clip || clipPos.z >  clip ||
	   clipPos.x < -clip || clipPos.y < -clip || clipPos.z < -clip)
		return false;

	//compute covariance matrix:
	//---------------
	DCmat3 scaleMat = dc_mat3_scale(dc_vec3_load(&gaussians.scales[idx * 3]));
	DCmat3 rotMat = dc_quat_to_mat3(dc_vec4_load(&gaussians.rotations[idx * 4]));

	DCmat3 M = dc_mat3_mult(scaleMat, rotMat);
	DCmat3 cov = dc_mat3_mult(dc_mat3_transpose(M), M);

	//project covariance matrix to 2D:
	//---------------
	DCmat3 J = {{
		{ -settings.focalX / camPos.z, 0.0f,                        (settings.focalX * camPos.x) / (camPos.z * camPos.z) },
		{ 0.0f,                        -settings.focalY / camPos.z, (settings.focalY * camPos.y) / (camPos.z * camPos.z) },
		{ 0.0f,                        0.0f,                        0.0f                                                 }
	}};

	DCmat3 W = dc_mat3_transpose(dc_mat4_top_left(settings.view));
	DCmat3 T = dc_mat3_mult(W, J);

	DCmat3 cov2d = dc_mat3_mult(
		dc_mat3_transpose(T),
		dc_mat3_mult(cov, T)
	);

	//compute inverse 2d covariance:
	//---------------
	float det = cov2d.m[0][0] * cov2d.m[1][1] - cov2d.m[0][1] * cov2d.m[0][1];
	if(det == 0.0f)
		return false;

	float invDet = 1.0f / det;
	DCvec3 conic = { cov2d.m[1][1] * invDet, -cov2d.m[0][1] * invDet, cov2d.m[0][0] * invDet };

	//compute eigenvalues:
	//---------------
	float midpoint = (cov2d.m[0][0] + cov2d.m[1][1]) / 2.0f;
	float radius = dc_vec2_length(DCvec2{ (cov2d.m[0][0] - cov2d.m[1][1]) / 2.0f, cov2d.m[0][1] });

	float lambda1 = midpoint + radius;
	float lambda2 = midpoint - radius;

	//compute image tiles:
	//---------------
	DCvec2 pixCenter = {
		((clipPos.x / clipPos.w + 1.0f) * 0.5f * settings.width ) - 0.5f,
		((clipPos.y / clipPos.w + 1.0f) * 0.5f * settings.height) - 0.5f
	};

	float pixRadius = std::ceil(3.0f * std::sqrt(std::max(lambda1, lambda2)));

	_ddgs_cpu_get_tile_bounds(
		settings.width, settings.height, pixCenter, pixRadius,
		outTiles
	);

	if(outTiles.minX >= outTiles.maxX || outTiles.minY >= outTiles.maxY)
		return false;

	//write out:
	//---------------
	//NOTE: harmonics are read as plain rgb at a stride of 3, matching the CUDA
	//renderer, which assumes a single (degree 0) band per gaussian
	outSplat.centerX = pixCenter.x;
	outSplat.centerY = pixCenter.y;
	outSplat.conicX = conic.x;
	outSplat.conicY = conic.y;
	outSplat.conicZ = conic.z;
	outSplat.opacity = gaussians.opacities[idx];
	outSplat.colorR = gaussians.harmonics[idx * 3 + 0];
	outSplat.colorG = gaussians.harmonics[idx * 3 + 1];
	outSplat.colorB = gaussians.harmonics[idx * 3 + 2];
	outSplat.depth = camPos.z;

	std::memcpy(&outSplat.depthBits, &outSplat.depth, sizeof(uint32_t));

	return true;
}

/* composites every gaussian binned into one tile, in front to back order
 * (transcribed from _ddgs_forward_splat_kernel)
 */
static void _ddgs_cpu_rasterize_tile(const DDGSCPUsettings& settings, const std::vector<DCsplat>& splats,
                                     const uint64_t* tileEntries, int64_t numToRender, uint32_t tileX, uint32_t tileY,
                                     float* outColor, float* outAlpha, float* outDepth)
{
	//compute pixel bounds:
	//---------------
	uint32_t pixelMinX = tileX * DDGS_CPU_TILE_SIZE;
	uint32_t pixelMinY = tileY * DDGS_CPU_TILE_SIZE;

	uint32_t pixelMaxX = std::min(pixelMinX + DDGS_CPU_TILE_SIZE, settings.width );
	uint32_t pixelMaxY = std::min(pixelMinY + DDGS_CPU_TILE_SIZE, settings.height);

	//pixels outside the image are considered done from the start, matching the
	//threads a partial tile wastes on the GPU
	uint32_t numInside = (pixelMaxX - pixelMinX) * (pixelMaxY - pixelMinY);
	uint32_t numDone = 0;

	//per pixel state:
	//---------------
	float transmittance[DDGS_CPU_TILE_LEN];
	float color[DDGS_CPU_TILE_LEN * 3];
	float depthHit[DDGS_CPU_TILE_LEN];
	bool done[DDGS_CPU_TILE_LEN];

	for(uint32_t i = 0; i < DDGS_CPU_TILE_LEN; i++)
	{
		transmittance[i] = 1.0f;
		color[i * 3 + 0] = 0.0f;
		color[i * 3 + 1] = 0.0f;
		color[i * 3 + 2] = 0.0f;
		depthHit[i] = 0.0f;
		done[i] = false;
	}

	//loop over batches of gaussians until every pixel is done:
	//---------------
	uint32_t numRounds = _ddgs_cpu_ceildivide32((uint32_t)numToRender, DDGS_CPU_TILE_LEN);

	for(uint32_t round = 0; round < numRounds; round++)
	{
		//exit early if all pixels are done
		if(numDone == numInside)
			break;

		int64_t batchLen = std::min((int64_t)DDGS_CPU_TILE_LEN, numToRender);

		for(int64_t j = 0; j < batchLen; j++)
		{
			const DCsplat& splat = splats[(uint32_t)(tileEntries[round * DDGS_CPU_TILE_LEN + j] & 0xFFFFFFFFull)];

			for(uint32_t ty = 0; ty < pixelMaxY - pixelMinY; ty++)
			{
				float dy = splat.centerY - (float)(pixelMinY + ty);

				for(uint32_t tx = 0; tx < pixelMaxX - pixelMinX; tx++)
				{
					float dx = splat.centerX - (float)(pixelMinX + tx);

					float power = -0.5f * (splat.conicX * dx * dx + splat.conicZ * dy * dy) - splat.conicY * dx * dy;
					if(power > 0.0f)
						continue;

					float alpha = std::min(DDGS_CPU_MAX_ALPHA, splat.opacity * std::exp(power));
					if(alpha < DDGS_CPU_MIN_ALPHA)
						continue;

					uint32_t i = ty * DDGS_CPU_TILE_SIZE + tx;

					float accumTransmittance = transmittance[i];
					float newAccumTransmittance = accumTransmittance * (1.0f - alpha);

					if(accumTransmittance >= DDGS_CPU_DEPTH_TRANSMITTANCE_CUTOFF &&
					   newAccumTransmittance < DDGS_CPU_DEPTH_TRANSMITTANCE_CUTOFF)
						depthHit[i] = splat.depth;

					if(newAccumTransmittance < DDGS_CPU_TRANSMITTANCE_CUTOFF)
					{
						if(!done[i])
						{
							done[i] = true;
							numDone++;
						}

						continue;
					}

					float weight = alpha * accumTransmittance;
					color[i * 3 + 0] += splat.colorR * weight;
					color[i * 3 + 1] += splat.colorG * weight;
					color[i * 3 + 2] += splat.colorB * weight;

					transmittance[i] = newAccumTransmittance;
				}
			}
		}

		numToRender -= DDGS_CPU_TILE_LEN;
	}

	//write final color:
	//---------------
	for(uint32_t pixelY = pixelMinY; pixelY < pixelMaxY; pixelY++)
	for(uint32_t pixelX = pixelMinX; pixelX < pixelMaxX; pixelX++)
	{
		uint64_t pixelId = pixelX + (uint64_t)settings.width * pixelY;
		uint32_t i = (pixelY - pixelMinY) * DDGS_CPU_TILE_SIZE + (pixelX - pixelMinX);

		outColor[pixelId * 3 + 0] = color[i * 3 + 0];
		outColor[pixelId * 3 + 1] = color[i * 3 + 1];
		outColor[pixelId * 3 + 2] = color[i * 3 + 2];

		outAlpha[pixelId] = 1.0f - transmittance[i];
		outDepth[pixelId] = -depthHit[i];
	}
}

} //namespace

//-------------------------------------------//

uint64_t ddgs_cpu_forward(const DDGSCPUsettings& settings, const DDGSCPUgaussians& gaussians,
                          float* outColor, float* outAlpha, float* outDepth)
{
	using Clock = std::chrono::high_resolution_clock;

	//validate:
	//---------------
	if(gaussians.count == 0)
		return 0;

	auto tStartTotal = Clock::now();

	uint32_t tilesWidth  = _ddgs_cpu_ceildivide32(settings.width , DDGS_CPU_TILE_SIZE);
	uint32_t tilesHeight = _ddgs_cpu_ceildivide32(settings.height, DDGS_CPU_TILE_SIZE);
	uint64_t numTiles = (uint64_t)tilesWidth * tilesHeight;

	//preprocess:
	//---------------
	auto tStartPreprocess = Clock::now();

	std::vector<DCsplat> splats(gaussians.count);
	std::vector<DCtileRange> tileRanges(gaussians.count);

	_ddgs_cpu_parallel(gaussians.count, 4096, settings.numThreads, [&](uint64_t begin, uint64_t end) {
		for(uint64_t idx = begin; idx < end; idx++)
		{
			if(!_ddgs_cpu_preprocess_one(settings, gaussians, (uint32_t)idx, splats[idx], tileRanges[idx]))
				tileRanges[idx] = { 0, 0, 0, 0 };
		}
	});

	auto tEndPreprocess = Clock::now();

	//count how many gaussians land in each tile:
	//---------------
	auto tStartBinning = Clock::now();

	std::vector<std::atomic<uint32_t>> tileCounts(numTiles);
	for(uint64_t tile = 0; tile < numTiles; tile++)
		tileCounts[tile].store(0, std::memory_order_relaxed);

	_ddgs_cpu_parallel(gaussians.count, 4096, settings.numThreads, [&](uint64_t begin, uint64_t end) {
		for(uint64_t idx = begin; idx < end; idx++)
		{
			const DCtileRange& tiles = tileRanges[idx];

			for(uint32_t y = tiles.minY; y < tiles.maxY; y++)
			for(uint32_t x = tiles.minX; x < tiles.maxX; x++)
				tileCounts[x + (uint64_t)tilesWidth * y].fetch_add(1, std::memory_order_relaxed);
		}
	});

	std::vector<uint64_t> tileOffsets(numTiles + 1);
	uint64_t numRendered = 0;

	for(uint64_t tile = 0; tile < numTiles; tile++)
	{
		tileOffsets[tile] = numRendered;
		numRendered += tileCounts[tile].load(std::memory_order_relaxed);
	}
	tileOffsets[numTiles] = numRendered;

	if(numRendered == 0)
		return 0;

	//bin gaussians into their tiles:
	//---------------
	/* an entry packs the raw bits of the view space depth above the gaussian
	 * index. sorting those ascending puts the nearest gaussian first (negative
	 * floats compare in reverse as unsigned, and everything in front of the
	 * camera has negative z) and breaks ties by gaussian index, which is the
	 * order the GPU's stable radix sort produces
	 */
	std::vector<uint64_t> entries(numRendered);

	for(uint64_t tile = 0; tile < numTiles; tile++)
		tileCounts[tile].store(0, std::memory_order_relaxed);

	_ddgs_cpu_parallel(gaussians.count, 4096, settings.numThreads, [&](uint64_t begin, uint64_t end) {
		for(uint64_t idx = begin; idx < end; idx++)
		{
			const DCtileRange& tiles = tileRanges[idx];
			uint64_t entry = ((uint64_t)splats[idx].depthBits << 32) | (uint32_t)idx;

			for(uint32_t y = tiles.minY; y < tiles.maxY; y++)
			for(uint32_t x = tiles.minX; x < tiles.maxX; x++)
			{
				uint64_t tile = x + (uint64_t)tilesWidth * y;
				uint32_t slot = tileCounts[tile].fetch_add(1, std::memory_order_relaxed);

				entries[tileOffsets[tile] + slot] = entry;
			}
		}
	});

	_ddgs_cpu_parallel(numTiles, 16, settings.numThreads, [&](uint64_t begin, uint64_t end) {
		for(uint64_t tile = begin; tile < end; tile++)
			std::sort(entries.begin() + tileOffsets[tile], entries.begin() + tileOffsets[tile + 1]);
	});

	auto tEndBinning = Clock::now();

	//splat:
	//---------------
	auto tStartSplat = Clock::now();

	_ddgs_cpu_parallel(numTiles, 1, settings.numThreads, [&](uint64_t begin, uint64_t end) {
		for(uint64_t tile = begin; tile < end; tile++)
		{
			_ddgs_cpu_rasterize_tile(
				settings, splats,
				entries.data() + tileOffsets[tile], (int64_t)(tileOffsets[tile + 1] - tileOffsets[tile]),
				(uint32_t)(tile % tilesWidth), (uint32_t)(tile / tilesWidth),
				outColor, outAlpha, outDepth
			);
		}
	});

	auto tEndSplat = Clock::now();
	auto tEndTotal = Clock::now();

	//print timing information:
	//---------------
	if(settings.debug)
	{
		auto ms = [](Clock::time_point start, Clock::time_point end) {
			return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
		};

		std::printf("\nTOTAL FRAME TIME (forwards): %.3fms\n", ms(tStartTotal, tEndTotal));
		std::printf("\t- Preprocessing: %.3fms\n", ms(tStartPreprocess, tEndPreprocess));
		std::printf("\t- Binning:       %.3fms\n", ms(tStartBinning, tEndBinning));
		std::printf("\t- Rasterizing:   %.3fms\n", ms(tStartSplat, tEndSplat));
		std::printf("\t  (%llu gaussians, %llu tile entries)\n\n",
			(unsigned long long)gaussians.count, (unsigned long long)numRendered);
	}

	return numRendered;
}
