# ddgs_cpu

A multithreaded, CPU only rasterizer for 3D gaussian splats, forward pass only.
It exists so this project can run on machines without an NVIDIA GPU, where the
CUDA renderer ([DDGS](https://github.com/splatsdotcom/DDGS)) cannot be built.

It is a drop-in replacement for the rendering half of `ddgs`:

```python
import ddgs_cpu as ddgs   # same Settings / RenderOutputs / render / Renderer
```

`spatial_photos.py` picks the backend automatically: `ddgs` when
`torch.cuda.is_available()`, `ddgs_cpu` otherwise.

## Building

Nothing to install. The extension compiles on first import (a few seconds) and
is cached in `~/.cache/torch_extensions`, which needs a C++ toolchain (on macOS,
the Xcode command line tools) and `ninja`. Set `DDGS_CPU_VERBOSE=1` to see the
compiler output, and `DDGS_CPU_THREADS=N` to cap the worker count (default: one
per hardware thread).

## What matches the CUDA renderer

The per-gaussian math is a transcription of `DDGS/csrc/src/ddgs_forward.cu`, so
the pixels match to floating point rounding:

- the same frustum cull (`1.2 * w` clip bounds), covariance construction
  (`Sigma = (S R)^T (S R)`) and EWA projection through the same Jacobian, with
  no 2D low pass filter added to the projected covariance
- the same 16x16 tiles, the same bounding box of tiles per gaussian from a
  `ceil(3 * sqrt(lambda_max))` pixel radius, and the same 3x3 conic evaluation
  over *every* pixel of every tile the box covers
- the same alpha handling: `min(0.99, opacity * exp(power))`, discarded below
  `1/255`, front to back compositing into a premultiplied colour buffer, and the
  same `1e-5` transmittance cutoff applied in batches of 256 gaussians, so early
  termination happens at the same granularity as the GPU's thread blocks
- depth is the same unusual quantity: the view space distance of the gaussian at
  which accumulated transmittance first drops below `0.95`, or `0` if no gaussian
  ever pushes it that far. It is *not* an expected or median depth
- gaussians are composited in the same order. The GPU sorts `(tile, raw bits of
  view space z)` with a stable radix sort; here each tile's list is sorted on a
  key that packs those depth bits above the gaussian index, which reproduces the
  GPU's tie-breaking exactly. Results do not depend on the thread count
- spherical harmonics are read as plain RGB at a stride of 3, i.e. only the
  degree 0 band, matching the CUDA kernel's `harmonics[idx * 3]` indexing

## What is different by design

- **Forward pass only.** No gradients, and nothing that only a backward pass
  needs is computed or kept (2D means, per-pixel contributor counts, the geometry
  and binning scratch buffers). `render()` detaches its inputs and warns if any
  of them required grad.
- **CPU parallel structure.** Tiles are handed to worker threads dynamically
  instead of to thread blocks; there is no cooperative loading of gaussians into
  shared memory, no global radix sort over all `(gaussian, tile)` pairs, and no
  prefix scan - per tile counting, offsets and sorts replace them.
- Results are still bit-identical run to run, and identical whatever
  `DDGS_CPU_THREADS` is set to.

## Testing

`test_parity.py` checks the renderer against an independent reference
implementation of the CUDA forward pass, written in numpy from the `.cu` source
with the covariance projection derived analytically rather than transcribed, so a
transposition error in either implementation shows up as a mismatch. It covers
sparse, dense/opaque, faint (sub-cutoff alpha), clipped and partial-tile scenes,
plus single and empty inputs, and checks that single threaded and multithreaded
renders are bit-identical.

```bash
python ddgs_cpu/test_parity.py           # parity, ~10s
python ddgs_cpu/test_parity.py --bench   # timings at 1024x1024
```

Colour and alpha agree to ~2e-6 and depth is bit-exact. This is not a comparison
against CUDA output - that needs an NVIDIA GPU - but it does pin down the math.

## Layout

```
csrc/include/ddgs_cpu_math.h   column-major vec/mat helpers, QuickMath conventions
csrc/include/ddgs_cpu.h        constants, gaussian/settings structs, entry point
csrc/src/ddgs_cpu_forward.cpp  preprocess, binning and rasterization
csrc/ext.cpp                   torch bindings and input validation
__init__.py                    the ddgs-compatible python API
```
