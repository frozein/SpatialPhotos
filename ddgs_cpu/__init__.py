"""Multithreaded CPU renderer for 3D gaussian splats, forward pass only.

Build with `python -m ddgs_cpu`. The C++ extension is cached by PyTorch 
and also builds automatically on first render if needed. Building 
needs a compatible C++ compiler (C++20 for current PyTorch), `ninja`,
and an activated Python environment.
"""

import math
import os
import warnings
from typing import NamedTuple

import torch
import torch.nn as nn

# ------------------------------------------- #

_MODULE_NAME = "ddgs_cpu_C"

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC_DIR = os.path.join(_ROOT_DIR, "csrc")

_extension = None

def _load_extension(*, verbose=None):
	global _extension

	if _extension is not None:
		return _extension

	from torch.utils.cpp_extension import load

	sources = [
		os.path.join(_CSRC_DIR, "ext.cpp"),
		os.path.join(_CSRC_DIR, "src", "ddgs_cpu_forward.cpp"),
	]

	# only announce the compile when there is nothing cached to reuse:
	# ---------------
	needsBuild = True
	try:
		from torch.utils.cpp_extension import _get_build_directory

		buildDir = _get_build_directory(_MODULE_NAME, verbose=False)
		suffix = ".pyd" if os.name == "nt" else ".so"
		needsBuild = not os.path.isfile(os.path.join(buildDir, f"{_MODULE_NAME}{suffix}"))
	except Exception:
		pass

	if needsBuild:
		print("ddgs_cpu: compiling the CPU renderer (no CUDA required)...", flush=True)

	_extension = load(
		name=_MODULE_NAME,
		sources=sources,
		extra_include_paths=[os.path.join(_CSRC_DIR, "include")],
		extra_cflags=(["/O2"] if os.name == "nt" else
					  ["-O3", "-fno-math-errno", "-funroll-loops"]),
		with_cuda=False,
		verbose=os.environ.get("DDGS_CPU_VERBOSE", "0") == "1" if verbose is None else verbose,
	)

	return _extension

def _default_thread_count():
	threads = os.environ.get("DDGS_CPU_THREADS")

	try:
		return max(0, int(threads))
	except (TypeError, ValueError):
		return 0

def _to_render_input(tensor):
	return tensor.detach().to("cpu", torch.float32).contiguous()

# ------------------------------------------- #

def _look(eye, forward, up):
	f = forward / torch.norm(forward)
	u = up / torch.norm(up)

	s = torch.cross(f, u, dim=0)
	s = s / torch.norm(s)

	u = torch.cross(s, f, dim=0)

	m = torch.eye(4, dtype=torch.float32, device=eye.device)
	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)

	return m

def _perspective(fovY, aspect, zNear, zFar):
	tanHalfFov = math.tan(fovY / 2)

	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tanHalfFov)
	m[1, 1] = 1 / tanHalfFov
	m[2, 2] = -(zFar + zNear) / (zFar - zNear)
	m[2, 3] = -(2 * zFar * zNear) / (zFar - zNear)
	m[3, 2] = -1.0

	return m

# ------------------------------------------- #

class RenderOutputs:
	COLOR = 1 << 0
	ALPHA = 1 << 1
	DEPTH = 1 << 2

class Settings:
	def __init__(self, width: int, height: int,
				 view: torch.Tensor, proj: torch.Tensor, focalX: float, focalY: float,
				 outputs: RenderOutputs = RenderOutputs.COLOR, debug: bool = False,
				 threads: int = None):

		self.outputs = outputs

		self.width = int(width)
		self.height = int(height)
		self.view = view
		self.proj = proj
		self.focalX = float(focalX)
		self.focalY = float(focalY)
		self.debug = bool(debug)
		self.threads = _default_thread_count() if threads is None else int(threads)

	@staticmethod
	def from_colmap(width: int, height: int, focalX: float, focalY: float, R: torch.Tensor, T: torch.Tensor,
					debug: bool = False):

		# flip sign convention:
		# ---------------
		R = -R

		# get view matrix
		# ---------------
		camPos = -R.T @ T

		forward = R[2, :3]
		up = R[1, :3]

		view = _look(camPos, forward, up)

		# get proj matrix:
		# ---------------
		aspect = width / height
		fovY = 2 * math.atan(height / (2 * focalY))
		zNear = 0.1
		zFar = 1000.0

		proj = _perspective(fovY, aspect, zNear, zFar).to(R.device)

		focalX = width / (2 * math.tan(fovY / 2))
		focalY = focalX

		# return settings:
		# ---------------
		return Settings(
			width, height,
			view, proj, focalX, focalY,
			debug=debug
		)

# ------------------------------------------- #

class RenderResult(NamedTuple):
	color: torch.Tensor | None
	alpha: torch.Tensor | None
	depth: torch.Tensor | None

def render(settings: Settings,
		   means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor,
		   normalizeRotations=True) -> RenderResult:

	if normalizeRotations:
		rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True).clamp(min=1e-8)

	if any(t.requires_grad for t in (means, scales, rotations, opacities, harmonics)):
		warnings.warn("ddgs_cpu renders without gradients, inputs will be detached", stacklevel=2)

	device = means.device
	extension = _load_extension()

	img, alpha, depth = extension.forward(
		width=settings.width, height=settings.height,
		view=_to_render_input(settings.view), proj=_to_render_input(settings.proj),
		focalX=settings.focalX, focalY=settings.focalY,
		means=_to_render_input(means),
		scales=_to_render_input(scales),
		rotations=_to_render_input(rotations),
		opacities=_to_render_input(opacities),
		harmonics=_to_render_input(harmonics),
		debug=settings.debug, numThreads=settings.threads,
	)

	if device.type != "cpu":
		img = img.to(device)
		alpha = alpha.to(device)
		depth = depth.to(device)

	return RenderResult(
		color=img   if settings.outputs & RenderOutputs.COLOR else None,
		alpha=alpha if settings.outputs & RenderOutputs.ALPHA else None,
		depth=depth if settings.outputs & RenderOutputs.DEPTH else None,
	)

class Renderer(nn.Module):
	def __init__(self, settings, normalizeRotations=True):
		super().__init__()
		self.settings = settings
		self.normalizeRotations = normalizeRotations

	def forward(self, means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, opacities: torch.Tensor, harmonics: torch.Tensor):
		return render(
			self.settings,
			means, scales, rotations, opacities, harmonics,
			self.normalizeRotations
		)
