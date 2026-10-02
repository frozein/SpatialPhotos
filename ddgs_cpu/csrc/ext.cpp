/* ext.cpp
 *
 * torch bindings for the CPU splat renderer
 */

#include <torch/extension.h>
#include <tuple>

#include "ddgs_cpu.h"

//-------------------------------------------//

static uint32_t _ddgs_cpu_validate_gaussians(const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations,
                                             const at::Tensor& opacities, const at::Tensor& harmonics)
{
	if(means.dtype() != torch::kFloat32 || scales.dtype() != torch::kFloat32 || rotations.dtype() != torch::kFloat32 ||
	   opacities.dtype() != torch::kFloat32 || harmonics.dtype() != torch::kFloat32)
		throw std::invalid_argument("Inputs must be float32");
	if(means.device().type() != torch::kCPU || scales.device().type() != torch::kCPU || rotations.device().type() != torch::kCPU ||
	   opacities.device().type() != torch::kCPU || harmonics.device().type() != torch::kCPU)
		throw std::invalid_argument("Inputs must be in CPU memory");

	int64_t numGaussians = means.size(0);

	if(means    .dim() != 2 || means    .size(0) != numGaussians || means    .size(1) != 3)
		throw std::invalid_argument("Means must have shape (numGaussians, 3)");
	if(scales   .dim() != 2 || scales   .size(0) != numGaussians || scales   .size(1) != 3)
		throw std::invalid_argument("Scales must have shape (numGaussians, 3)");
	if(rotations.dim() != 2 || rotations.size(0) != numGaussians || rotations.size(1) != 4)
		throw std::invalid_argument("Rotations must have shape (numGaussians, 4)");
	if(opacities.dim() != 2 || opacities.size(0) != numGaussians || opacities.size(1) != 1)
		throw std::invalid_argument("Opacities must have shape (numGaussians, 1)");
	if(harmonics.dim() != 3 || harmonics.size(0) != numGaussians || harmonics.size(2) != 3)
		throw std::invalid_argument("Harmonics must have shape (numGaussians, (degree + 1)^2, 3)");

	return (uint32_t)numGaussians;
}

//-------------------------------------------//

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ddgs_cpu_render(int64_t width, int64_t height, const at::Tensor& view, const at::Tensor& proj, double focalX, double focalY,
                const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations,
                const at::Tensor& opacities, const at::Tensor& harmonics,
                bool debug, int64_t numThreads)
{
	//validate:
	//---------------
	if(width <= 0 || height <= 0)
		throw std::invalid_argument("Image dimensions must be positive");
	if(width * height > UINT32_MAX)
		throw std::invalid_argument("Image dimensions are too large! Must be < UINT32_MAX pixels");

	if(view.dtype() != torch::kFloat32 || proj.dtype() != torch::kFloat32)
		throw std::invalid_argument("View and projection matrices must be float32");
	if(view.dim() != 2 || view.size(0) != 4 || view.size(1) != 4 ||
	   proj.dim() != 2 || proj.size(0) != 4 || proj.size(1) != 4)
		throw std::invalid_argument("View and projection matrices must have shape (4, 4)");

	if(focalX <= 0.0 || focalY <= 0.0)
		throw std::invalid_argument("Focal lengths must be positive");
	if(numThreads < 0)
		throw std::invalid_argument("Thread count must not be negative");

	uint32_t numGaussians = _ddgs_cpu_validate_gaussians(means, scales, rotations, opacities, harmonics);

	//populate settings:
	//---------------
	at::Tensor viewCpu = view.to(torch::kCPU).contiguous();
	at::Tensor projCpu = proj.to(torch::kCPU).contiguous();

	DDGSCPUsettings settings;
	settings.width = (uint32_t)width;
	settings.height = (uint32_t)height;
	settings.view = dc_mat4_load_row_major(viewCpu.data_ptr<float>());
	settings.proj = dc_mat4_load_row_major(projCpu.data_ptr<float>());
	settings.focalX = (float)focalX;
	settings.focalY = (float)focalY;
	settings.debug = debug;
	settings.numThreads = (uint32_t)numThreads;

	//allocate output tensors:
	//---------------
	torch::TensorOptions floatOpts = torch::TensorOptions(torch::kFloat32).device(torch::kCPU);

	at::Tensor outColor = torch::zeros({ (int64_t)settings.height, (int64_t)settings.width, 3 }, floatOpts);
	at::Tensor outAlpha = torch::zeros({ (int64_t)settings.height, (int64_t)settings.width, 1 }, floatOpts);
	at::Tensor outDepth = torch::zeros({ (int64_t)settings.height, (int64_t)settings.width, 1 }, floatOpts);

	//populate gaussians struct:
	//---------------
	at::Tensor meansC     = means    .contiguous();
	at::Tensor scalesC    = scales   .contiguous();
	at::Tensor rotationsC = rotations.contiguous();
	at::Tensor opacitiesC = opacities.contiguous();
	at::Tensor harmonicsC = harmonics.contiguous();

	DDGSCPUgaussians gaussians;
	gaussians.count = numGaussians;
	gaussians.means     = meansC    .data_ptr<float>();
	gaussians.scales    = scalesC   .data_ptr<float>();
	gaussians.rotations = rotationsC.data_ptr<float>();
	gaussians.opacities = opacitiesC.data_ptr<float>();
	gaussians.harmonics = harmonicsC.data_ptr<float>();

	//render:
	//---------------
	{
		pybind11::gil_scoped_release release;

		ddgs_cpu_forward(
			settings, gaussians,
			outColor.data_ptr<float>(),
			outAlpha.data_ptr<float>(),
			outDepth.data_ptr<float>()
		);
	}

	//return:
	//---------------
	return { outColor, outAlpha, outDepth };
}

//-------------------------------------------//

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def(
		"forward", &ddgs_cpu_render,
		"Rasterizes gaussian splats on the CPU, returns (color, alpha, depth)",
		pybind11::arg("width"), pybind11::arg("height"),
		pybind11::arg("view"), pybind11::arg("proj"),
		pybind11::arg("focalX"), pybind11::arg("focalY"),
		pybind11::arg("means"), pybind11::arg("scales"), pybind11::arg("rotations"),
		pybind11::arg("opacities"), pybind11::arg("harmonics"),
		pybind11::arg("debug") = false, pybind11::arg("numThreads") = 0
	);
}
