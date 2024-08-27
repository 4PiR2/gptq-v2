#include <torch/torch.h>


int accumulate_hessian_kernel(
    void*,
    const void*,
    const int,
    const int
);


void accumulate_hessian(
    torch::Tensor& mat_hessian,
    const torch::Tensor& mat_input
) {
	int size_batch = mat_input.size(0), size_hidden = mat_input.size(1);
	// int dev = mat_input.get_device();
	int err = accumulate_hessian_kernel(
		mat_hessian.data_ptr(),
		mat_input.data_ptr(),
		size_hidden,
		size_batch
	);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("accumulate_hessian", &accumulate_hessian, "H += X.t() @ X (FP16 x FP16 = FP32 matmul)");
}
