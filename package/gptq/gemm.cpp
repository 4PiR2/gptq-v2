#include <torch/torch.h>


int gemm_kernel(
    const void*,
    const void*,
    void*,
    const int,
    const int,
    const int,
    const int
);


void mul(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
	int size_b = A.size(0);
	int size_m = A.size(-2);
	int size_n = C.size(-1);
	int size_k = A.size(-1);
	int dev = A.get_device();
	int err = gemm_kernel(
		A.data_ptr(),
		B.data_ptr(),
		C.data_ptr(),
		size_m,
		size_n,
		size_k,
		size_b
	);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("mul", &mul, "FP16 x FP16 = FP32 matmul");
}
