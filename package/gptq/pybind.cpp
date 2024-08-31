#include <torch/torch.h>


void accumulate_hessian_fp16_fp32(
    torch::Tensor& mat_hessian,
    const torch::Tensor& mat_input
);


void accumulate_hessian_bf16_fp32(
    torch::Tensor& mat_hessian,
    const torch::Tensor& mat_input
);


void quantize_range
(
    torch::Tensor quant,
    torch::Tensor scale,
    torch::Tensor out_q,
    torch::Tensor qzero,
    float maxq,
    torch::Tensor hessian_inv,
    torch::Tensor weights,
    torch::Tensor error,
    int a,
    int b
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("accumulate_hessian_fp16_fp32", &accumulate_hessian_fp16_fp32, "H += X.t() @ X (FP16 x FP16 = FP32 matmul)");
    m.def("accumulate_hessian_bf16_fp32", &accumulate_hessian_bf16_fp32, "H += X.t() @ X (BF16 x BF16 = FP32 matmul)");
    m.def("quantize_range", &quantize_range, "");
}
