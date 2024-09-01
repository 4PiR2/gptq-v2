#include <torch/torch.h>


void accumulate_hessian_fp16_fp32(
    torch::Tensor&,
    const torch::Tensor&
);


void accumulate_hessian_bf16_fp32(
    torch::Tensor&,
    const torch::Tensor&
);


void quantize_range
(
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    float,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    int,
    int
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("accumulate_hessian_fp16_fp32", &accumulate_hessian_fp16_fp32, "H += X.t() @ X (FP16 x FP16 = FP32 matmul)");
    m.def("accumulate_hessian_bf16_fp32", &accumulate_hessian_bf16_fp32, "H += X.t() @ X (BF16 x BF16 = FP32 matmul)");
    m.def("gptq_quantize_range", &quantize_range, "Quantize Weights within Range Using GPTQ");
}
