#include <torch/torch.h>


int accumulate_hessian_fp16_fp32_kernel(
    void*,
    const void*,
    const int,
    const int
);


int accumulate_hessian_bf16_fp32_kernel(
    void*,
    const void*,
    const int,
    const int
);


void accumulate_hessian_fp16_fp32(
    torch::Tensor& mat_hessian,
    const torch::Tensor& mat_input
) {
    int size_batch = mat_input.size(0), size_hidden = mat_input.size(1);
    int err = accumulate_hessian_fp16_fp32_kernel(
        mat_hessian.data_ptr(),
        mat_input.data_ptr(),
        size_hidden,
        size_batch
    );
}


void accumulate_hessian_bf16_fp32(
    torch::Tensor& mat_hessian,
    const torch::Tensor& mat_input
) {
    int size_batch = mat_input.size(0), size_hidden = mat_input.size(1);
    int err = accumulate_hessian_bf16_fp32_kernel(
        mat_hessian.data_ptr(),
        mat_input.data_ptr(),
        size_hidden,
        size_batch
    );
}
