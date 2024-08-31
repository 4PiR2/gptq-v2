// https://github.com/4PiR2/exllamav2/blob/gptq/exllamav2/exllamav2_ext/cpp/quantize_func.cpp


#include <torch/torch.h>


void fused_quantize_adjust_cuda
(
    const float*    weights,
    float*          quant,
    const float*    scale,
    uint16_t*       out_q,
    const float*    hessian_inv,
    float*          error,
    int row,
    int rows,
    int columns,
    const float*    qzero,
    float maxq
);


void vv_mul_sub_cuda
(
    const float* x,
    const float* y,
    float* z,
    int x_size,
    int y_size
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
)
{
    int columns = weights.size(1);
    int hcolumns = hessian_inv.size(1);
    TORCH_CHECK(hcolumns == weights.size(0), "H shape mismatch")

    for (int c = a; c < b; c++)
    {
        fused_quantize_adjust_cuda
        (
            (const float*) weights.data_ptr(),
            (float*) quant.data_ptr(),
            (const float*) scale.data_ptr(),
            out_q.device().is_meta() ? NULL : (uint16_t*) out_q.data_ptr(),
            (const float*) hessian_inv.data_ptr(),
            (float*) error.data_ptr(),
            c,          // row
            hcolumns,   // rows
            columns,
            (const float*) qzero.data_ptr(),
            maxq
        );

        vv_mul_sub_cuda
        (
            ((const float*) hessian_inv.data_ptr()) + (uint64_t)c * (uint64_t)hcolumns + (uint64_t)c,
            ((const float*) error.data_ptr()) + (uint64_t)c * (uint64_t)columns,
            ((float*) weights.data_ptr()) + (uint64_t)c * (uint64_t)columns,
            b - c,
            columns
        );
    }

    torch::Tensor x = hessian_inv.slice(0, a, b).slice(1, b).transpose(0, 1);
    torch::Tensor y = error.slice(0, a, b);
    weights.slice(0, b).addmm_(x, y, 1.0f, -1.0f);
}
