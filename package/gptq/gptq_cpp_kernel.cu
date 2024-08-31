#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
#define MAX_P_GRID 64


__forceinline__ __device__ float clamp(float x, float a, float b)
{
    return fmaxf(a, fminf(b, x));
}


__global__ void fused_quantize_adjust_kernel
(
    const float* __restrict__ weights,          // input  weights           [rows, columns]
    float*       __restrict__ quant,            // output quantized weights [rows, columns]
    const float* __restrict__ scale,            // input  scales            [1, columns]
    uint16_t*    __restrict__ out_q,            // output qweights          [rows, columns]
    const float* __restrict__ hessian_inv,      // input hessian            [rows, rows]
    float*       __restrict__ error,            // output error             [rows, columns]
    int row,                                    // row index to quantize
    int rows,                                   // num rows
    int columns,                                // num columns
    const float* __restrict__ qzero,            // 2^(bits - 1)
    float maxq                                  // (2^bits) - 1
)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    uint64_t idx = (uint64_t)row * (uint64_t)columns + (uint64_t)column;

    // Quantize

    float x = weights[idx];
    float s = scale[column];
    float z = qzero[column];
    x /= s;
    x = rintf(x);
    x += z;
    x = clamp(x, 0.0f, maxq);

    // Optionally save quant

    if (out_q) out_q[idx] = static_cast<uint16_t>(x);

    // Downcast while quantizing

    half h_s = __float2half_rn(s);
    half h_x = __float2half_rn(x);
    half h_z = __float2half_rn(z);

    // Dequantize

    h_x = __hsub(h_x, h_z);
    h_x = __hmul(h_x, h_s);
    float q = __half2float(h_x);
    quant[idx] = q;

    // Adjust error

    uint64_t d_idx = (uint64_t)row * (uint64_t)rows + (uint64_t)row;
    float d = hessian_inv[d_idx];  // H diagonal
    float w = weights[idx];
    error[idx] = (w - q) / d;
}


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
)
{
    dim3 threads(BLOCKSIZE_X, 1);
    dim3 blocks(DIVIDE(columns, BLOCKSIZE_X), 1);

    fused_quantize_adjust_kernel<<<blocks, threads>>>
    (
        weights,
        quant,
        scale,
        out_q,
        hessian_inv,
        error,
        row,
        rows,
        columns,
        qzero,
        maxq
    );
}


// Compute z = z - x.T @ y

__global__ void vv_mul_sub_kernel
(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int x_size,
    int y_size
)
{
    int y_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int x_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_idx >= y_size) return;
    if (x_idx >= x_size) return;

    uint64_t z_idx = (uint64_t)y_size * (uint64_t)x_idx + (uint64_t)y_idx;

    float vx = x[x_idx];
    float4 vy = *((float4*) (y + y_idx));
    float4 vz = *((float4*) (z + z_idx));
    vz.x -= vy.x * vx;
    vz.y -= vy.y * vx;
    vz.z -= vy.z * vx;
    vz.w -= vy.w * vx;
    *((float4*) (z + z_idx)) = vz;
}


void vv_mul_sub_cuda
(
    const float* x,
    const float* y,
    float* z,
    int x_size,
    int y_size
)
{
    dim3 blockDim, gridDim;
    blockDim.x = BLOCKSIZE_X;
    blockDim.y = BLOCKSIZE_Y;
    blockDim.z = 1;
    gridDim.x = DIVIDE(y_size / 4, BLOCKSIZE_X);
    gridDim.y = DIVIDE(x_size, BLOCKSIZE_Y);
    gridDim.z = 1;

    vv_mul_sub_kernel<<<gridDim, blockDim>>>(x, y, z, x_size, y_size);
}
