import time

import torch

import gptq


device = torch.device('cuda:0')

batch_size = 2048 * 2
n_hidden_dims = 8192 * 2

mat_input = torch.randn(batch_size, n_hidden_dims, dtype=torch.float16, device=device)
mat_hessian = torch.randn(n_hidden_dims, n_hidden_dims, dtype=torch.float32, device=device)

n_warmups = 10
n_iters = 100

print('PyTorch 16x16=>32 32+32=>32 baseline')

for _ in range(n_warmups):
    x = mat_input.to(dtype=torch.float32)
    mat_hessian += x.t() @ x

torch.cuda.synchronize(device=device)
t_start = time.time()

for _ in range(n_iters):
    x = mat_input.to(dtype=torch.float32)
    mat_hessian += x.t() @ x

torch.cuda.synchronize(device=device)
t_end = time.time()
print(t_end - t_start)

print('PyTorch 16x16=>16 16+32=>32')

for _ in range(n_warmups):
    mat_hessian += mat_input.t() @ mat_input

torch.cuda.synchronize(device=device)
t_start = time.time()

for _ in range(n_iters):
    mat_hessian += mat_input.t() @ mat_input

torch.cuda.synchronize(device=device)
t_end = time.time()
print(t_end - t_start)

print('Kernel')

for _ in range(n_warmups):
    gptq.accumulate_hessian(mat_hessian, mat_input)

torch.cuda.synchronize(device=device)
t_start = time.time()

for _ in range(n_iters):
    gptq.accumulate_hessian(mat_hessian, mat_input)

torch.cuda.synchronize(device=device)
t_end = time.time()
print(t_end - t_start)
