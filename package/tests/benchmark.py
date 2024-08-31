import time

import torch

import gptq_py


device = torch.device('cuda:1')

batch_size = 16
n_hidden_dims = 4096

mat_input = torch.randn(batch_size, n_hidden_dims, dtype=torch.float16, device=device)
mat_hessian = torch.randn(n_hidden_dims, n_hidden_dims, dtype=torch.float32, device=device)

n_warmups = 100
n_iters = 1000

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

for _ in range(n_warmups):
    gptq.accumulate_hessian(mat_hessian, mat_input)

torch.cuda.synchronize(device=device)
t_start = time.time()

for _ in range(n_iters):
    gptq.accumulate_hessian(mat_hessian, mat_input)

torch.cuda.synchronize(device=device)
t_end = time.time()
print(t_end - t_start)
