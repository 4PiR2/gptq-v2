import gptq
import torch


device = 'cuda:1'

type_x = torch.float16
type_h = torch.float32

x = torch.randn(256, 512, dtype=type_x, device=device)
h = torch.randn(512, 512, dtype=type_h, device=device)
x_32 = x.to(type_h)

ref = h + x.t() @ x
# print(ref.dtype)
print(ref)

ref2 = h + x_32.t() @ x_32
# print(ref2.dtype)
print(ref2)

gptq.accumulate_hessian(h, x)
print(h)

diff = h - ref2
print(diff)
print(diff.abs().max().item())
