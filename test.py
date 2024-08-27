import gptq
import torch


device = 'cuda:1'

type_a = torch.float16
type_b = torch.float16
type_c = torch.float32

a = torch.randn(77, 310, 370, dtype=type_a, device=device)
b = torch.randn(77, 370, 410, dtype=type_b, device=device)
c = torch.randn(77, 310, 410, dtype=type_c, device=device)

# ref = a @ b + c
# print(ref.dtype)
# print(ref)

ref2 = a.to(type_c) @ b.to(type_c) + c
# print(ref2.dtype)
# print(ref2)

gptq.mul(a, b, c)
# print(c)

diff = c - ref2
print(diff)
print(diff.abs().max().item())
