import torch
# Create a fake mode


x = torch.randn(3, 4)

from torch._subclasses.fake_tensor import FakeTensorMode
fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
# Fakeify some real tensors
fake_x = fake_mode.from_tensor(x)
with fake_mode:
    # Do some operations on the fake tensors
    y = x * 2 + x
    fake_y = fake_x * 2 + x
    # Factory operations automatically get fakeified in the context manager
    fake_z = torch.empty(20)

fake_y = fake_x * 2 + x
a = 0
