import time
import torch
from torch import nn
import torchvision
import torchlens as tl


device = torch.device('meta')
model = torchvision.models.resnet18(pretrained=False)
model.to(device=device)
x = torch.rand(1, 3, 224, 224, device=device)
# y = model(x)

model_history = tl.log_forward_pass(
    model=model,
    input_args=x,
    layers_to_save=None,
    # vis_opt='unrolled',
)

layer_dict_main_keys = model_history.layer_dict_main_keys

for layer_key, layer_value in layer_dict_main_keys.items():
    parents = [layer_dict_main_keys[l].containing_module_origin for l in layer_value.parent_layers]
    children = [layer_dict_main_keys[l].containing_module_origin for l in layer_value.child_layers]
    print(parents, layer_key, layer_value.containing_module_origin, children)

print(model_history)
