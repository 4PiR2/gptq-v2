from collections import deque


import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
import torchvision
import transformers


_label: str = 'label'
_input: str = 'input'


class BackwardWrapper(nn.Module):
    def __init__(self, module: nn.Module, label: str = None):
        super().__init__()

        class autograd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args: torch.Tensor, **kwargs):
                setattr(ctx, _label, label)
                params = [nn.Parameter(i) for i in args]
                setattr(ctx, 'inputs', params)
                with torch.enable_grad():
                    outputs = module(*params, **kwargs)
                setattr(ctx, 'outputs', outputs if isinstance(outputs, tuple) else (outputs,))
                return tuple(o.detach() for o in outputs) if isinstance(outputs, tuple) else outputs.detach()

            @staticmethod
            def backward(ctx, *grad_outputs):
                torch.autograd.backward(tensors=getattr(ctx, 'outputs'), grad_tensors=grad_outputs)
                return tuple(i.grad for i in getattr(ctx, 'inputs'))

        self.autograd: type[torch.autograd.Function] = autograd
        self.module: nn.Module = module
        self.label: str = label

    def forward(self, *args, **kwargs):
        return self.autograd.apply(*args, **kwargs)


def extract_dependencies(
        model: nn.Module,
        module_types: list[type[nn.Module]],
        input_shape: torch.Size,
        device: torch.device = torch.device('meta'),
) -> list[tuple]:
    # replace layers with wrapper
    layers = find_submodules_by_types(model, module_types=module_types)
    for key, layer in layers.items():
        replace_submodule_by_name(model, key, BackwardWrapper(module=layer, label=key))

    fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
    # fake_tensor = fake_mode.from_tensor(input_tensor)
    with fake_mode:
        fake_tensor = torch.empty(input_shape, device=device)
    output, *_ = model(BackwardWrapper(nn.Identity(), label=_input)(nn.Parameter(fake_tensor)))
    root_grad_fn = output.grad_fn

    # extract the whole backward tree from the gradient compute graph using BFS
    backward_tree: dict[object, set[object]] = {}
    queue = deque()
    queue.append(root_grad_fn)
    while queue:
        grad_fn = queue.popleft()
        next_grad_fns = set(fn for fn, _ in grad_fn.next_functions if fn is not None)
        backward_tree[grad_fn] = next_grad_fns
        queue.extend(next_grad_fns.difference(backward_tree.keys()))
    all_labels_bfs = [getattr(grad_fn, _label) for grad_fn in backward_tree.keys() if hasattr(grad_fn, _label)]

    # find the equivalence sets for hessian computation
    backward_tree_reversed: dict[tuple, list[object]] = {}
    for grad_fn, next_grad_fns in backward_tree.items():
        next_hash: tuple = tuple(sorted([id(next_grad_fn) for next_grad_fn in next_grad_fns]))
        if next_hash in backward_tree_reversed:
            backward_tree_reversed[next_hash].append(grad_fn)
        else:
            backward_tree_reversed[next_hash] = [grad_fn]
    equivalence_lists: list[list[str]] = [
        [getattr(fn, _label) for fn in fn_list if hasattr(fn, _label)]
        for fn_list in backward_tree_reversed.values()
    ]
    equivalence_lists = [e_list for e_list in equivalence_lists if e_list]
    equivalence_lists.sort(key=lambda e_list: all_labels_bfs.index(e_list[0]), reverse=True)  # topological ordering

    # find the module dependencies for hessian computation, non-interesting ops are filtered out
    forward_parents: dict[str, set[str]] = {}
    for grad_fn in backward_tree.keys():
        if not hasattr(grad_fn, _label):
            continue
        label = getattr(grad_fn, _label)
        forward_parents[label] = set()
        queue = deque()
        queue.extend(backward_tree[grad_fn])
        visited = set()
        while queue:
            fn = queue.popleft()
            visited.add(fn)
            if hasattr(fn, _label):
                forward_parents[label].add(getattr(fn, _label))
            else:
                queue.extend(backward_tree[fn].difference(visited))

    forward_children: dict[str, set[str]] = {label: set() for label in forward_parents.keys()}
    for label, parent_labels in forward_parents.items():
        for parent_label in parent_labels:
            forward_children[parent_label].add(label)

    # make sure all layers are reachable from the input
    queue = deque()
    queue.append(_input)
    visited = set()
    while queue:
        label = queue.popleft()
        visited.add(label)
        queue.extend(forward_children[label].difference(visited))
    non_reachable: set[str] = set(forward_children.keys()).difference(visited)
    # assert not non_reachable

    result: list[tuple] = []
    for equivalence_list in equivalence_lists:
        if not non_reachable.isdisjoint(equivalence_list):
            continue
        parent_labels = forward_parents[equivalence_list[0]]
        for parent_label in parent_labels:
            for label in equivalence_list:
                forward_children[parent_label].remove(label)
        to_release = [parent_label for parent_label in parent_labels if not forward_children[parent_label]]
        for label in to_release:
            forward_children.pop(label)
        result.append((equivalence_list, list(parent_labels), to_release))
    # for children in forward_children.values():
    #     assert not children

    # replace layers back
    for key, layer in layers.items():
        replace_submodule_by_name(model, key, layer)

    return result


def find_submodules_by_types(
        module: nn.Module, module_types: list[type] = None, name: str = '',
) -> dict[str, nn.Module]:
    """
    find layers by types
    """
    if module_types is None:
        module_types = [nn.Conv2d, nn.Linear]
    if type(module) in module_types:
        return {name: module}
    res = {}
    for child_name, child_module in module.named_children():
        res.update(find_submodules_by_types(
            module=child_module, module_types=module_types, name=name + '.' + child_name if name else child_name,
        ))
    return res


def replace_submodule_by_name(
        module: nn.Module, search_name: str, module_to_replace: nn.Module = None,
) -> nn.Module | None:
    """
    replace module by key name (if new module is not None) and return the old module
    """
    names = search_name.split('.')
    parent_module = module
    for name in names:
        next_module = None
        for child_name, child_module in module.named_children():
            if child_name == name:
                next_module = child_module
                break
        if next_module is None:
            return None
        parent_module, module = module, next_module
    if module_to_replace is not None:
        parent_module.__setattr__(names[-1], module_to_replace)
    return module


def test():
    torch.manual_seed(0)
    device = torch.device('cpu')
    # model = torchvision.models.resnet34(pretrained=False).to(device=device)
    model = torchvision.models.inception_v3(weights=None, init_weights=False).to(device=device)

    result = extract_dependencies(model, [nn.Linear, nn.Conv2d], torch.Size((1, 3, 512, 512)), device)

    a = 0
    pass


if __name__ == '__main__':
    test()
