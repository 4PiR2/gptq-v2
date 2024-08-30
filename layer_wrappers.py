from collections import deque


import torch
from torch import nn
from torch._subclasses import FakeTensorMode

from gptq_2 import HessianHook


def move_to_device(nested_structure, device: torch.device = torch.device('cpu')):
    if isinstance(nested_structure, torch.Tensor):
        # If it's a tensor, move it to the CPU
        return nested_structure.to(device=device)
    elif isinstance(nested_structure, dict):
        # If it's a dictionary, recursively move each value to the CPU
        return {k: move_to_device(v, device) for k, v in nested_structure.items()}
    elif isinstance(nested_structure, list):
        # If it's a list, recursively move each element to the CPU
        return [move_to_device(item, device) for item in nested_structure]
    elif isinstance(nested_structure, tuple):
        # If it's a tuple, recursively move each element to the CPU
        return tuple(move_to_device(item, device) for item in nested_structure)
    else:
        # If it's neither a tensor, dictionary, list, nor tuple, return it as is
        return nested_structure


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


def find_and_replace_submodule_by_name(
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


class Catcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.inps: list[torch.Tensor] = []
        self.inp_kwargs: dict = {}

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # catch inputs
        self.inps.append(hidden_states)
        self.inp_kwargs = kwargs
        raise ValueError


class RecorderWrapper(nn.Module):
    fake_mode: FakeTensorMode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
    stage_fake_mode: int = 0
    stage_hessian_accumulation: int = 1
    stage_recording_mode: int = 2
    stage_replay_mode: int = 3

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module: nn.Module = module
        self.hessian_hook: HessianHook | None = None
        self.stage: int = RecorderWrapper.stage_fake_mode
        self.outs: list[torch.Tensor] = []
        self.out_pointer: int = 0

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        device: torch.device = hidden_states.device
        match self.stage:
            case RecorderWrapper.stage_fake_mode:
                with RecorderWrapper.fake_mode:
                    out = torch.empty(
                        *hidden_states.shape[:-1], self.module.weight.size(0),
                        dtype=hidden_states.dtype,
                        device=device,
                    )
                return out
            case RecorderWrapper.stage_hessian_accumulation:
                self.hessian_hook.add_batch(hidden_states)
                raise ValueError
            case RecorderWrapper.stage_recording_mode:
                out: torch.Tensor = self.module(hidden_states, **kwargs)
                self.outs.append(out.cpu())
                return RecorderWrapper.fake_mode.from_tensor(out)
            case RecorderWrapper.stage_replay_mode:
                out = self.outs[self.out_pointer]
                self.out_pointer = (self.out_pointer + 1) % len(self.outs)
                return out.to(device=device)
            case _:
                raise NotImplementedError


_label: str = 'label'
_input: str = 'input'
_output: str = 'output'


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


@torch.enable_grad()
def extract_dependencies(
        model: nn.Module,
        module_types: list[type[nn.Module]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        input_device: torch.device = torch.device('meta'),
        inp_kwargs: dict = {},
) -> list[tuple]:
    # replace layers with wrapper
    layers = find_submodules_by_types(model, module_types=module_types)
    for key, layer in layers.items():
        find_and_replace_submodule_by_name(model, key, BackwardWrapper(module=layer, label=key))

    fake_mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True)
    # fake_tensor = fake_mode.from_tensor(input_tensor)
    with fake_mode:
        fake_tensor = torch.empty(input_shape, dtype=input_dtype, device=input_device)
    output = BackwardWrapper(nn.Identity(), label=_output)(model(BackwardWrapper(nn.Identity(), label=_input)(nn.Parameter(fake_tensor)), **inp_kwargs)[0])
    root_grad_fn = output.grad_fn

    # extract the whole backward graph from the gradient compute graph using BFS
    backward_graph: dict[object, set[object]] = {}
    queue = deque()
    queue.append(root_grad_fn)
    while queue:
        grad_fn = queue.popleft()
        next_grad_fns = set(fn for fn, _ in grad_fn.next_functions if fn is not None)
        backward_graph[grad_fn] = next_grad_fns
        queue.extend(next_grad_fns.difference(backward_graph.keys()))

    all_labels_topo = [getattr(grad_fn, _label) for grad_fn in _topological_sort(backward_graph) if hasattr(grad_fn, _label)]

    # find the equivalence sets for hessian computation
    backward_graph_reversed: dict[tuple, list[object]] = {}
    for grad_fn, next_grad_fns in backward_graph.items():
        next_hash: tuple = tuple(sorted([id(next_grad_fn) for next_grad_fn in next_grad_fns]))
        if next_hash in backward_graph_reversed:
            backward_graph_reversed[next_hash].append(grad_fn)
        else:
            backward_graph_reversed[next_hash] = [grad_fn]
    equivalence_lists: list[list[str]] = [
        [getattr(fn, _label) for fn in fn_list if hasattr(fn, _label)]
        for fn_list in backward_graph_reversed.values()
    ]
    equivalence_lists = [e_list for e_list in equivalence_lists if e_list]
    equivalence_lists.sort(key=lambda e_list: all_labels_topo.index(e_list[0]), reverse=True)  # topological ordering

    # find the module dependencies for hessian computation, non-interesting ops are filtered out
    forward_parents: dict[str, set[str]] = {}
    for grad_fn in backward_graph.keys():
        if not hasattr(grad_fn, _label):
            continue
        label = getattr(grad_fn, _label)
        forward_parents[label] = set()
        queue = deque()
        queue.extend(backward_graph[grad_fn])
        visited = set()
        while queue:
            fn = queue.popleft()
            visited.add(fn)
            if hasattr(fn, _label):
                forward_parents[label].add(getattr(fn, _label))
            else:
                queue.extend(backward_graph[fn].difference(visited))

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
    assert not non_reachable

    # add to_release
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
        result.append((equivalence_list, to_release))
    for children in forward_children.values():
        assert not children

    # replace layers back
    for key, layer in layers.items():
        find_and_replace_submodule_by_name(model, key, layer)

    assert result.pop(0) == ([_input], [])
    return result


def _topological_sort(backward_graph: dict[object, set[object]]) -> list:
    # Step 1: Calculate in-degrees
    in_degree = {node: 0 for node in backward_graph}
    for deps in backward_graph.values():
        for dep in deps:
            in_degree[dep] = in_degree.get(dep, 0) + 1

    # Step 2: Initialize the queue with nodes having in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []

    # Step 3: Kahn's algorithm for topological sorting
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in backward_graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return topo_order
