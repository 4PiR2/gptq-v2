import logging
import time

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaForCausalLM

from gptq_py import HessianHook, gptq_quant
from model_utils import move_to_device, find_submodules_by_types, find_and_replace_submodule_by_name, extract_dependencies, Catcher, RecorderWrapper
from quant import reconstruct_nn_linear


def get_llama(model_path: str) -> LlamaForCausalLM:
    nn.init.kaiming_uniform_ = nn.init.uniform_ = nn.init.normal_ = lambda *args, **kwargs: None
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype='auto')
    return model


@torch.no_grad()
def get_initial_inputs(
        model: LlamaForCausalLM,
        encodings: torch.Tensor,
        device: torch.device,
        batch_size: int = 1,
        save_device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, dict]:
    """
    catch first layer's input
    encodings: (B, N=SeqLen), int64
    inps: inputs, (B, N=SeqLen, D), float16 or bfloat16
    attention_mask = None
    position_ids: (1, N=SeqLen), int64
    past_key_value: None
    output_attentions: bool = False
    use_cache: bool = False
    cache_position: (N=SeqLen), int64
    position_embeddings: tuple, FP16, (1, N=SeqLen, 128), (1, N=SeqLen, 128)
    """
    catcher: Catcher = Catcher()
    gpt_blocks: nn.ModuleList = model.model.layers

    gpt_block_0, gpt_blocks[0] = gpt_blocks[0], catcher
    use_cache, model.config.use_cache = model.config.use_cache, False
    model.model.embed_tokens.to(device=device)
    model.model.rotary_emb.to(device=device)

    inps: list[torch.Tensor] = []
    inp_kwargs: dict = {}
    for bi in range(0, len(encodings), batch_size):
        try:
            model(encodings[bi : bi + batch_size].to(device=device))
        except ValueError as value_error:
            inps.append(value_error.args[0][0].to(device=save_device))
            inp_kwargs: dict = value_error.args[1]

    gpt_blocks[0] = gpt_block_0
    model.model.embed_tokens.cpu()
    model.model.rotary_emb.cpu()
    model.config.use_cache = use_cache
    return torch.cat(inps, dim=0), move_to_device(inp_kwargs, save_device)


@torch.no_grad()
def quantize_llama(
        model: LlamaForCausalLM,
        encodings: torch.Tensor,
        device: torch.device,
        batch_size: int = 1,
        save_gpu_mem: bool = False,  # whether to save input tensors in cpu ram, may impact performance
) -> dict[str, dict[str, dict]]:
    """
    start quantize
    """
    _input, _output = 'input', 'output'
    cpu_device: torch.device = torch.device('cpu')
    inps, inp_kwargs = get_initial_inputs(model, encodings, device, batch_size, save_device=cpu_device if save_gpu_mem else device)
    use_cache, model.config.use_cache = model.config.use_cache, False
    dtype: torch.dtype = inps.dtype
    inp_kwargs_cpu: dict = move_to_device(inp_kwargs, device=cpu_device)

    results: dict[str, dict[str, dict]] = {
        'data': {},  # dict[str, dict[str, torch.Tensor | None]]
        'metrics': {},  # dict[str, dict[str, float]]
    }

    gpt_blocks: nn.ModuleList = model.model.layers

    for gi, gpt_block in enumerate(gpt_blocks):
        block_start_time: float = time.time()

        # find dependency info
        dependency_info: list[tuple] = extract_dependencies(gpt_block, [nn.Linear], inps.shape, dtype, cpu_device, inp_kwargs_cpu)

        # wrap layers
        wrapper_layers_dict: dict[str, RecorderWrapper] = {}
        for linear_layer_name, linear_layer_module in find_submodules_by_types(gpt_block, [nn.Linear]).items():
            linear_layer_wrapper: RecorderWrapper = RecorderWrapper(linear_layer_module)
            linear_layer_wrapper.stage = RecorderWrapper.stage_fake_mode
            wrapper_layers_dict[linear_layer_name] = linear_layer_wrapper
            find_and_replace_submodule_by_name(gpt_block, linear_layer_name, linear_layer_wrapper)

        # move norm layers to gpu
        gpt_block.input_layernorm.to(device=device)
        gpt_block.post_attention_layernorm.to(device=device)

        # start quantization
        for di, (quantizing_layer_names, to_release_layer_names) in enumerate(dependency_info):
            if quantizing_layer_names == [_output]:
                break

            # compute hessian
            hessian_hook: HessianHook = HessianHook()
            for quantizing_layer_name in quantizing_layer_names:
                quantizing_layer_wrapper: RecorderWrapper = wrapper_layers_dict[quantizing_layer_name]
                quantizing_layer_wrapper.hessian_hook = hessian_hook
                quantizing_layer_wrapper.stage = RecorderWrapper.stage_hessian_accumulation
            hidden_states: list[torch.Tensor] = []
            if save_gpu_mem:
                inp_kwargs: dict | None = move_to_device(inp_kwargs_cpu, device=device)
            for bi in range(0, len(inps), batch_size):
                try:
                    gpt_block(inps[bi : bi + batch_size].to(device=device), **inp_kwargs)
                except ValueError as value_error:
                    hidden_states.append(value_error.args[0].to(device=cpu_device if save_gpu_mem else device))
            if save_gpu_mem:
                inp_kwargs: dict | None = None

            # parent layers no longer needed: output fake tensors
            for to_release_layer_name in to_release_layer_names:
                if to_release_layer_name == _input:
                    inps = RecorderWrapper.fake_mode.from_tensor(inps)
                    continue
                to_release_layer_wrapper: RecorderWrapper = wrapper_layers_dict[to_release_layer_name]
                to_release_layer_wrapper.outs = []
                to_release_layer_wrapper.stage = RecorderWrapper.stage_fake_mode

            hessian_hook.invert(damp_ratio=1e-2, act_order=True)

            # quantize a layer
            for quantizing_layer_name in quantizing_layer_names:
                assert quantizing_layer_name not in [_input, _output]
                quantizing_layer_wrapper: RecorderWrapper = wrapper_layers_dict[quantizing_layer_name]
                weight: torch.Tensor = quantizing_layer_wrapper.module.weight.to(device=device)
                n_out_features, n_in_features = weight.shape
                group_sizes: torch.Tensor = torch.full([n_in_features // 128], 128, dtype=torch.int32, device=device)
                group_bit_widths: torch.Tensor = torch.full([n_in_features // 128], 4, dtype=torch.int32, device=device)
                gptq_result: dict[str, dict] = gptq_quant(
                    weight=weight,
                    hessian_hook=hessian_hook,
                    group_sizes=group_sizes,
                    group_bit_widths=group_bit_widths,
                    scale_bit_width=None,
                    gptq_use_kernel=True,
                    gptq_block_sizes=group_sizes,
                    quant_symmetric=False,
                    quant_mse=False,
                    quant_max_shrink=.8,
                    quant_n_grid=100,
                    quant_norm=2.4,
                    quant_use_kernel=False,
                    save_device=cpu_device if save_gpu_mem else device,
                )
                del weight
                quantizing_layer_wrapper.hessian_hook = None
                # log metrics
                canonical_name: str = f'model.layers.{gi}.{quantizing_layer_name}'
                metrics: dict[str, float] = gptq_result['metrics']
                results['metrics'][canonical_name] = metrics
                logging.debug(f'{canonical_name} {metrics}')
                # reconstruct layer and record outputs
                reconstructed_nn_linear: nn.Linear = reconstruct_nn_linear(gptq_result['quant_meta'], dtype=dtype, device=device)
                results['data'][canonical_name] = move_to_device(gptq_result['quant_meta'], cpu_device)
                quantizing_layer_wrapper.module = reconstructed_nn_linear
                del gptq_result
                for hidden_state in hidden_states:
                    quantizing_layer_wrapper.outs.append(reconstructed_nn_linear(hidden_state.to(device=device)).to(device=cpu_device if save_gpu_mem else device))
                reconstructed_nn_linear.cpu()
                quantizing_layer_wrapper.stage = RecorderWrapper.stage_replay_mode

            del hessian_hook, hidden_states

        # inputs of the next block
        outs: torch.Tensor = torch.empty(*inps.shape, dtype=dtype, device=cpu_device if save_gpu_mem else device)
        if save_gpu_mem:
            inp_kwargs: dict | None = move_to_device(inp_kwargs_cpu, device=device)
        for bi in range(0, len(inps), batch_size):
            outs[bi : bi + batch_size], = gpt_block(inps[bi : bi + batch_size].to(device=device), **inp_kwargs)
        if save_gpu_mem:
            inp_kwargs: dict | None = None
        inps: torch.Tensor = outs

        # move norm layers to cpu
        gpt_block.input_layernorm.cpu()
        gpt_block.post_attention_layernorm.cpu()

        # un-wrap layers
        for linear_layer_name, linear_layer_wrapper in wrapper_layers_dict.items():
            find_and_replace_submodule_by_name(gpt_block, linear_layer_name, linear_layer_wrapper.module)

        block_end_time: float = time.time()
        logging.info(f'finished block {gi} in {block_end_time - block_start_time:.2f} s')

    return results


@torch.no_grad()
def evaluate_llama(
        model: LlamaForCausalLM,
        encodings: torch.Tensor,
        device: torch.device,
        batch_size: int = 1,
) -> torch.Tensor:
    inps, inp_kwargs = get_initial_inputs(model, encodings, device, batch_size, save_device=device)
    use_cache, model.config.use_cache = model.config.use_cache, False

    layers: nn.ModuleList = model.model.layers
    outs: torch.Tensor = torch.empty_like(inps)

    for i, layer in tqdm(enumerate(layers), total=len(layers)):
        layer.to(device)
        for j in range(0, len(inps), batch_size):
            outs[j:j+batch_size], = layer(inps[j:j+batch_size].to(device=device), **inp_kwargs)
        layer.cpu()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm.to(device=device)
        for j in range(0, len(inps), batch_size):
            outs[j:j+batch_size] = model.model.norm(inps[j:j+batch_size].to(device=device))
        inps, outs = outs, inps
        model.model.norm.cpu()

    model.lm_head.to(device=device)
    loss_fct: nn.Module = nn.CrossEntropyLoss()
    nlls: list[torch.Tensor] = []
    for i in range(0, len(inps), batch_size):
        lm_logits = model.lm_head(inps[i:i+batch_size].to(device=device))  # (B, N=SeqLen, D)
        shift_logits = lm_logits[:, :-1, :]  # (B, N=SeqLen-1, D)
        shift_labels = encodings[i:i+batch_size, 1:].to(device=device)  # (B, N=SeqLen-1)
        neg_log_likelihood = loss_fct(shift_logits.flatten(end_dim=-2), shift_labels.flatten())
        nlls.extend([neg_log_likelihood] * len(lm_logits))
    ppl = torch.stack(nlls).to(dtype=torch.float32).mean().exp()
    model.lm_head.cpu()

    model.config.use_cache = use_cache
    return ppl
