import torch

from model_data import ModelData, QuantizationLevels


# reminder- just doing this for sft. no consideration of max output length b/c not doing any novel generation

# disclaimer- to make this project even remotely feasible, restricting scope to the options and behavior
# in the HuggingFace transformers/peft/bitsandbytes libraries

def calc_num_trainable_weights(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool, lora_mlp: bool):
    trainable_param_count: int = 0

    if lora_embed:
        trainable_param_count += model_data.vocab_size * lora_r + lora_r * model_data.model_dim

    if lora_attn:
        train_params_per_attn_block_in_the_out_proj = (
                model_data.num_query_heads * model_data.head_size * lora_r + lora_r * model_data.model_dim)
        trainable_param_count_per_attn_block = (
                (model_data.model_dim * lora_r + lora_r * model_data.head_size)
                * (model_data.num_query_heads + 2 * model_data.num_kv_heads)
                + train_params_per_attn_block_in_the_out_proj)
        trainable_param_count += trainable_param_count_per_attn_block * model_data.num_layers

    if lora_mlp:
        trainable_param_count += model_data.num_layers * (
                (model_data.model_dim * lora_r + lora_r * model_data.feed_forward_hidden_dim)
                + (model_data.feed_forward_hidden_dim * lora_r + lora_r * model_data.model_dim))

    return trainable_param_count


def predict_activations_mem():
    # batch size and sequence length are important factors here
    # pytorch/transformers defaults to recomputing activations to save memory/io iff you set gradient_checkpointing=True
    # which, based on the medium post linked by the docs for that, seems like it probably only saves the activations
    # for sqrt-n layers?
    # mem usage for activations from full fine-tuning with no gradient checkpointing is pretty well described by this (assumes layer norm and multi-head attention)
    # https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff
    # but with LoRA and gradient checkpointing complicating the picture (also, gemma uses RMSNorm rather than LayerNorm, which I think might halve the activation memory usage from those norm blocks),
    # I'm going to need to do tons of experiments
    #  e.g. with gradient checkpointing, does transformers save every single sub-element of the activations of the selected layers (sqrt(L) layers chosen this way automatically), or just the final output from each layer? or something else?

    return -1.0


def predict_frozen_weights_mem(model_data: ModelData, quantization_level: QuantizationLevels, double_quantize: bool):
    # depends on model size, choices about quantization of weights (including whether to double/nested quantize)
    # need to keep track of both embedding and non-embedding weights!
    frozen_weights_mem = 0.0

    per_param_byte_usage = quantization_level.value / 8
    if double_quantize:
        per_param_byte_usage -= 0.4

    # todo find out experimentally if bitsandbytes also applies to embedding weights

    frozen_weights_mem += model_data.num_non_embed_params * per_param_byte_usage

    # todo confirm formula from bitsandbytes docs about effect of nested quantization (0.4bits per param)

    return frozen_weights_mem


def predict_trainable_weights_mem(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                  lora_mlp: bool):
    # depends on which parts of the model you're applying lora to,
    # and choices about quantization of weights
    # todo double check whether this is still true for lora weights when doing quantization of the rest
    # particularly concerned b/c of the bitsandbytes setting about compute dtype being float16
    bytes_per_weight = 4  # or 2??

    bytes_from_trainable_weights = bytes_per_weight * calc_num_trainable_weights(model_data, lora_r, lora_embed,
                                                                                 lora_attn, lora_mlp)
    # todo use calc_num_trainable_weights()
    # don't inline it b/c that can be used again by predict_optimizer_states_mem

    return bytes_from_trainable_weights


def predict_optimizer_states_mem():
    # depends on optimizer choice, # trainable parameters, and choices about quantization of optimizer states

    # one article seemed to claim that a second copy of the current parameter value was part of the optimizer states for each parameter
    # need to try to confirm this empirically
    return -1.0


def predict_gradients_mem():
    # depends on # trainable parameters, batch size, and whether/how-much gradients are quantized (not recommended)
    # ALSO SEQUENCE LENGTH
    # might want to support gradient accumulation (need to check- it might have exactly zero effect on peak vram usage)

    return -1.0


# todo final predictor might need fudge factor for the persistent peak calculation errors that are recorded in the experiment results spreadsheet (on the order of tens of KiB or sometimes 100-200KiB)

# todo final predictor should start by checking the "peak at end of quantization" candidate and seeing if that exceeds available vram


def measure_vram_capacity() -> int:
    return torch.cuda.mem_get_info()[0]


def calculate_quantization_peak(model_data: ModelData) -> int:
    return model_data.initial_massive_params_chunk_size + model_data.total_size_of_frozen_weight_small_tensors + model_data.persistent_massive_params_chunk_size

def calculate_forward_pass_highest_layer_peak(model_data: ModelData) -> int:
    return -1#todo implement this in future, after gathering more data
    #todo at least document/explain this before delivery

def calculate_central_activations_peak(model_data: ModelData) -> int:
    """
    This is the peak from stacked activations blocks in between the processing of the highest layer for the forward 
    pass and the processing of the highest layer for the backward pass  
    :return: the size in bytes of this peak of memory usage
    """
    return -1#todo implement this b4 delivery

def calculate_central_autograd_peak(model_data: ModelData) -> int:
    """
    This is the peak from stacked autograd blocks (on top of a big activations block) in between the processing of the 
    highest layer for the forward pass and the processing of the highest layer for the backward pass
    This also comes right after the "central activations peak"  
    :return: the size in bytes of this peak of memory usage
    """
    return -1#todo implement this b4 delivery

def calculate_backward_pass_highest_layer_peak(model_data: ModelData) -> int:
    return -1#todo implement this in future, after gathering more data
    #todo at least document/explain this before delivery

def calculate_backward_pass_lowest_layer_mid_peak(model_data: ModelData) -> int:
    return -1#todo implement this b4 delivery

def calculate_backward_pass_lowest_layer_late_peak(model_data: ModelData) -> int:
    return -1#todo implement this in future, after gathering more data
    #todo at least document/explain this before delivery

def calculate_end_of_batch_autograd_peak(model_data: ModelData) -> int:
    return -1#todo implement this in future, after gathering more data
    #todo at least document/explain this before delivery

# todo in main method, throw error if torch.is_cuda_available() is false
