import dataclasses
import json
import os.path
from typing import List

import torch
import argparse

from marshmallow import EXCLUDE

from model_data import ModelData


# reminder - just doing this for sft. no consideration of max output length b/c not doing any novel generation

# disclaimer- to make this project even remotely feasible, restricting scope to the options and behavior
# in the HuggingFace transformers/peft/bitsandbytes libraries
# Also only considering 4-bit quantization of frozen model weights, use of LoRA, and gradient checkpointing

def calc_num_trainable_weights(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                               lora_mlp: bool) -> float:
    trainable_param_count: float = 0.0

    if lora_embed:
        trainable_param_count += (model_data.vocab_size * lora_r + lora_r * model_data.model_dim)

    if lora_attn:
        out_proj_adapter_size = model_data.num_query_heads * model_data.head_size * lora_r + lora_r * model_data.model_dim
        query_proj_adapter_size = model_data.model_dim * lora_r + lora_r * model_data.head_size * model_data.num_query_heads
        key_proj_adapter_size = model_data.model_dim * lora_r + lora_r * model_data.head_size * model_data.num_kv_heads
        value_proj_adapter_size = model_data.model_dim * lora_r + lora_r * model_data.head_size * model_data.num_kv_heads
        attn_block_adapters_size = query_proj_adapter_size + key_proj_adapter_size + value_proj_adapter_size + out_proj_adapter_size
        trainable_param_count += attn_block_adapters_size * model_data.num_layers

    if lora_mlp:
        mlp_block_size: float = 0
        hidden_layer_result_width = model_data.feed_forward_hidden_dim
        if model_data.is_mlp_gated:
            hidden_layer_result_width /= 2
            # adding the gate proj
            mlp_block_size += model_data.model_dim * lora_r + lora_r * hidden_layer_result_width
        # up proj and down proj
        mlp_block_size += (model_data.model_dim * lora_r + lora_r * hidden_layer_result_width
                           + hidden_layer_result_width * lora_r + lora_r * model_data.model_dim)
        trainable_param_count += model_data.num_layers * mlp_block_size

    return trainable_param_count


def predict_activations_mem(model_data: ModelData, seq_len: int, batch_size: int) -> (float, float):
    """
    predicts information about the sizes of long-lived (within the scope of a mini-batch) activation tensors for a
    given model
    :param model_data: information about the model
    :param seq_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: a tuple with the size of the small long-lived activation tensor for this model and the size which a bunch
        of larger long-lived activation tensors all have in common
    """
    # from experimental analysis (look for "sl x bs x ..." in the spreadsheet under the column labelled #4), it is
    #  extremely unclear what the small activation tensor is scaling with (besides sequence length and batch size),
    #  but there's definitely something
    # Going with a base multiplier of 256 as a compromise between the results seen in different experiments
    small_activation_size = 256 * seq_len * batch_size

    # experimental analysis (look for "sl x bs x ..." in the spreadsheet under the column labelled #5) shows extremely
    #  consistent behavior for the larger activation tensors- they scale linearly with sequence length and batch size
    #  with model-specific base multipliers of 8192 for gemma2b and 12288 for gemma7b
    medium_activation_size = model_data.repeated_activation_tensor_scaling_factor * seq_len * batch_size

    return small_activation_size, medium_activation_size


# todo check whether the tiny (8 or 12 KiB) params tensors have the same count in every run of a particular model type
#  or whether they vary with the other configuration options like lora-r, lora-embed, lora-attn, and lora-mlp
#  if the former, their total size should be measured and added to each model's details json file, then used here

def predict_trainable_weights_mem(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                  lora_mlp: bool) -> float:
    bytes_per_weight = 4
    bytes_from_trainable_weights = bytes_per_weight * calc_num_trainable_weights(model_data, lora_r, lora_embed,
                                                                                 lora_attn, lora_mlp)
    return bytes_from_trainable_weights


def predict_optimizer_states_mem(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                 lora_mlp: bool) -> float:
    # depends on optimizer choice (frozen as paged-adamw-32bit for now), # trainable parameters, and choices about quantization of optimizer states (not relevant for now)

    # one article seemed to claim that a second copy of the current parameter value was part of the optimizer states for each parameter
    # need to try to confirm this empirically (not really, looks like adamw optimizer memory states are at worst ~2x the size of the trainable parameters memory, not 3x)

    # this is going to be complicated/unreliable because the optimizer is paged
    # for Gemma2b, there's this weird pattern where it's close to 2x the amount of params memory if amount of params
    #   memory is in single or low double digits of MiB, but then _sometimes_ above that point it's close to 1/2 the
    #   amount of params memory (e.g. scenario 12 with 95megs of params memory and 44megs of optimizer states memory)
    #   but then once you're in the hundreds of megabytes of params memory, it's always just 4.7megs (except this 4.7megs
    #   of optimizer memory behavior also occurs with just 59megs of params memory in cases like scenarios 16n17)
    # Then the behavior for Gemma7b looks nothing like that at all :''(
    #   it never forces to just 4.7megs of vram once the params memory is in the hundreds of megabytes; optimizer states
    #   are 1.5-2x the size of params memory for a few cases where params memory was in the 15-55meg range, but then
    #   for cases with params memory higher than that the optimizer memory varies from 1/5 to 3/5 of params memory

    # rough-and-ready approximation of the above:
    trainable_params_mem = predict_trainable_weights_mem(model_data, lora_r, lora_embed, lora_attn, lora_mlp)
    if trainable_params_mem < 75_000_000:
        return trainable_params_mem * 2
    else:
        return trainable_params_mem / 2


test_vram_capacity: None|int = None
def measure_vram_capacity() -> int:
    return torch.cuda.mem_get_info()[0] if test_vram_capacity is None else test_vram_capacity


def calculate_quantization_peak(model_data: ModelData) -> int:
    """
    This is the peak of memory usage that occurs near the end of the process of quantizing the frozen base model weights
    :param model_data: information about the model being fine-tuned
    :return: the size in bytes of this peak of memory usage
    """
    return model_data.initial_massive_params_chunk_size + model_data.total_size_of_frozen_weight_small_tensors + model_data.persistent_massive_params_chunk_size


def calculate_static_vram_usage(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                lora_mlp: bool) -> float:
    # total size of tensors that are present for all peaks after the quantization peak
    return (model_data.total_size_of_frozen_weight_small_tensors + model_data.persistent_massive_params_chunk_size +
            predict_trainable_weights_mem(model_data, lora_r, lora_embed, lora_attn, lora_mlp) +
            predict_optimizer_states_mem(model_data, lora_r, lora_embed, lora_attn, lora_mlp))


# todo final predictor might need fudge factor for the persistent peak calculation errors that are recorded in the experiment results spreadsheet (on the order of tens of KiB or sometimes 100-200KiB)
#  this fudge factor could be tracked separately for each peak type (since it differed in practice between peak types in the spreadsheet for gemma 2b)


def calculate_forward_pass_highest_layer_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                              lora_mlp: bool,
                                              sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from the short-lived and large 'input' tensors that're created in the
    forward pass through the highest layer of the model
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """
    # somehow depends on model size (maybe model_data.model_dim?) because I can't reproduce this being a peak with gemma2b
    #  I suspect it also depends on sequence length and batch size but haven't checked that yet

    # this will require hard-coding something in the gemma2b vs 7b models' jsons about how the former only has
    # num_layers+3 activation tensors at this point while the latter has num_layers+4 activation tensors at this point

    return -1  # todo implement this in future, after gathering more data


def calculate_central_activations_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                       lora_mlp: bool,
                                       sequence_len: int, batch_size: int) -> float:
    """
    This is the peak from stacked activations blocks in between the processing of the highest layer for the forward
    pass and the processing of the highest layer for the backward pass
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """

    small_activations_size, medium_activations_size = predict_activations_mem(model_data, sequence_len, batch_size)
    activations_buildup_size = small_activations_size + medium_activations_size * (model_data.num_layers + 3)

    # the stacked big activation tensors here scale in a confusingly noisy way, but there's a clear (if still
    # perplexing) trend that their total size scales linearly with sequence length,
    # when batch size is one, their total size scales linearly with sequence length with constant factor of ~2million
    # when batch size >1, their total size scales linearly with both sequence-length/batch-size with constant factor of
    # ~3million

    large_stacked_activation_tensors = 2_000_000 * sequence_len
    if batch_size > 1:
        large_stacked_activation_tensors = 3_000_000 * sequence_len * batch_size
    elif batch_size < 1:
        raise Exception("Batch size must be at least 1")

    return (calculate_static_vram_usage(model_data, lora_r, lora_embed, lora_attn, lora_mlp)
            + activations_buildup_size + large_stacked_activation_tensors)


def calculate_central_autograd_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                    lora_mlp: bool,
                                    sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from stacked autograd blocks (on top of a big activations block) in between the processing of the
    highest layer for the forward pass and the processing of the highest layer for the backward pass
    This also comes right after the "central activations peak"
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """

    small_activations_size, medium_activations_size = predict_activations_mem(model_data, sequence_len, batch_size)
    activations_buildup_size = small_activations_size + medium_activations_size * (model_data.num_layers + 2)

    # accumulated activations at this peak and also the big stacked activation/autograd_detail tensors at this peak
    # seem to scale almost exactly linearly with sequence length for both gemma2b and gemma7b
    # in this case, both also seem to scale exactly linearly with batch size
    # multipliers are approximate
    large_activation_tensor = 1_000_000 * sequence_len * batch_size
    large_stacked_autograd_tensors = 2_000_000 * sequence_len * batch_size

    return (calculate_static_vram_usage(model_data, lora_r, lora_embed, lora_attn, lora_mlp) + activations_buildup_size
            + large_activation_tensor + large_stacked_autograd_tensors)


def calculate_backward_pass_highest_layer_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                               lora_mlp: bool,
                                               sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from the spikes of large/short-lived temporary tensors being allocated
    during the backward pass through the highest layer of the model (when almost all of the built-up activations
    tensors are still allocated)
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """

    return -1  # todo implement this in future, after gathering more data


def calculate_backward_pass_lowest_layer_mid_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                                  lora_mlp: bool,
                                                  sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from the middle of the backward pass through the lowest layer of the model
    when gradients have built up but not yet been applied to the trainable parameters
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """
    small_activations_size, medium_activations_size = predict_activations_mem(model_data, sequence_len, batch_size)
    remaining_activations_size = small_activations_size + medium_activations_size * 2

    # formula for built-up gradients: I can't see any trend in the experimental data between sequence length/batch-size
    #  and the size of the built-up gradients at this peak
    #  The best I can do is note that the build of gradients at this peak are almost always 80-120% of the size of the
    #  trainable parameters in memory; I'm going with a rough estimate of assuming that they're equal
    gradients_buildup_estimate = predict_trainable_weights_mem(model_data, lora_r, lora_embed, lora_attn, lora_mlp)

    # lora params for mlp blocks contribute to the extra temporary tensor above the new grad (the 3rd small one under column labelled #22 in spreadsheet), but not attn blocks or embed matrix (if no mlp lora, that 3rd temporary tensor above the new grads doesn't show up)
    #   that 3rd tensor scales linearly with batch size
    #   scales sort-of linearly with sequence length?:
    #       when lora-r fixed to 32, for gemma7b, increasing seq_len 64->96->128 yields 6mib->9mib->12mib for this tensor, but seq_len 16 yields 2mib (maybe pytorch coarse-grained allocation, and it should've been 1.5mib?)
    #           so here the slope was 3*2^15 increase in bytes of this tensor per token of increase in sequence length
    #       comparing gemma7b scenarios 3-5, it doesn't seem to scale at all with lora-r
    #       For gemma 2b:
    #           definitely doesn't scale with lora-r
    #           seems to scale as 2mib per 32 tokens of sequence length, so 2^21 bytes per 2^5 tokens of sequence length, so 2^16 bytes per token of sequence length
    # I don't want to add a new hacky detail to the model jsons right now, so I'll split the difference between a scaling rate of 2*2^15 and 3*2^15 with 2.5*2^15
    temp_tensor_for_mlp_lora_above_new_grads = 0
    if lora_mlp:
        temp_tensor_for_mlp_lora_above_new_grads = sequence_len * batch_size * 2.5 * 2 ** 15

    # for the other tensors (under column #19 in the spreadsheet):
    #  scales ~linearly with sequence length and batch size
    #  doesn't scale with lora-r
    # very different scaling base multipliers for gemma 2b and 7b (~321_500 and ~500_000 respectively, based on partial survey of experimental data)
    misc_temp_tensors_in_back_pass_spike = sequence_len * batch_size * model_data.backward_pass_spike_temporary_tensors_scaling_factor

    # the two highest 'temporary' tensors in this peak G are 144.0mib/288.0mib for gemma7b but only 64.0mib/128.0mib for gemma2b

    return (calculate_static_vram_usage(model_data, lora_r, lora_embed, lora_attn, lora_mlp)
            + remaining_activations_size + gradients_buildup_estimate + temp_tensor_for_mlp_lora_above_new_grads
            + misc_temp_tensors_in_back_pass_spike + model_data.large_temporary_tensors_in_backward_pass)


def calculate_backward_pass_lowest_layer_late_peak(model_data: ModelData, lora_r: int, lora_embed: bool,
                                                   lora_attn: bool, lora_mlp: bool,
                                                   sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from the spikes of large/short-lived temporary tensors being allocated
    near the end of the backward pass through the lowest layer of the model, when the maximum amount of gradient
    tensors have been built up in GPU memory (but not yet applied to the trainable parameters)
    This is only relevant (in addition to the backward_pass_lowest_layer_*mid*_peak function) if large enough gradient
    tensors are created at various points in the backward pass through the lowest layer of the model that their creation
    outweighs the gradual deallocation of temporary tensors over the second half of the backward pass through that layer
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """
    if not lora_mlp:
        # this peak scenario simply can't occur if lora was not applied to MLP blocks because the middle temporary
        # spikes during the backward-pass through a layer will be taller than the final temporary spike for that
        # layer's backward pass
        return 0
    else:
        return -1  # todo implement this in future, after gathering more data


def calculate_end_of_batch_autograd_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                         lora_mlp: bool,
                                         sequence_len: int, batch_size: int) -> float:
    """
    This calculates the height of the peak from the spikes of 2 stacked autograd_detail tensors being allocated by
    Pytorch shortly after the end of the backward pass, when the gradient tensors have all collected in GPU memory
    but haven't been applied to the trainable parameters yet
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """
    if not lora_embed:
        return 0  # this peak simply doesn't occur if lora was not applied to embedding matrix
    else:
        return -1  # todo implement this in future, after gathering more data


def calc_config_vram_utilization(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool, lora_mlp: bool,
                                 sequence_len: int, batch_size: int) -> float:
    available_vram = measure_vram_capacity()
    highest_peak_vram_usage_for_config = predict_peak_vram_usage(model_data, lora_r, lora_embed, lora_attn, lora_mlp,
                                                                 sequence_len, batch_size)
    return highest_peak_vram_usage_for_config / available_vram


# skips quantization peak because that isn't specific to a particular configuration (and so is checked separately in the main function)
def predict_peak_vram_usage(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool, lora_mlp: bool,
                            sequence_len: int, batch_size: int) -> float:
    return max(
        calculate_forward_pass_highest_layer_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len,
                                                  batch_size),
        calculate_central_activations_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len,
                                           batch_size),
        calculate_central_autograd_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len, batch_size),
        calculate_backward_pass_highest_layer_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len,
                                                   batch_size),
        calculate_backward_pass_lowest_layer_mid_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len,
                                                      batch_size),
        calculate_backward_pass_lowest_layer_late_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp,
                                                       sequence_len, batch_size),
        calculate_end_of_batch_autograd_peak(model_data, lora_r, lora_embed, lora_attn, lora_mlp, sequence_len,
                                             batch_size)
    )


@dataclasses.dataclass
class TrainingConfiguration:
    lora_r: int
    lora_embed: bool
    lora_attn: bool
    lora_mlp: bool
    sequence_len: int
    batch_size: int
    utilization: float  # all other fields are config details, this is the resulting config's vram utilization


lora_r_possibilities = (1, 2, 4, 8, 16, 32, 64, 128)
seq_len_possibilities = (16, 32, 64, 128, 256, 512)
batch_size_possibilities = (1, 2, 4, 8)


def main():
    parser = argparse.ArgumentParser(description="Predict peak VRAM usage for a given model and LoRA settings")
    parser.add_argument("model", required=True, type=str,
                        help="The org and type of the model to predict VRAM usage for (e.g. google/gemma_2b for the 2b "
                             "size of one of Google's Gemma models or, once supported, meta_llama/llama3_8b for the 8b "
                             "size of Meta's Llama3")
    parser.add_argument("-r", "--lora-r", type=int, help="The rank value to use for LoRA matrices")
    parser.add_argument("-e", "--lora-embed", type=bool,
                        help="Whether to apply LoRA to the embedding matrix (if not specified, report will try both "
                             "true and false for this)")
    parser.add_argument("-a", "--lora-attn", type=bool,
                        help="Whether to apply LoRA to attention blocks (if not specified, report will try both true "
                             "and false for this)")
    parser.add_argument("-m", "--lora-mlp", type=bool,
                        help="Whether to apply LoRA to MLP blocks (if not specified, report will try both true and "
                             "false for this)")
    parser.add_argument("-s", "--sequence-len", type=int, help="The sequence length to use for the SFT")
    parser.add_argument("-b", "--batch-size", type=int, help="The batch size to use for the SFT")
    parser.add_argument("--num-configs", type=int, default=10,
                        help="The number of viable configurations to report (default 10)")
    parser.add_argument("--test-set-gpu-mem-capacity", type=int,
                        help="The amount of GPU memory to pretend is available for testing purposes")

    args = parser.parse_args()

    if not torch.is_cuda_available() and args.test_set_gpu_mem_capacity is None:
        raise Exception("CUDA is not available")

    if args.test_set_gpu_mem_capacity is not None:
        global test_vram_capacity
        test_vram_capacity = args.test_set_gpu_mem_capacity
    else:
        free_vram, total_vram = torch.cuda.mem_get_info()
        existing_vram_usage = total_vram - free_vram
        print(f"{existing_vram_usage} bytes of VRAM are already in use, out of max GPU capacity of {total_vram} bytes")

    if args.model.count("/") != 1:
        raise Exception("Model arg must be in the form 'org_name/model_family_and_size' (size is needed because "
                        "architectural details often differ between sizes within a family)")

    model_details_file_path = f"model_details/{args.model}.json"
    if not os.path.exists(model_details_file_path):
        raise Exception(f"Model details file not found at ${os.getcwd()}/{model_details_file_path}")
    model_data = ModelData.Schema().load(json.load(open(model_details_file_path)), unknown=EXCLUDE)

    lora_r_candidates: List[int] = [args.lora_r] if args.lora_r is not None else lora_r_possibilities
    lora_embed_candidates: List[bool] = [args.lora_embed] if args.lora_embed is not None else [False, True]
    lora_attn_candidates: List[bool] = [args.lora_attn] if args.lora_attn is not None else [False, True]
    lora_mlp_candidates: List[bool] = [args.lora_mlp] if args.lora_mlp is not None else [False, True]
    seq_len_candidates: List[int] = [args.sequence_len] if args.sequence_len is not None else seq_len_possibilities
    batch_size_candidates: List[int] = [args.batch_size] if args.batch_size is not None else batch_size_possibilities

    quantization_peak = calculate_quantization_peak(model_data)
    if quantization_peak > measure_vram_capacity():
        raise Exception(f"Model's quantization peak {quantization_peak} exceeds available VRAM")

    viable_configs: List[TrainingConfiguration] = []

    # the below early stopping logic relies on the assumption that each list of candidates is in ascending order of VRAM usage
    for curr_lora_embed in lora_embed_candidates:
        any_viable_configs_for_curr_lora_embed = False
        for curr_lora_attn in lora_attn_candidates:
            any_viable_configs_for_curr_lora_attn = False
            for curr_lora_mlp in lora_mlp_candidates:
                any_viable_configs_for_curr_lora_mlp = False
                for curr_lora_r in lora_r_candidates:
                    any_viable_configs_for_curr_lora_r = False
                    for curr_seq_len in seq_len_candidates:
                        any_viable_configs_for_curr_seq_len = False
                        for curr_batch_size in batch_size_candidates:
                            utilization = calc_config_vram_utilization(model_data, curr_lora_r, curr_lora_embed,
                                                                       curr_lora_attn, curr_lora_mlp, curr_seq_len,
                                                                       curr_batch_size)
                            if utilization < 1:
                                any_viable_configs_for_curr_lora_embed = True
                                any_viable_configs_for_curr_lora_attn = True
                                any_viable_configs_for_curr_lora_mlp = True
                                any_viable_configs_for_curr_lora_r = True
                                any_viable_configs_for_curr_seq_len = True
                                viable_configs.append(
                                    TrainingConfiguration(curr_lora_r, curr_lora_embed, curr_lora_attn, curr_lora_mlp,
                                                          curr_seq_len, curr_batch_size, utilization))
                            else:
                                break
                        if not any_viable_configs_for_curr_seq_len:
                            break
                    if not any_viable_configs_for_curr_lora_r:
                        break
                if not any_viable_configs_for_curr_lora_mlp:
                    break
            if not any_viable_configs_for_curr_lora_attn:
                break
        if not any_viable_configs_for_curr_lora_embed:
            break

    if len(viable_configs) == 0:
        print("No viable configurations found")
    else:
        viable_configs = sorted(viable_configs, key=lambda x: x.utilization, reverse=True)
        if len(viable_configs) > args.num_configs:
            viable_configs = viable_configs[:args.num_configs]

        print(f"Top {args.num_configs} viable configurations:\n\t" + "\n\t".join(
            [str(config) for config in viable_configs]))


if __name__ == "__main__":
    main()
