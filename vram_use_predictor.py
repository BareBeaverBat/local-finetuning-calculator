import dataclasses
import json
import os.path
from typing import List

import torch
import argparse

from marshmallow import EXCLUDE

from model_data import ModelData, QuantizationLevels


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
            #adding the gate proj
            mlp_block_size += model_data.model_dim*lora_r + lora_r*hidden_layer_result_width
        #up proj and down proj
        mlp_block_size += (model_data.model_dim * lora_r + lora_r * hidden_layer_result_width
                           + hidden_layer_result_width*lora_r + lora_r*model_data.model_dim)
        trainable_param_count += model_data.num_layers * mlp_block_size
        #the gating thing does matter!!! it affects the size of the down projection adapter!!

    return trainable_param_count


def predict_activations_mem() -> float:
    # batch size and sequence length are important factors here
    # pytorch/transformers defaults to recomputing activations to save memory/io iff you set gradient_checkpointing=True
    # which, based on the medium post linked by the docs for that, seems like it probably only saves the activations
    # for sqrt-n layers?
    # mem usage for activations from full fine-tuning with no gradient checkpointing is pretty well described by this (assumes layer norm and multi-head attention)
    # https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff
    # but with LoRA and gradient checkpointing complicating the picture (also, gemma uses RMSNorm rather than LayerNorm, which I think might halve the activation memory usage from those norm blocks),
    # I'm going to need to do tons of experiments
    #  e.g. with gradient checkpointing, does transformers save every single sub-element of the activations of the selected layers (sqrt(L) layers chosen this way automatically), or just the final output from each layer? or something else?

    return -1


def predict_frozen_weights_mem(model_data: ModelData, quantization_level: QuantizationLevels,
                               double_quantize: bool) -> float:
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


# todo check whether the tiny (8 or 12 KiB) params tensors have the same count in every run of a particular model type
#  or whether they vary with the other configuration options like lora-r, lora-embed, lora-attn, and lora-mlp
#  if the former, their total size should be measured and added to each model's details json file, then used here

def predict_trainable_weights_mem(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                  lora_mlp: bool) -> float:
    bytes_per_weight = 4
    bytes_from_trainable_weights = bytes_per_weight * calc_num_trainable_weights(model_data, lora_r, lora_embed,
                                                                                 lora_attn, lora_mlp)
    return bytes_from_trainable_weights


def predict_optimizer_states_mem() -> float:
    # depends on optimizer choice, # trainable parameters, and choices about quantization of optimizer states

    # one article seemed to claim that a second copy of the current parameter value was part of the optimizer states for each parameter
    # need to try to confirm this empirically

    # this is going to be complicated/unreliable because the optimizer is paged
    # todo implement before delivery!

    return -1


def predict_gradients_mem() -> int:
    # depends on # trainable parameters, batch size, and whether/how-much gradients are quantized (not recommended)
    # ALSO SEQUENCE LENGTH
    # might want to support gradient accumulation (need to check- it might have exactly zero effect on peak vram usage)

    return -1


def measure_vram_capacity() -> int:
    return torch.cuda.mem_get_info()[0]


def calculate_quantization_peak(model_data: ModelData) -> int:
    """
    This is the peak of memory usage that occurs near the end of the process of quantizing the frozen base model weights
    :param model_data: information about the model being fine-tuned
    :return: the size in bytes of this peak of memory usage
    """
    return model_data.initial_massive_params_chunk_size + model_data.total_size_of_frozen_weight_small_tensors + model_data.persistent_massive_params_chunk_size


# todo make helper method for the total size of tensors that are present for all peaks after the quantization peak
# frozen weights tensors, persistent massive params chunk, trainable params, and optimizer state


# todo final predictor might need fudge factor for the persistent peak calculation errors that are recorded in the experiment results spreadsheet (on the order of tens of KiB or sometimes 100-200KiB)
#  this fudge factor could be tracked separately for each peak type (since it differed in practice between peak types in the spreadsheet for gemma 2b)


def calculate_forward_pass_highest_layer_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                              lora_mlp: bool,
                                              sequence_len: int, batch_size: int) -> float:
    """
    This is the peak from todo finish this explanation before delivery
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

    return -1  # todo implement this b4 delivery based on data collected so far


def calculate_central_autograd_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                    lora_mlp: bool,
                                    sequence_len: int, batch_size: int) -> float:
    """
    This is the peak from stacked autograd blocks (on top of a big activations block) in between the processing of the
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

    return -1  # todo implement this b4 delivery based on data collected so far


def calculate_backward_pass_highest_layer_peak(model_data: ModelData, lora_r: int, lora_embed: bool, lora_attn: bool,
                                               lora_mlp: bool,
                                               sequence_len: int, batch_size: int) -> float:
    """
    todo at least document/explain this before delivery
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
    todo document/explain this before delivery
    :param model_data: information about the model being fine-tuned
    :param lora_r: the rank of the LoRA matrices
    :param lora_embed: whether to add a LoRA adapter to the embedding matrix
    :param lora_attn: whether to add LoRA adapters to the attention blocks
    :param lora_mlp: whether to add LoRA adapters to the MLP blocks
    :param sequence_len: the sequence length for each training record for the current training configuration
    :param batch_size: the batch size for the current training configuration
    :return: the size in bytes of this peak of memory usage
    """

    return -1  # todo implement this b4 delivery based on data collected so far


def calculate_backward_pass_lowest_layer_late_peak(model_data: ModelData, lora_r: int, lora_embed: bool,
                                                   lora_attn: bool, lora_mlp: bool,
                                                   sequence_len: int, batch_size: int) -> float:
    """
    todo at least document/explain this before delivery
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
    todo at least document/explain this before delivery
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
    if not torch.is_cuda_available():
        raise Exception("CUDA is not available")

    parser = argparse.ArgumentParser(description="Predict peak VRAM usage for a given model and LoRA settings")
    parser.add_argument("model", required=True, type=str,
                        help="The org and type of the model to predict VRAM usage for (e.g. google/gemma_2b for the 2b "
                             "size of one of Google's Gemma models or, once supported, meta_llama/llama3_8b for the 8b "
                             "size of Meta's Llama3")
    parser.add_argument("-r", "--lora_r", type=int, help="The rank value to use for LoRA matrices")
    parser.add_argument("-e", "--lora_embed", type=bool,
                        help="Whether to apply LoRA to the embedding matrix (if not specified, report will try both "
                             "true and false for this)")
    parser.add_argument("-a", "--lora_attn", type=bool,
                        help="Whether to apply LoRA to attention blocks (if not specified, report will try both true "
                             "and false for this)")
    parser.add_argument("-m", "--lora_mlp", type=bool,
                        help="Whether to apply LoRA to MLP blocks (if not specified, report will try both true and "
                             "false for this)")
    parser.add_argument("-s", "--sequence_len", type=int, help="The sequence length to use for the SFT")
    parser.add_argument("-b", "--batch_size", type=int, help="The batch size to use for the SFT")
    parser.add_argument("--num_configs", type=int, default=10,
                        help="The number of viable configurations to report (default 10)")

    args = parser.parse_args()

    if args.model.count("/") != 1:
        raise Exception("Model arg must be in the form 'org_name/model_family_and_size' (size is needed because "
                        "architectural details often differ between sizes within a family)")

    model_details_file_path = f"model_details/{args.model}.json"
    if not os.path.exists(model_details_file_path):
        raise Exception(f"Model details file not found at ${os.getcwd()}/{model_details_file_path}")
    model_data = ModelData.Schema().load(json.load(open(model_details_file_path)), unknown=EXCLUDE)

    lora_r_candidates: List[int] = [args.lora_r] if args.lora_r is not None else lora_r_possibilities
    lora_embed_candidates: List[bool] = [args.lora_embed] if args.lora_embed is not None else [True, False]
    lora_attn_candidates: List[bool] = [args.lora_attn] if args.lora_attn is not None else [True, False]
    lora_mlp_candidates: List[bool] = [args.lora_mlp] if args.lora_mlp is not None else [True, False]
    seq_len_candidates: List[int] = [args.sequence_len] if args.sequence_len is not None else seq_len_possibilities
    batch_size_candidates: List[int] = [args.batch_size] if args.batch_size is not None else batch_size_possibilities

    free_vram, total_vram = torch.cuda.mem_get_info()
    existing_vram_usage = total_vram - free_vram
    print(f"{existing_vram_usage} bytes of VRAM are already in use, out of max GPU capacity of {total_vram} bytes")

    quantization_peak = calculate_quantization_peak(model_data)
    if quantization_peak > measure_vram_capacity():
        raise Exception(f"Model's quantization peak {quantization_peak} exceeds available VRAM")

    viable_configs: List[TrainingConfiguration] = []

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
    g2_model = ModelData.Schema().load(json.load(open("model_details/google/gemma_2b.json")), unknown=EXCLUDE)
    g2_s4_err = 313_927_920 - predict_trainable_weights_mem(g2_model, 64, False, True, True)
    g2_s5_err = 379_998_208 - predict_trainable_weights_mem(g2_model, 64, True, True, True)
    g2_s6_err = 23_891_968 - predict_trainable_weights_mem(g2_model, 4, True, True, True)
    g2_s7_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    # g2_s8_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    # g2_s9_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    # g2_s10_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    # g2_s11_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    # g2_s12_err = 95_113_216 - predict_trainable_weights_mem(g2_model, 16, True, True, True)
    g2_s13_err = 14_897_152 - predict_trainable_weights_mem(g2_model, 16, False, True, False)
    g2_s14_err = 59_133_952 - predict_trainable_weights_mem(g2_model, 64, False, True, False)
    g2_s15_err = 3_837_952 - predict_trainable_weights_mem(g2_model, 4, False, True, False)
    # g2_s16_err = 59_133_952 - predict_trainable_weights_mem(g2_model, 64, False, True, False)
    # g2_s17_err = 59_125_760 - predict_trainable_weights_mem(g2_model, 64, False, True, False)
    g2_s19_err = 1_994_752 - predict_trainable_weights_mem(g2_model, 2, False, True, False)
    # g2_s20_err = 1_994_752 - predict_trainable_weights_mem(g2_model, 2, False, True, False)
    # g2_s21_err = 379_998_208 - predict_trainable_weights_mem(g2_model, 64, True, True, True)
    g2_s22_err = 125_194_240 - predict_trainable_weights_mem(g2_model, 64, True, True, False)
    g2_s23_err = 256_236_928 - predict_trainable_weights_mem(g2_model, 128, True, True, False)

    g7_model = ModelData.Schema().load(json.load(open("model_details/google/gemma_7b.json")), unknown=EXCLUDE)
    g7_s1_err = 17_963_008 - predict_trainable_weights_mem(g7_model, 4, True, True, False)
    g7_s2_err = 55_121_920 - predict_trainable_weights_mem(g7_model, 4, True, True, True)
    # g7_s3_err = 55_121_920 - predict_trainable_weights_mem(g7_model, 4, True, True, True)
    g7_s4_err = 217_995_264 - predict_trainable_weights_mem(g7_model, 16, True, True, True)
    g7_s5_err = 433_543_168 - predict_trainable_weights_mem(g7_model, 32, True, True, True)
    # g7_s6_err = 433_543_168 - predict_trainable_weights_mem(g7_model, 32, True, True, True)
    # g7_s7_err = 433_543_168 - predict_trainable_weights_mem(g7_model, 32, True, True, True)
    # g7_s8_err = 217_995_264 - predict_trainable_weights_mem(g7_model, 16, True, True, True)
    # g7_s9_err = 217_995_264 - predict_trainable_weights_mem(g7_model, 16, True, True, True)
    g7_s10_err = 446_781_440 - predict_trainable_weights_mem(g7_model, 48, False, False, True)
    g7_s11_err = 154_490_880 - predict_trainable_weights_mem(g7_model, 48, False, True, False)
    g7_s12_err = 50_085_888 - predict_trainable_weights_mem(g7_model, 48, True, False, False)
    g7_s13_err = 400_381_952 - predict_trainable_weights_mem(g7_model, 32, False, True, True)
    g7_s14_err = 330_782_720 - predict_trainable_weights_mem(g7_model, 32, True, False, True)
    print("dummy statement")


    # todo test predict_optimizer_states_mem() against collected data for gemma2b and gemma7b

    # main()
    pass
