
#reminder- just doing this for sft. no consideration of max output length b/c not doing any novel generation

#disclaimer- to make this project even remotely feasible, restricting scope to the options and behavior
# in the HuggingFace transformers/peft/bitsandbytes libraries

def calc_num_trainable_weights():
    # assuming use of lora: number of model layers, lora-r, which parts of model to apply lora to
    #   (query, key, value, projection, mlp)
    # I'm ignoring lora matrices being inserted in initial embedding layer (for now), and also the idea of doing a lora matrix for the final output/head layer (seems very unlikely to be useful)




    return -1

def predict_activations_mem():
    #batch size and sequence length are important factors here
    #pytorch/transformers defaults to recomputing activations to save memory/io iff you set gradient_checkpointing=True
    # which, based on the medium post linked by the docs for that, seems like it probably only saves the activations
    # for sqrt-n layers?
    #might also be affected by some quantization options?
    # mem usage for activations from full fine-tuning with no gradient checkpointing is pretty well described by this (assumes layer norm and multi-head attention)
    # https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff
    # but with LoRA and gradient checkpointing complicating the picture (also, gemma uses RMSNorm rather than LayerNorm, which I think might halve the activation memory usage from those norm blocks),
    # I'm going to need to do tons of experiments
    #  e.g. with gradient checkpointing, does transformers save every single sub-element of the activations of the selected layers (sqrt(L) layers chosen this way automatically), or just the final output from each layer? or something else?
    return -1.0

def predict_frozen_weights_mem():
    #depends on model size, choices about quantization of weights (including whether to double/nested quantize)
    # need to keep track of both embedding and non-embedding weights!
    return -1.0

def predict_trainable_weights_mem(num_layers: int, num_q_heads: int, num_kv_heads: int, model_dim: int, qkv_dim: int, mlp_dim: int, num_classes: int, lora_r: int, lora_attn: bool, lora_mlp: bool):
    # depends on which parts of the model you're applying lora to,
    # and choices about quantization of weights
    #todo double check whether this is still true for lora weights when doing quantization of the rest
    # particularly concerned b/c of the bitsandbytes setting about compute dtype being float16
    bytes_per_weight = 4#or 2??

    #todo use calc_num_trainable_weights()
    # don't inline it b/c that can be used again by predict_optimizer_states_mem


    return -1.0

def predict_optimizer_states_mem():
    #depends on optimizer choice, # trainable parameters, and choices about quantization of optimizer states

    #one article seemed to claim that a second copy of the current parameter value was part of the optimizer states for each parameter
    # need to try to confirm this empirically
    return -1.0

def predict_gradients_mem():
    #depends on # trainable parameters, batch size, and whether/how-much gradients are quantized (not recommended)
    #ALSO SEQUENCE LENGTH
    # might want to support gradient accumulation (need to check- it might have exactly zero effect on peak vram usage)
    return -1.0


#todo consider later whether it makes sense to break these up further based on different architecture components
# like attention, MLP, embedding,
# If so, how to do that elegantly/DRY-ly?





