import json

from marshmallow_dataclass import dataclass
from typing import ClassVar, Type, List
from marshmallow import Schema, EXCLUDE
from enum import Enum


@dataclass
class ModelData:
    name: str
    num_embed_params: int
    num_non_embed_params: int
    num_layers: int
    model_dim: int
    feed_forward_hidden_dim: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    vocab_size: int
    embed_module_name: str
    attn_module_names: List[str]
    mlp_module_names: List[str]
    # this currently assumes the gating's elementwise product is done in the hidden dimension, before down projecting
    #  back to the model dimension
    is_mlp_gated: bool
    initial_massive_params_chunk_size: int
    total_size_of_frozen_weight_small_tensors: int
    persistent_massive_params_chunk_size: int
    Schema: ClassVar[Type[Schema]] = Schema


gemma_2b_data: ModelData = ModelData.Schema().load(json.load(open("./model_details/google/gemma_2b.json")), unknown=EXCLUDE)
print(gemma_2b_data)


class QuantizationLevels(Enum):
    FP32 = 32
    TF32 = 32
    FP16 = 16
    BF16 = 16
    INT8 = 8
    FP8 = 8
    NF4 = 4#levels above 4 not currently supported
    FP4 = 4
