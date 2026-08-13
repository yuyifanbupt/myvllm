from dataclasses import dataclass, field
from typing import Any


@dataclass
class Qwen3_5MoeVisionConfig:
    base_config_key: str = "vision_config"
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    depth: int = 27
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    in_channels: int = 3
    initializer_range: float = 0.02
    intermediate_size: int = 4304
    model_type: str = "qwen3_5_moe"
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2048
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class Qwen3_5MoeTextConfig:
    base_config_key = "text_config"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    bos_token_id: int = 248044
    dtype: str = "bfloat16"
    eos_token_id: int = 248044
    full_attention_interval: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    layer_types: list[str] = field(
        default_factory=lambda: [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    )
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    mamba_ssm_dtype: str = "float32"
    max_position_embeddings: int = 262144
    model_type: str = "qwen3_5_moe_text"
    moe_intermediate_size: int = 512
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    num_attention_heads: int = 16
    num_experts: int = 256
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 40
    num_key_value_heads: int = 2
    output_router_logits: bool = False
    pad_token_id = None
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-06
    rope_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        }
    )
    router_aux_loss_coef: float = 0.001
    shared_expert_intermediate_size: int = 512
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 248320


@dataclass
class Qwen3_5MoeConfig:
    model_type: str = "qwen3_5_moe"
    vision_config: Qwen3_5MoeVisionConfig = field(default_factory=Qwen3_5MoeVisionConfig)
    text_config: Qwen3_5MoeTextConfig = field(default_factory=Qwen3_5MoeTextConfig)
    architectures: list[str] = field(default_factory=lambda: ["Qwen3_5MoeForConditionalGeneration"])
    image_token_id: int = 248056
    tie_word_embeddings: bool = False
    transformers_version: str = "4.57.1"
    video_token_id: int = 248057
    vision_end_token_id: int = 248054
    vision_start_token_id: int = 248053

    @classmethod
    def from_dict(cls, dic: dict[str, Any]) -> "Qwen3_5MoeConfig":
        if "vision_config" in dic:
            vision_config = Qwen3_5MoeVisionConfig(**dic["vision_config"])
            dic["vision_config"] = vision_config

        if "text_config" in dic:
            text_config = Qwen3_5MoeTextConfig(**dic["text_config"])
            dic["text_config"] = text_config

        return Qwen3_5MoeConfig(**dic)
