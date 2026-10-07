import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adapters.deepseek_ocr_external_worker import _patch_llama_attention_compat


def main():
    from transformers import LlamaConfig
    from transformers.models.llama import modeling_llama

    _patch_llama_attention_compat()

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_bias=False,
    )
    config._attn_implementation = "eager"

    attn = modeling_llama.LlamaAttention(config=config, layer_idx=0).eval()
    hidden_states = torch.randn(1, 8, 64)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)

    output = attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )

    assert isinstance(output, tuple)
    assert len(output) == 3
    assert output[0].shape == hidden_states.shape


if __name__ == "__main__":
    main()
