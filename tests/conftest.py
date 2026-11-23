import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    path = tmp_path_factory.mktemp("tiny_model")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)
