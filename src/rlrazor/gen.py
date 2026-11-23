import gc
from typing import Literal

# vLLM's share of GPU memory when it runs in-process (GSM8K eval, KL samples, RFT/rewrite sets). A 0.5B model
# needs ~1 GB of weights; the rest of this share is KV cache. The remainder stays free for what runs next in the
# same process: the HF fp32 KL scorer or an SFT trainer (ruling R21).
GPU_MEMORY_UTILIZATION = 0.4


def _shutdown_vllm(llm) -> None:
    """Free a vLLM engine's GPU memory so the same process can load another model afterwards."""
    import torch

    engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()  # v1: stops the EngineCore process that holds the weights and KV cache
    del llm
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:  # best effort: nothing to tear down when the engine ran out-of-process
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    model_path: str,
    prompts: list[list[dict]],
    n: int,
    temperature: float,
    max_tokens: int = 512,
    seed: int = 0,
    backend: Literal["vllm", "hf"] = "vllm",
) -> list[list[str]]:
    if temperature <= 0 and n > 1:
        raise ValueError("greedy decoding with n>1 is degenerate")
    if backend == "vllm":
        from vllm import LLM, SamplingParams  # lazy: not installed locally

        llm = LLM(model=model_path, seed=seed, dtype="bfloat16",
                  gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
        sp = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        try:
            results = llm.chat(prompts, sp)
            return [[o.text for o in r.outputs] for r in results]
        finally:
            _shutdown_vllm(llm)
    if backend != "hf":
        raise ValueError(f"unknown backend {backend!r}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).eval()
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    torch.manual_seed(seed)
    sampling = (
        dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0, repetition_penalty=1.0)
        if temperature > 0
        else dict(do_sample=False)
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            **sampling,
            num_return_sequences=n,
            max_new_tokens=max_tokens,
            pad_token_id=tok.pad_token_id,
        )
    new = out[:, enc["input_ids"].shape[1] :]
    dec = tok.batch_decode(new, skip_special_tokens=True)
    return [dec[i * n : (i + 1) * n] for i in range(len(prompts))]


class Sampler:
    """Repeated sampling from one model with one engine (the teacher's adaptive rounds).

    `sampler(prompts, n, seed)` -> n completions per chat prompt at the sampler's temperature / max_tokens.
    Use as a context manager: the vLLM engine is started once and shut down on exit.
    """

    def __init__(self, model_path: str, temperature: float, max_tokens: int = 512,
                 backend: Literal["vllm", "hf"] = "vllm", gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
                 max_model_len: int | None = None, engine_seed: int = 0):
        self.model_path, self.temperature, self.max_tokens, self.backend = model_path, temperature, max_tokens, backend
        self.gpu_memory_utilization, self.max_model_len, self.engine_seed = (gpu_memory_utilization, max_model_len,
                                                                             engine_seed)
        self.llm = None

    def __enter__(self):
        if self.backend == "vllm":
            from vllm import LLM  # lazy: not installed locally

            kw = {} if self.max_model_len is None else {"max_model_len": self.max_model_len}
            self.llm = LLM(model=self.model_path, seed=self.engine_seed, dtype="bfloat16",
                           gpu_memory_utilization=self.gpu_memory_utilization, **kw)
        elif self.backend != "hf":
            raise ValueError(f"unknown backend {self.backend!r}")
        return self

    def __exit__(self, *exc):
        if self.llm is not None:
            _shutdown_vllm(self.llm)
            self.llm = None
        return False

    def __call__(self, prompts: list[list[dict]], n: int, seed: int) -> list[list[str]]:
        if not prompts:
            return []
        if self.backend == "hf":
            return generate(self.model_path, prompts, n=n, temperature=self.temperature,
                            max_tokens=self.max_tokens, seed=seed, backend="hf")
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=self.temperature, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                            max_tokens=self.max_tokens, seed=seed)
        return [[o.text for o in r.outputs] for r in self.llm.chat(prompts, sp)]
