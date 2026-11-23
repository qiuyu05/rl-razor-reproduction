import torch


def interpolate_state_dicts(sd0: dict, sd1: dict, alpha: float) -> dict:
    """(1-alpha)*sd0 + alpha*sd1 for float tensors; non-float copied from sd1."""
    if set(sd0) != set(sd1):
        raise ValueError("state dict key sets differ")
    out = {}
    for k, v1 in sd1.items():
        v0 = sd0[k]
        if torch.is_floating_point(v1):
            out[k] = (1 - alpha) * v0 + alpha * v1
        else:
            out[k] = v1.clone()
    return out


def save_interpolated(base_path, ft_path, alpha: float, out_dir) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    ft = AutoModelForCausalLM.from_pretrained(ft_path, torch_dtype=torch.bfloat16)
    sd0 = {k: (v.float() if v.is_floating_point() else v) for k, v in base.state_dict().items()}
    sd1 = {k: (v.float() if v.is_floating_point() else v) for k, v in ft.state_dict().items()}
    merged = interpolate_state_dicts(sd0, sd1, alpha)
    merged = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in merged.items()}
    ft.load_state_dict(merged)
    ft.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(ft_path).save_pretrained(out_dir)
    return str(out_dir)
