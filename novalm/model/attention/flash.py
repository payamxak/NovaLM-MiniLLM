"""Flash-capable SDPA dispatcher; PyTorch selects a supported kernel or fallback."""
from novalm.model.attention.sdpa import sdpa_causal_attention


def flash_causal_attention(query, key, value, allowed):
    """Use PyTorch SDPA's kernel dispatcher (FlashAttention is hardware dependent)."""
    return sdpa_causal_attention(query, key, value, allowed)
