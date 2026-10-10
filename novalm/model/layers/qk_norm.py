"""Per-head query/key RMS normalization used before RoPE and attention."""
from novalm.model.layers.rmsnorm import RMSNorm


class QKNorm(RMSNorm):
    """RMSNorm specialized by API for query and key vectors on the head axis."""

