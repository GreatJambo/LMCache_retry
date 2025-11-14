# SPDX-License-Identifier: Apache-2.0
# Local
from .flash_attn import LMCFlashAttnBackend


def infer_attn_backend_from_vllm(vllm_attn, enable_sparse: bool = False):
    """
    Select the attention backend based on vLLM's implementation and config.

    This function avoids importing the FlashInfer-based sparse backend unless
    it is actually requested and the runtime indicates we are using the
    FlashInfer attention implementation. This prevents an unnecessary hard
    import-time dependency on the third-party `flashinfer` package when the
    sparse path is not in use.
    """
    attn_name = type(vllm_attn.impl).__name__
    if attn_name == "FlashInferImpl" and enable_sparse:
        try:
            # Delayed import so environments without flashinfer can still run
            # the default (non-sparse) path.
            from .flash_infer_sparse import LMCFlashInferSparseBackend  # noqa: WPS433
        except Exception as e:  # pragma: no cover - defensive guard
            raise ImportError(
                "FlashInfer sparse backend requested but 'flashinfer' is not "
                "installed. Either install flashinfer or disable sparse mode "
                "(do not set enable_sparse)."
            ) from e
        return LMCFlashInferSparseBackend(vllm_attn)
    if attn_name == "FlashAttentionImpl" and not enable_sparse:
        return LMCFlashAttnBackend(vllm_attn)
    raise ValueError(
        f"Attention backend {attn_name} with enable_sparse={enable_sparse} is not supported in LMCache."
    )
