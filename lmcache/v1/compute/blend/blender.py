# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        # Global selection across layers within a single blend() call
        self.selected_indices: Optional[torch.Tensor] = None
        self._last_layer_id: Optional[int] = None

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        # Reset per-layer metadata only when layer changes to avoid
        # losing indices within the same layer's multi-pass execution.
        if getattr(self, "_last_layer_id", None) != layer_id:
            self._last_layer_id = layer_id
            self.metadata.imp_indices = None
            self.metadata.positions = None
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if (
            self.selected_indices is None
            and layer_id in self.common_metadata.check_layers
            and self.metadata.imp_indices is None
        ):
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            total_len = diff_k.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])

            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")
            logger.debug(
                f"************************************************************ doing blending"
            )
            print(
                f"************************************************************ doing blending"
            )

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)
            # Persist selection for later layers within the same blend
            self.selected_indices = top_indices
        else:
            # Use existing selection to guide later layers without re-selecting
            if self.selected_indices is not None:
                self.metadata.imp_indices = self.selected_indices

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # def process_qkv_headaware(
    #     self,
    #     q: torch.Tensor,
    #     k: torch.Tensor,
    #     v: torch.Tensor,
    #     residual: torch.Tensor,
    #     layer_id: int,
    #     attn_output: Optional[torch.Tensor],
    #     attn_metadata: LMCAttnMetadata,
    #     *,
    #     agg: str = "max",
    # ):
    #     """
    #     Parallel interface to `process_qkv` that selects tokens to recompute
    #     using per-head K discrepancies aggregated into a per-token score.

    #     This does not change downstream contracts: we still recompute all heads
    #     for the selected tokens and write back their K/V rows.

    #     Args:
    #         agg: Aggregation across heads, one of {"max", "mean"}. Default "max".
    #     """
    #     logger.debug(f"Blender (head-aware) processing KV for layer {layer_id}")
    #     if getattr(self, "_last_layer_id", None) != layer_id:
    #         self._last_layer_id = layer_id
    #         self.metadata.imp_indices = None
    #         self.metadata.positions = None
    #     old_k, old_v = self.gpu_connector.get_kv(layer_id)

    #     if attn_output is None:
    #         attn_output = torch.empty(
    #             q.shape,
    #             dtype=q.dtype,
    #             device=q.device,
    #         )

    #     # perform positional encoding
    #     if self.metadata.positions is None:
    #         self.metadata.positions = torch.arange(
    #             q.shape[0], device=q.device, dtype=torch.int64
    #         )
    #     layer = self.layerwise_model.vllm_model.model.layers[layer_id]
    #     attn_layer = layer.self_attn
    #     q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

    #     if self.selected_indices is None and layer_id in self.common_metadata.check_layers and self.metadata.imp_indices is None:
    #         diff_k = self._compute_head_aware_token_scores(k, old_k, layer_id, agg)
    #         total_len = diff_k.shape[0]

    #         assert self.common_metadata.recomp_ratios is not None
    #         topk_num = int(total_len * self.common_metadata.recomp_ratios[0])

    #         top_indices = torch.topk(diff_k, k=topk_num).indices
    #         top_indices, _ = torch.sort(top_indices)

    #         k, v = k[top_indices], v[top_indices]
    #         q = q[top_indices]
    #         residual = residual[top_indices]

    #         logger.debug(f"(Head-aware) number of indices picked: {len(top_indices)}")

    #         self.metadata.imp_indices = top_indices
    #         self.metadata.positions = self.metadata.positions[top_indices]
    #         attn_output = attn_output[:topk_num]

    #         attn_metadata.update_from_top_indices(top_indices)
    #         self.selected_indices = top_indices
    #     else:
    #         if self.selected_indices is not None:
    #             self.metadata.imp_indices = self.selected_indices

    #     if self.metadata.imp_indices is not None:
    #         old_k[self.metadata.imp_indices] = k
    #         old_v[self.metadata.imp_indices] = v
    #         return q, old_k, old_v, residual, attn_output, attn_metadata
    #     else:
    #         return q, k, v, residual, attn_output, attn_metadata

    def _compute_head_aware_token_scores(
        self,
        k_new: torch.Tensor,
        k_old: torch.Tensor,
        layer_id: int,
        agg: str = "max",
    ) -> torch.Tensor:
        """Compute per-token discrepancy by comparing per-head K vectors and
        aggregating across heads.

        Returns:
            torch.Tensor of shape [T], where T is the token length.
        """
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn = layer.self_attn.attn
        num_kv_heads = attn.num_kv_heads

        head_size = attn.head_size

        T = k_new.shape[0]
        # Reshape to [T, H_kv, D]
        k_new_hd = k_new.to(torch.float32).view(T, num_kv_heads, head_size)
        k_old_hd = k_old.to(torch.float32).view(T, num_kv_heads, head_size)
        # Per-head L2 distance over head-dim -> [T, H_kv]
        diff_hd = torch.sum((k_new_hd - k_old_hd) ** 2, dim=-1)

        if agg == "mean":
            return torch.mean(diff_hd, dim=1)
        # default to max
        return torch.max(diff_hd, dim=1).values

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        # Record stats for blending mode
        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # Start stats tracking
        monitor_req_id = self.cache_engine.stats_monitor.on_retrieve_request(
            num_required_tokens
        )

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield

        # Consume the final yield (ret_mask) to get the actual retrieved tokens
        ret_mask = next(layerwise_retriever)

        # Record the retrieved tokens for stats
        if ret_mask is not None:
            retrieved_tokens = torch.sum(ret_mask).item()
        else:
            # If no mask returned, assume all tokens were retrieved
            retrieved_tokens = num_required_tokens

        self.cache_engine.stats_monitor.on_retrieve_finished(
            monitor_req_id, retrieved_tokens
        )

        # Write stats to file for easy access
        import os

        stats_file = os.path.join("/tmp", "lmcache_blending_stats.txt")
        with open(stats_file, "a") as f:
            f.write(f"requested={num_required_tokens},retrieved={retrieved_tokens}\n")

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        # Reset per-blend-call state
        self.selected_indices = None
        self._last_layer_id = None

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        # blend_layer yields: 1 initial + num_layers + 1 final = num_layers + 2
        for i in range(self.num_layers + 2):
            next(layerwise_blender)
