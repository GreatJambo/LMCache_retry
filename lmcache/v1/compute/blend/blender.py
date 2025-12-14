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
        # losing indices within the same layer's multi-pass execution (e.g. multi-step attention?).
        # HOWEVER, for chunked execution of the *same request* (L0 C1 -> L0 C2), we MUST reset imp_indices.

        # We need a robust way to know if we are starting a NEW chunk for this layer.
        # We can leverage the offset tracking we just added.
        # If current_offset changes, it implies we moved forward.

        # Let's rely on standard logic but FORCE RESET imp_indices if we are in chunked mode (mask > k)
        # AND we are in the check_layers.

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

        # Apply token mask to align newly computed K/V with cached K/V (old_k/old_v)
        # old_k/old_v typically only contains the "True" part of the mask
        # This should only happen for the first layer (check layer) where blending happens.
        # Subsequent layers receive sparse K/V inputs.
        if (
            self.metadata.token_mask is not None
            and layer_id in self.common_metadata.check_layers
        ):
            mask = self.metadata.token_mask.to(k.device)
            # print(f"DEBUG: process_qkv layer={layer_id} k.shape={k.shape} mask.shape={mask.shape} old_k.shape={old_k.shape}")

            # Handle chunked execution (e.g., vLLM processes 48 tokens, but mask is 320)
            num_tokens = k.shape[0]
            if num_tokens < mask.shape[0]:
                # We need to determine WHICH chunk of the mask/old_k corresponds to these tokens.
                # Heuristic: We track the cumulative number of tokens processed for this blend session.
                # Note: This assumes compute_layer is called sequentially for chunks of the same request.

                # Initialize per-layer offset tracker if not present
                if not hasattr(self.metadata, "_blend_layer_offsets"):
                    # Should be init in blend(), but safe fallback
                    self.metadata._blend_layer_offsets = {}

                # The most robust way is to assume linear processing of the retrieved suffix.
                current_offset = self.metadata._blend_layer_offsets.get(layer_id, 0)

                # If offset exceeds mask size, something is wrong or we wrapped around?
                # Assuming simple sequential processing for now.

                end_offset = min(current_offset + num_tokens, mask.shape[0])
                actual_tokens = end_offset - current_offset

                # Slice mask and old_k
                mask_chunk = mask[current_offset:end_offset]
                old_k = old_k[current_offset:end_offset]
                old_v = old_v[current_offset:end_offset]

                # Update offset for next call to this layer
                self.metadata._blend_layer_offsets[layer_id] = end_offset

                # If we updated k above, use the slice.
                # Wait, k is the input (48). We need to match it with mask_chunk (48).
                # But if actual_tokens < num_tokens (e.g. last jagged chunk), slice k too?
                if actual_tokens < num_tokens:
                    # This implies input k has extra tokens not covered by our retrieval mask?
                    # Or maybe we are blending only a part of the prompt.
                    # For safety, let's slice k to match the mask chunk we have.
                    k = k[:actual_tokens]
                    v = v[:actual_tokens]
                    q = q[:actual_tokens]
                    residual = residual[:actual_tokens]

                mask = mask_chunk

                # CRITICAL FIX for Chunked Mode:
                # If we are in Check Layer, we MUST re-select indices for THIS chunk.
                # The previous logic (lines 160+) would skip selection if imp_indices was not None.
                # But imp_indices serves two purposes:
                # 1. Reuse selection from L0 -> L1 (Good)
                # 2. Reuse selection from L0_Chunk1 -> L0_Chunk2 (BAD!)

                # How to distinguish?
                # If we are in Check Layer, we ALWAYS want to calculate fresh indices for the current input chunk.
                # We should NOT reuse "imp_indices" from a previous chunk of the same layer.
                # But we SHOULD reuse "imp_indices" if we are in L1 (from L0).

                if layer_id in self.common_metadata.check_layers:
                    self.metadata.imp_indices = None
                    # We also need to reset positions because they are derived from imp_indices
                    # Actually positions logic is below.
                    # If we set imp_indices to None, it will enter the selection block.

            k_subset = k[mask]
            v_subset = v[mask]
        else:
            mask = None
            k_subset = k
            v_subset = v

        if (
            self.selected_indices is None
            and layer_id in self.common_metadata.check_layers
            and self.metadata.imp_indices is None
            and self.metadata.token_mask is not None
        ):
            # Calculate difference
            # Slicing: Ensure we compare the overlapping part only.
            # Usually input (k_subset) >= cache (old_k) in prefix caching.
            min_len = min(k_subset.shape[0], old_k.shape[0])

            k_diff_input = k_subset[:min_len].to(torch.float32)
            old_k_input = old_k[:min_len].to(torch.float32)

            diff_k = (k_diff_input - old_k_input) ** 2
            diff_norm = diff_k.sum(dim=-1)

            # Select top k
            assert self.common_metadata.recomp_ratios is not None
            # TODO(Jiayi): remove `[0]` hardcode
            # Calculate topk based on the current chunk size
            num_candidates = diff_norm.shape[0]
            topk_num = int(num_candidates * self.common_metadata.recomp_ratios[0])
            # Ensure at least some tokens are picked if ratio > 0, or handle 0
            if (
                topk_num == 0
                and self.common_metadata.recomp_ratios[0] > 0
                and num_candidates > 0
            ):
                topk_num = 1

            # print(f"DEBUG: process_qkv layer={layer_id} num_candidates={num_candidates} topk_num={topk_num}")

            if num_candidates == 0:
                top_indices = torch.empty(0, dtype=torch.int64, device=diff_norm.device)
            elif topk_num >= num_candidates:
                # Take all
                top_indices = torch.arange(num_candidates, device=diff_norm.device)
            else:
                top_indices = torch.topk(
                    diff_norm, k=topk_num, dim=0, largest=False
                ).indices

            # Map valid subset indices back to full tensor indices if mask exists
            if mask is not None:
                mask_indices = torch.nonzero(mask, as_tuple=True)[0]
                real_indices = mask_indices[top_indices]
            else:
                real_indices = top_indices

            # Update cache with new values (using subset mapping)
            old_k[top_indices] = k_subset[top_indices]
            old_v[top_indices] = v_subset[top_indices]

            # Filter Q and Residual for sparse execution
            q = q[real_indices]
            residual = residual[real_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            # Store selection for this blend step
            # imp_indices are relative to the cached buffer (old_k) size
            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[real_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)
            # Persist selection for later layers within the same blend
            self.selected_indices = top_indices

            # If mask was used, we also persist the real indices mapping if needed?
            # Actually selected_indices (top_indices) is sufficient for old_k updates in later layers.
            # But q/residual in later layers are already sparse, so we don't need real_indices again
            # (they flow through the network).

        else:
            # Use existing selection to guide later layers without re-selecting
            if self.selected_indices is not None:
                self.metadata.imp_indices = self.selected_indices

        if self.metadata.imp_indices is not None:
            # For subsequent layers, k is passed as sparse (size = topk_num).
            # We need to update the dense cache (old_k) with these sparse values.
            # In the first layer, k is dense, and we've already updated the cache
            # inside the selection block above, so we skip here.
            if k.shape[0] == self.metadata.imp_indices.shape[0]:
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

        stats_file = os.environ.get(
            "LMCACHE_STATS_FILE", os.path.join("/tmp", "lmcache_blending_stats.txt")
        )
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

        if mask is not None:
            mask = mask.to(tokens.device)

        # Reset per-blend-call state
        self.selected_indices = None
        self._last_layer_id = None

        # Reset and track slice offsets in metadata for robustness across compute_layer calls
        self.metadata.token_mask = mask  # Store mask for process_qkv
        if mask is not None:
            self.metadata._blend_layer_offsets = {}  # New tracking dictionary

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        # blend_layer yields: 1 initial + num_layers + 1 final = num_layers + 2
        for i in range(self.num_layers + 2):
            next(layerwise_blender)
