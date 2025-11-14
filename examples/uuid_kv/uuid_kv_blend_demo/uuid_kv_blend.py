"""
SPDX-License-Identifier: Apache-2.0

UUID-based multi-KV blending demo (vLLM-integrated).

This demo:
- Precomputes KV for multiple chunks via vLLM and registers each chunk to a UUID
- Optionally verifies retrieve_by_uuid using a standalone LMCache v1 engine
- Runs a blending scenario by reordering chunks (CacheBlend), using a fixed separator

Notes:
- This version forces static connector "LMCacheConnectorV1" to match a fixed vLLM build.
- It follows a minimal style inspired by cacheblend_min.py while preserving UUID features.
"""

from dataclasses import asdict
import argparse
import contextlib
import os
import time
import uuid as uuidlib

from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--use-disk", action="store_true", help="Use disk backend for LMCache")
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help="vLLM tokenizer mode (e.g., 'auto' or model-specific)",
    )
    p.add_argument("--enable-sparse", action="store_true")
    p.add_argument("--enable-async-loading", action="store_true")

    # Minimal-style controls
    p.add_argument("--max-model-len", type=int, default=18000)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--blend-sep", type=str, default=" # # ", help="Separator used by CacheBlend")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for both LMCache and vLLM",
    )

    p.add_argument(
        "--blend-recompute-ratios",
        type=str,
        default=None,
        help="Comma-separated recompute ratios for blending (e.g., '0.05' or '0.10,0.05')",
    )
    p.add_argument(
        "--blend-check-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to compute blending checks (e.g., '0,1')",
    )

    p.add_argument(
        "--chunks-json",
        type=str,
        default="examples/uuid_kv/uuid_kv_blend_demo/chunks.json",
        help="Path to JSON file containing a 'chunks' dict (e.g., {'chunk1': '...', 'chunk2': '...', 'chunk3': '...'})",
    )
    p.add_argument(
        "--sleep-after-first-secs",
        type=float,
        default=1.0,
        help="Seconds to sleep after precomputing/storing to allow KV persistence",
    )
    p.add_argument("--verify-uuid", action="store_true", help="Verify retrieve_by_uuid via dummy engine")
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="vLLM scheduler max_num_batched_tokens to avoid chunked prefill",
    )

    return p.parse_args()


def setup_environment_variables(
    use_disk: bool,
    enable_sparse: bool,
    enable_async_loading: bool,
    blend_recompute_ratios: str | None,
    blend_check_layers: str | None,
    blend_sep_text: str | None = None,
    log_level: str = "INFO",
):
    # Core LMCache knobs
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = "True" if enable_async_loading else "False"

    # Optional blending parameters
    if blend_check_layers is not None:
        os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = blend_check_layers
    if blend_recompute_ratios is not None:
        os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = blend_recompute_ratios

    # Always set a special separator when provided (minimal style)
    if blend_sep_text is not None:
        os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_sep_text

    # Sparse attention optional knobs
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    # Logging levels
    os.environ["LMCACHE_LOG_LEVEL"] = log_level
    os.environ["VLLM_LOGGING_LEVEL"] = log_level


@contextlib.contextmanager
def build_llm_with_lmcache(
    model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int | None = None,
    tokenizer_mode: str | None = None,
):
    # Force static connector name to match fixed vLLM build
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
        tokenizer_mode=tokenizer_mode or "auto",
        max_num_batched_tokens=max_num_batched_tokens,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend created by vLLM integration
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt_tokens: List[int],
    sampling_params: SamplingParams,
    tag: str,
):
    start = time.time()
    outputs = llm.generate(prompts={"prompt_token_ids": prompt_tokens}, sampling_params=sampling_params)
    duration = time.time() - start
    print("-" * 50)
    for output in outputs:
        text = output.outputs[0].text
        print(f"[{tag}] generated: {text!r}")
    print(f"[{tag}] took {duration:.2f}s")
    print("-" * 50)
    return duration


def prepare_tokens(tokenizer, chunks_json_path: str, sep_text: str | None) -> Dict[str, List[int]]:
    import json
    from pathlib import Path

    sys_prompt = tokenizer.encode("You are a very helpful assistant.")

    # Load chunk texts from JSON: {"chunks": {"chunk1": "...", "chunk2": "...", "chunk3": "..."}}
    json_path = Path(chunks_json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", {})
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError("Invalid chunks JSON: expect {'chunks': {'chunk1': '...', ...}}")

    def pick(name: str, default: str = "") -> str:
        return str(chunks.get(name, default))

    # Encode each chunk (drop BOS)
    chunk1 = tokenizer.encode(pick("chunk1", ""))[1:]
    chunk2 = tokenizer.encode(pick("chunk2", ""))[1:]
    chunk3 = tokenizer.encode(pick("chunk3", ""))[1:]

    # Warmup: unrelated small text
    warmup = tokenizer.encode("This is a warmup prompt to initialize the cache system.")[1:]

    # Short suffix/questions
    q1 = tokenizer.encode("Hello, my name is")[1:]
    q2 = tokenizer.encode("Hello, how are you?")[1:]
    q3 = tokenizer.encode("Hello, what's up?")[1:]

    sep_tokens = tokenizer.encode(sep_text)[1:] if sep_text is not None else []

    return {
        "sys": sys_prompt,
        "c1": chunk1,
        "c2": chunk2,
        "c3": chunk3,
        "warmup": warmup,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "sep": sep_tokens,
    }


def register_uuids_and_store(
    llm: LLM,
    uuids: List[str],
    chunk_tokens_list: List[List[int]],
    sleep_secs: float,
) -> bool:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)

    # 1) Precompute & store KV for each chunk by running a tiny generate
    for i, chunk in enumerate(chunk_tokens_list):
        tag = f"precompute-c{i+1}"
        print_output(llm, chunk, sampling_params, tag)

    # 2) Register each chunk as a UUID mapping using the SAME engine behind vLLM
    engine = LMCacheEngineBuilder.get(ENGINE_NAME)
    if engine is not None:
        for uid, chunk in zip(uuids, chunk_tokens_list, strict=False):
            total = engine.register_uuid(uid, tokens=chunk)
            print(f"Registered UUID={uid}, total_tokens={total}")
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        return True

    # Register in worker via internal API server (/run_script)
    def _post_run_script(port: int, uid: str, tokens: list[int]) -> str:
        script = (
            "adapter = app.state.lmcache_adapter\n"
            "engine = adapter.lmcache_engine\n"
            f"uid = {uid!r}\n"
            f"tokens = {tokens!r}\n"
            "result = engine.register_uuid(uid, tokens=tokens)\n"
        )
        boundary = "----lmcacheformboundary"
        body = (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"script\"; filename=\"s.py\"\r\n"
            "Content-Type: text/plain\r\n\r\n"
            f"{script}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        conn.request("POST", "/run_script", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", errors="replace")
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"run_script failed: {resp.status} {data}")
        return data

    # Fallback only if internal API server is explicitly enabled
    enabled = os.getenv("LMCACHE_INTERNAL_API_SERVER_ENABLED", "").lower() in ("true", "1", "yes", "on")
    if not enabled:
        print(
            "[uuid-register] LMCache engine not found and internal API server not enabled; "
            "skip UUID registration. Set LMCACHE_INTERNAL_API_SERVER_ENABLED=true to enable fallback."
        )
        return False

    port_start = int(os.getenv("LMCACHE_INTERNAL_API_SERVER_PORT_START", "6999"))
    worker_port = port_start + 1  # worker 0

    try:
        for uid, chunk in zip(uuids, chunk_tokens_list, strict=False):
            resp = _post_run_script(worker_port, uid, chunk)
            print(f"[worker] Registered UUID={uid}, resp={resp!r}")
    except Exception as e:
        print(
            f"[uuid-register] Internal API fallback failed: {e}. "
            "Ensure internal API server is running (set LMCACHE_INTERNAL_API_SERVER_ENABLED=true)."
        )
        return False

    # 3) Wait for persistence if needed
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return True


class DummyGPUConnector(torch.nn.Module):
    """Minimal connector that is NO-OP for batched_to_gpu, suitable for UUID verification."""

    def __init__(self, hidden_dim_size: int, num_layers: int):
        super().__init__()
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    # Single transfers unused here
    def to_gpu(self, memory_obj, start: int, end: int, **kwargs):
        return None

    def from_gpu(self, memory_obj, start: int, end: int, **kwargs):
        return None

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        return None

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        # Intentionally a NO-OP for verification
        return None

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])


def optional_verify_by_uuid_with_dummy(
    use_disk: bool,
    model: str,
    uuids: List[str],
):
    if not use_disk:
        print("[verify] Skipped: verification requires disk backend to share KV across engines.")
        return

    # Build a minimal LMCache v1 engine matching metadata used by vLLM engine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.config import LMCacheEngineMetadata

    cfg = LMCacheEngineConfig.from_defaults()
    cfg.chunk_size = int(os.getenv("LMCACHE_CHUNK_SIZE", "256"))
    cfg.local_cpu = True
    cfg.max_local_cpu_size = 1.0
    cfg.local_disk = os.getenv("LMCACHE_LOCAL_DISK", "file://local_disk/")
    cfg.max_local_disk_size = 10.0
    cfg.pre_caching_hash_algorithm = "sha256"
    cfg.use_layerwise = False

    # Metadata must match resolve() stored meta: fmt/model/world_size/worker_id
    kv_dtype = torch.float16
    kv_shape = (2, 2, cfg.chunk_size, 1, 64)  # Dummy; not used by NO-OP connector
    meta = LMCacheEngineMetadata(
        model_name=model,
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
    )

    connector = DummyGPUConnector(hidden_dim_size=64, num_layers=2)

    def _bcast_fn(t: torch.Tensor, dst: int):
        return None

    def _bcast_obj_fn(obj, dst: int):
        return obj

    instance_id = "uuid_kv_blend_verify"
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        cfg,
        meta,
        connector,
        _bcast_fn,
        _bcast_obj_fn,
    )
    engine.post_init()

    try:
        for uid in uuids:
            print(f"[verify] retrieving by UUID={uid} ...")
            mask = engine.retrieve_by_uuid(uid)
            got = int(torch.sum(mask).item())
            print(f"[verify] UUID={uid}, retrieved tokens={got}")
    finally:
        engine.close()


def build_uuid_blend_prompts(tokens: Dict[str, List[int]], use_sep_in_final: bool) -> Tuple[List[int], List[int], List[int]]:
    sys_ = tokens["sys"]
    c1, c2, c3 = tokens["c1"], tokens["c2"], tokens["c3"]
    q1, q2, q3 = tokens["q1"], tokens["q2"], tokens["q3"]
    sep = tokens["sep"] if use_sep_in_final else []

    first = sys_ + (sep + c1 if sep else c1) + (sep + c2 if sep else c2) + (sep + c3 if sep else c3) + q1
    second = sys_ + (sep + c2 if sep else c2) + (sep + c1 if sep else c1) + (sep + c3 if sep else c3) + q2
    third = sys_ + (sep + c3 if sep else c3) + (sep + c1 if sep else c1) + (sep + c2 if sep else c2) + q3
    return first, second, third


def run_uuid_blend_scenario(
    llm: LLM,
    warmup_tokens: List[int],
    prompts: Tuple[List[int], List[int], List[int]],
    sleep_after_first_secs: float,
):
    sp = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)
    summary: Dict[str, float] = {}

    d = print_output(llm, warmup_tokens, sp, "warmup")
    summary["warmup"] = d

    first, second, third = prompts
    summary["first"] = print_output(llm, first, sp, "blend-first")
    if sleep_after_first_secs > 0:
        time.sleep(sleep_after_first_secs)
    summary["second"] = print_output(llm, second, sp, "blend-second")
    if sleep_after_first_secs > 0:
        time.sleep(sleep_after_first_secs)
    summary["third"] = print_output(llm, third, sp, "blend-third")

    print("==== Summary (seconds) ====")
    for k, v in summary.items():
        print(k, v)


def main():
    args = parse_args()

    setup_environment_variables(
        use_disk=args.use_disk,
        enable_sparse=args.enable_sparse,
        enable_async_loading=args.enable_async_loading,
        blend_recompute_ratios=args.blend_recompute_ratios,
        blend_check_layers=args.blend_check_layers,
        blend_sep_text=args.blend_sep,
        log_level=args.log_level,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Propagate model name for offline UUID registration metadata
    os.environ.setdefault("LMCACHE_MODEL_NAME", args.model)

    sep_text = args.blend_sep
    toks = prepare_tokens(tokenizer, args.chunks_json, sep_text)

    with build_llm_with_lmcache(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tokenizer_mode=args.tokenizer_mode,
    ) as llm:
        # Prepare UUIDs for three chunks
        uuids = [str(uuidlib.uuid4()) for _ in range(3)]

        # Precompute and register UUIDs
        ok = register_uuids_and_store(
            llm, uuids, [toks["c1"], toks["c2"], toks["c3"]], args.sleep_after_first_secs
        )

        # Optional verification (disk-backed only) — only when registration succeeded
        if ok and args.verify_uuid:
            optional_verify_by_uuid_with_dummy(args.use_disk, args.model, uuids)
        elif args.verify_uuid and not ok:
            print("[verify] Skipped: UUID registration failed or was skipped.")

        prompts = build_uuid_blend_prompts(toks, True)
        run_uuid_blend_scenario(llm, toks["warmup"], prompts, args.sleep_after_first_secs)


if __name__ == "__main__":
    main()


