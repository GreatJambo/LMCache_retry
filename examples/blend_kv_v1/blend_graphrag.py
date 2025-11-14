"""
SPDX-License-Identifier: Apache-2.0

LMCache blending test using a GraphRAG-style prompt file.

Reads a chat-style prompt (system + user), splits the system section
into three large segments (approx. Entities / Relationships / Trailing
instructions) and issues three requests with different segment orders,
separated by the blending separator. This mirrors examples/blend.py
behavior but sources segments from a real GraphRAG prompt.
"""

from dataclasses import asdict
import argparse
import contextlib
import os
import time

from pathlib import Path
from typing import List, Tuple, Dict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
):
    # LMCache-related environment variables
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Enable blending + segment separator
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_LOG_LEVEL"] = "DEBUG"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    os.environ["LMCACHE_DEBUG_SEGMENTS"] = "1"

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


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        # Let vLLM derive max_model_len from the model config to
        # avoid overriding models with smaller context (e.g., Llama3 8B = 8192)
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def _extract_system_and_user(raw: str) -> Tuple[str, str]:
    """Extract system and user text chunks from a chat-formatted GraphRAG prompt."""
    sys_hdr = "<|start_header_id|>system<|end_header_id|>"
    eot = "<|eot_id|>"
    user_hdr = "<|start_header_id|>user<|end_header_id|>"

    # System section
    sys_start = raw.find(sys_hdr)
    if sys_start >= 0:
        sys_start += len(sys_hdr)
    else:
        sys_start = 0
    eot_pos = raw.find(eot, sys_start)
    if eot_pos < 0:
        eot_pos = len(raw)
    system_text = raw[sys_start:eot_pos]

    # User section
    uq_start = raw.find(user_hdr, eot_pos)
    question = ""
    if uq_start >= 0:
        uq_start += len(user_hdr)
        uq_eot = raw.find(eot, uq_start)
        if uq_eot < 0:
            uq_eot = len(raw)
        question = raw[uq_start:uq_eot].strip()

    return system_text, question


def _split_system_into_chunks(system_text: str) -> Tuple[str, str, str]:
    """Split the system text into three segments using common GraphRAG markers.

    Fallback to roughly equal thirds if markers are absent.
    """
    ent_mark = "-----Entities-----"
    rel_mark = "-----Relationships-----"
    goal_mark = "---Goal---"

    ent_pos = system_text.find(ent_mark)
    rel_pos = system_text.find(rel_mark)

    # Next goal marker after relationships
    post_start = None
    if rel_pos >= 0:
        post_start = system_text.find(goal_mark, rel_pos + len(rel_mark))
        if post_start < 0:
            post_start = None

    if ent_pos < 0 and rel_pos < 0:
        n = len(system_text)
        a = n // 3
        b = (2 * n) // 3
        return system_text[:a], system_text[a:b], system_text[b:]

    # pre + entities until relationships (if present) or until entities (if only entities present)
    cut_pos = None
    if rel_pos >= 0 and ent_pos >= 0:
        cut_pos = rel_pos
    elif ent_pos >= 0:
        cut_pos = ent_pos
    else:
        cut_pos = len(system_text)
    pre_plus_entities = system_text[:cut_pos]

    relationships = ""
    if rel_pos >= 0:
        relationships = system_text[rel_pos : post_start if post_start is not None else len(system_text)]

    post = system_text[post_start:] if post_start is not None else ""

    if not pre_plus_entities:
        n = len(system_text)
        a = n // 3
        b = (2 * n) // 3
        return system_text[:a], system_text[a:b], system_text[b:]

    return pre_plus_entities, relationships, post


def prepare_tokens_from_template(
    tokenizer: AutoTokenizer,
    template_path: str,
    sep_text: str,
) -> Dict[str, List[int]]:
    raw = Path(template_path).read_text(encoding="utf-8")
    system_text, question_text = _extract_system_and_user(raw)
    c1_text, c2_text, c3_text = _split_system_into_chunks(system_text)

    # Tokenization: keep BOS on sys, drop BOS for appended segments
    sys_tokens = tokenizer.encode("You are a very helpful assistant.")
    sep_tokens = tokenizer.encode(sep_text)[1:]

    def enc(txt: str) -> List[int]:
        return tokenizer.encode(txt)[1:]

    warmup = enc("This is a warmup prompt for GraphRAG blending test.")
    c1 = enc(c1_text)
    c2 = enc(c2_text)
    c3 = enc(c3_text)
    # Keep the same user question across runs; if empty, use a default
    q = enc(question_text if question_text else "Answer the question briefly.")

    return {
        "sys": sys_tokens,
        "sep": sep_tokens,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "q": q,
        "warmup": warmup,
    }


def build_blend_prompts(tokens: Dict[str, List[int]]) -> Tuple[List[int], List[int], List[int]]:
    sys_ = tokens["sys"]
    sep = tokens["sep"]
    c1, c2, c3 = tokens["c1"], tokens["c2"], tokens["c3"]
    q = tokens["q"]

    def join(*parts: List[int]) -> List[int]:
        out: List[int] = []
        for p in parts:
            out.extend(p)
        return out

    first = join(sys_, sep, c1, sep, c2, sep, c3, sep, q)
    second = join(sys_, sep, c2, sep, c1, sep, c3, sep, q)
    third = join(sys_, sep, c3, sep, c1, sep, c2, sep, q)
    return first, second, third


def _count_splits(token_ids: List[int], sep_ids: List[int]) -> int:
    """Count occurrences of sep_ids as contiguous subsequences inside token_ids."""
    import torch
    sep_len = len(sep_ids)
    if sep_len == 0:
        return 0
    t = torch.tensor(token_ids, dtype=torch.long)
    if t.numel() < sep_len:
        return 0
    windows = t.unfold(0, sep_len, 1)
    sep = torch.tensor(sep_ids, dtype=torch.long)
    return int(((windows == sep).all(dim=1)).sum().item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--use-disk", action="store_true")
    parser.add_argument("-b", "--blend-special-str", default=" ### ")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    # parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--enable-sparse", action="store_true")
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="/home/jambo/Learn/LMCache_retry/examples/blend_kv_v1/demo_graohrag/load.txt",
        help="Path to GraphRAG-style prompt file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(args.use_disk, args.blend_special_str, args.enable_sparse)

    tokenizer = AutoTokenizer.from_pretrained(model)
    # Prepare tokens from the GraphRAG-style template
    tokens = prepare_tokens_from_template(
        tokenizer, args.prompt_path, args.blend_special_str
    )
    first_prompt, second_prompt, third_prompt = build_blend_prompts(tokens)

    # Optional diagnostics: print how many explicit separators exist
    # in each source chunk and in the combined prompts. Helps debug
    # segmentation mismatches causing low LMCache hits.
    if os.getenv("LMCACHE_DEBUG_SEGMENTS", "").lower() in ("1", "true", "on", "yes"):
        sep = tokens["sep"]
        print(f"[debug] sep_len={len(sep)}")
        for name in ("c1", "c2", "c3"):
            cnt = _count_splits(tokens[name], sep)
            print(f"[debug] splits in {name}: {cnt}")
        for name, prompt in ("first", first_prompt), ("second", second_prompt), ("third", third_prompt):
            cnt = _count_splits(prompt, sep)
            print(f"[debug] splits in {name} prompt: {cnt}")

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
        print_output(llm, tokens["warmup"], sampling_params, "warmup")
        print_output(llm, first_prompt, sampling_params, "first")
        time.sleep(1)
        print_output(llm, second_prompt, sampling_params, "second (warming up blend code path)")
        time.sleep(1)
        print_output(llm, third_prompt, sampling_params, "third")


if __name__ == "__main__":
    main()
