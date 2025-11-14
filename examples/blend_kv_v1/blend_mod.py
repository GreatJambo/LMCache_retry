# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
    enable_async_loading: bool = True,
    blend_recompute_ratios: str | None = None,
    blend_check_layers: str | None = None,
    enable_blending: bool = True,
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True" if enable_blending else "False"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    if enable_blending:
        # Allow override via args or keep existing env; otherwise use defaults
        os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = (
            blend_check_layers
            if blend_check_layers is not None
            else os.getenv("LMCACHE_BLEND_CHECK_LAYERS", "1")
        )
        os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = (
            blend_recompute_ratios
            if blend_recompute_ratios is not None
            else os.getenv("LMCACHE_BLEND_RECOMPUTE_RATIOS", "0.15")
        )

    os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] = (
        "True" if enable_async_loading else "False"
    )

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(
    lmcache_connector: str,
    model: str,
    max_num_batched_tokens: int | None = None,
    tokenizer_mode: str | None = None,
):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32648,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True,
        tokenizer_mode=tokenizer_mode or "auto",
        max_num_batched_tokens=max_num_batched_tokens,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    duration = time.time() - start
    print(f"Generation took {duration:.2f} seconds, {req_str} request done.")
    print("-" * 50)
    return duration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        help=(
            "vLLM tokenizer mode (e.g., 'auto' (default) or 'mistral' for"
            " Mistral tokenizer when model assets support it)"
        ),
    )

    parser.add_argument(
        "--enable-sparse",
        action="store_true",
    )

    parser.add_argument(
        "--enable-async-loading",
        action="store_true",
        help=(
            "Enable LMCache async lookup/prefetch loading (default: disabled). "
            "Disable to make lookups synchronous when demonstrating cache hits."
        ),
    )

    parser.add_argument(
        "--blend-recompute-ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated list of recompute ratios for blending (e.g., '0.05' or '0.10,0.05'). "
            "If not set, falls back to env LMCACHE_BLEND_RECOMPUTE_RATIOS or default 0.15."
        ),
    )

    parser.add_argument(
        "--blend-check-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated list of layer indices to run blending checks (e.g., '0,1'). "
            "If not set, falls back to env LMCACHE_BLEND_CHECK_LAYERS or default '1'."
        ),
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="blend3",
        choices=["pure", "prefix1", "blend3", "compare"],
        help=(
            "Experiment scenario: 'pure' (no reuse, no save), 'prefix1' (save first segment, "
            "then reuse as prefix without blending), 'blend3' (save 3 segments, reorder to blend), "
            "or 'compare' to run all three on the same data for side-by-side comparison."
        ),
    )

    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "vLLM scheduler max_num_batched_tokens. Set higher than the prompt "
            "length (e.g., 20000) to avoid chunked prefill and enable clear "
            "LMCache hits."
        ),
    )

    parser.add_argument(
        "--sleep-after-first-secs",
        type=float,
        default=1.0,
        help=(
            "Seconds to sleep after the first request to allow KV store to finish. "
            "Increase (e.g., 8) to reliably observe cache hits in the second request."
        ),
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=500,
        help=(
            "Repeat count for the synthetic chunks (default: 500). "
            "Lower (e.g., 300) to keep total prompt length under 8192 to avoid "
            "chunked prefill completely."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Build tokens once to ensure the "same data" across scenarios
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    sep_tokens = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR", " # # "))[1:]
    chunk1_prompt = tokenizer.encode("Hello, how are you?" * args.repeat)[1:]
    chunk2_prompt = tokenizer.encode("Hello, what's up?" * args.repeat)[1:]
    chunk3_prompt = tokenizer.encode("Hi, what are you up to?" * args.repeat)[1:]
    warmup_prompt = tokenizer.encode("Nice to meet you" * args.repeat)[1:]

    target_prompt = (
        sys_prompt
        + sep_tokens
        + chunk1_prompt
        + sep_tokens
        + chunk2_prompt
        + sep_tokens
        + chunk3_prompt
        + sep_tokens
        + tokenizer.encode("Hello, my name is")[1:]
    )
    prefix1_store = sys_prompt + chunk1_prompt
    prefix1_second = sys_prompt + chunk1_prompt + chunk2_prompt + chunk3_prompt
    blend3_first = target_prompt
    blend3_second = (
        sys_prompt
        + sep_tokens
        + chunk2_prompt
        + sep_tokens
        + chunk1_prompt
        + sep_tokens
        + chunk3_prompt
        + sep_tokens
        + tokenizer.encode("Hello, how are you?")[1:]
    )
    blend3_third = (
        sys_prompt
        + sep_tokens
        + chunk3_prompt
        + sep_tokens
        + chunk1_prompt
        + sep_tokens
        + chunk2_prompt
        + sep_tokens
        + tokenizer.encode("Hello, what's up?")[1:]
    )

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    def run_one_scenario(name: str, enable_blending: bool):
        setup_environment_variables(
            args.use_disk,
            args.blend_special_str,
            args.enable_sparse,
            args.enable_async_loading,
            args.blend_recompute_ratios,
            args.blend_check_layers,
            enable_blending=enable_blending,
        )
        with build_llm_with_lmcache(
            lmcache_connector,
            model,
            args.max_num_batched_tokens,
            args.tokenizer_mode,
        ) as llm:
            summary: dict[str, float] = {}
            d = print_output(llm, warmup_prompt, sampling_params, f"{name}-warmup")
            summary["warmup"] = d
            if name == "pure":
                os.environ["LMCACHE_FORCE_SKIP_SAVE"] = "True"
                try:
                    summary["pure"] = print_output(
                        llm, target_prompt, sampling_params, "pure"
                    )
                finally:
                    os.environ.pop("LMCACHE_FORCE_SKIP_SAVE", None)
            elif name == "prefix1":
                summary["store"] = print_output(
                    llm, prefix1_store, sampling_params, "prefix1-store"
                )
                time.sleep(args.sleep_after_first_secs)
                summary["second"] = print_output(
                    llm, prefix1_second, sampling_params, "prefix1-second"
                )
            else:  # blend3
                summary["first"] = print_output(
                    llm, blend3_first, sampling_params, "blend3-first"
                )
                time.sleep(args.sleep_after_first_secs)
                summary["second"] = print_output(
                    llm, blend3_second, sampling_params, "blend3-second"
                )
                time.sleep(args.sleep_after_first_secs)
                summary["third"] = print_output(
                    llm, blend3_third, sampling_params, "blend3-third"
                )
            return summary

    if args.scenario == "compare":
        results = {}
        results["pure"] = run_one_scenario("pure", enable_blending=False)
        results["prefix1"] = run_one_scenario("prefix1", enable_blending=False)
        results["blend3"] = run_one_scenario("blend3", enable_blending=True)
        print("==== Summary (seconds) ====")
        for k, v in results.items():
            print(k, v)
    elif args.scenario == "pure":
        run_one_scenario("pure", enable_blending=False)
    elif args.scenario == "prefix1":
        run_one_scenario("prefix1", enable_blending=False)
    else:
        run_one_scenario("blend3", enable_blending=True)


if __name__ == "__main__":
    main()
