# SPDX-License-Identifier: Apache-2.0
# Simple microbenchmark test for KV cache blending
import time
import json
from openai import OpenAI
from transformers import AutoTokenizer
from lmcache_vllm.blend_adapter import (
    OnlineKVPreCompute,
    combine_input_prompt_chunks,
)

def main():
    # Server configuration
    port = 8000
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    
    # Load context chunks
    context_files = ["chunk1.txt", "chunk2.txt"]
    chunks = []
    
    for context_file in context_files:
        with open(context_file, "r") as fin:
            content = fin.read()
        chunks.append(content)
    
    print("=== CacheBlend KV Cache Blending Microbenchmark ===")
    print(f"Loading chunks: {len(chunks)} chunks")
    print(f"Chunk 1 length: {len(chunks[0])} characters")
    print(f"Chunk 2 length: {len(chunks[1])} characters")
    
    # Initialize tokenizer and precompute service
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    precompute_kv = OnlineKVPreCompute(openai_api_key, openai_api_base, tokenizer)
    
    # OpenAI client for queries
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")
    
    # Phase 1: Pre-compute KV cache for chunks
    print("\n=== Phase 1: Pre-computing KV cache for chunks ===")
    precompute_times = []
    
    for i, chunk in enumerate(chunks):
        start_time = time.perf_counter()
        precompute_kv.precompute_kv(chunk)
        end_time = time.perf_counter()
        precompute_time = end_time - start_time
        precompute_times.append(precompute_time)
        print(f"Pre-computed chunk {i+1} in {precompute_time:.3f}s")
    
    # Phase 2: Test blended query (should use cached KV)
    print("\n=== Phase 2: Testing KV cache blending ===")
    sys_prompt = "Here's a document from the user: "
    question = "Question: What does this document mainly talk about? Answer: "
    
    # Create blended prompt using the combine function
    user_prompt = combine_input_prompt_chunks([sys_prompt, chunks[0], chunks[1], question])
    
    print("Sending blended query to server...")
    start_time = time.perf_counter()
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Total query time: {total_time:.3f}s")
    
    # Phase 3: Performance analysis
    print("\n=== Phase 3: Performance Analysis ===")
    print(f"Pre-compute Phase:")
    for i, ptime in enumerate(precompute_times):
        print(f"  Chunk {i+1}: {ptime:.3f}s")
    print(f"  Total pre-compute time: {sum(precompute_times):.3f}s")
    
    print(f"\nBlended Query Phase:")
    print(f"  Query response time: {total_time:.3f}s")
    print(f"  Expected benefit: Reduced prefill time due to cached KV")
    
    # Phase 4: Compare with non-blended query (baseline)
    print("\n=== Phase 4: Baseline comparison (no caching) ===")
    baseline_prompt = f"{sys_prompt}{chunks[0]}{chunks[1]}{question}"
    
    start_time = time.perf_counter()
    baseline_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": baseline_prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
    end_time = time.perf_counter()
    baseline_time = end_time - start_time
    
    print(f"Baseline response: {baseline_response.choices[0].message.content}")
    print(f"Baseline query time: {baseline_time:.3f}s")
    
    # Final analysis
    print("\n=== Final Performance Summary ===")
    print(f"Blended query time:   {total_time:.3f}s")
    print(f"Baseline query time:  {baseline_time:.3f}s")
    if baseline_time > total_time:
        improvement = ((baseline_time - total_time) / baseline_time) * 100
        print(f"Performance improvement: {improvement:.1f}%")
        print("✅ KV cache blending is working - reduced query time!")
    else:
        print("❌ No performance improvement detected")
    
    print(f"Total pre-compute overhead: {sum(precompute_times):.3f}s")
    net_benefit = baseline_time - total_time - sum(precompute_times)
    if net_benefit > 0:
        print(f"Net benefit (accounting for pre-compute): {net_benefit:.3f}s")
    else:
        print(f"Net overhead: {abs(net_benefit):.3f}s")

if __name__ == "__main__":
    main()