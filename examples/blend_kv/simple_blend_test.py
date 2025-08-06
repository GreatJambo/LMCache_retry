# Simple test for CacheBlend functionality without OnlineKVPreCompute
import time
import json
from openai import OpenAI
from lmcache_vllm.blend_adapter import combine_input_prompt_chunks

def test_cache_blend_functionality():
    """Test CacheBlend functionality by comparing blended vs non-blended queries"""
    
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
    
    print("=== CacheBlend Functionality Test ===")
    print(f"Testing with {len(chunks)} document chunks")
    print(f"Chunk 1 length: {len(chunks[0])} characters")  
    print(f"Chunk 2 length: {len(chunks[1])} characters")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")
    
    # Create prompts
    sys_prompt = "Here's a document from the user: "
    question = "Question: What does this document mainly talk about? Answer: "
    
    # Test 1: Blended prompt using CacheBlend separator
    print("\n=== Test 1: Blended Query (with CacheBlend separators) ===")
    blended_prompt = combine_input_prompt_chunks([sys_prompt, chunks[0], chunks[1], question])
    print(f"Blended prompt length: {len(blended_prompt)} characters")
    print("Blended prompt preview:", blended_prompt[:200] + "..." if len(blended_prompt) > 200 else blended_prompt)
    
    start_time = time.perf_counter()
    blended_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": blended_prompt}],
        temperature=0.0,
        max_tokens=100
    )
    blended_time = time.perf_counter() - start_time
    
    print(f"Blended response: {blended_response.choices[0].message.content}")
    print(f"Blended query time: {blended_time:.3f}s")
    
    # Test 2: Regular concatenated prompt (baseline)
    print("\n=== Test 2: Regular Query (baseline, no separators) ===") 
    regular_prompt = f"{sys_prompt}{chunks[0]}{chunks[1]}{question}"
    print(f"Regular prompt length: {len(regular_prompt)} characters")
    
    start_time = time.perf_counter()
    regular_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": regular_prompt}],
        temperature=0.0, 
        max_tokens=100
    )
    regular_time = time.perf_counter() - start_time
    
    print(f"Regular response: {regular_response.choices[0].message.content}")
    print(f"Regular query time: {regular_time:.3f}s")
    
    # Test 3: Individual chunk queries
    print("\n=== Test 3: Individual Chunk Queries ===")
    chunk_times = []
    
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"{sys_prompt}{chunk}{question}"
        print(f"Chunk {i+1} prompt length: {len(chunk_prompt)} characters")
        
        start_time = time.perf_counter()
        chunk_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": chunk_prompt}],
            temperature=0.0,
            max_tokens=50
        )
        chunk_time = time.perf_counter() - start_time
        chunk_times.append(chunk_time)
        
        print(f"Chunk {i+1} response: {chunk_response.choices[0].message.content}")
        print(f"Chunk {i+1} query time: {chunk_time:.3f}s")
    
    # Performance Analysis
    print("\n=== Performance Analysis ===")
    print(f"Blended query time:     {blended_time:.3f}s")
    print(f"Regular query time:     {regular_time:.3f}s") 
    print(f"Individual chunks time: {sum(chunk_times):.3f}s")
    
    if blended_time < regular_time:
        improvement = ((regular_time - blended_time) / regular_time) * 100
        print(f"Blended vs Regular improvement: {improvement:.1f}%")
        print("✅ CacheBlend separators provide performance benefit!")
    else:
        degradation = ((blended_time - regular_time) / regular_time) * 100  
        print(f"Blended vs Regular degradation: {degradation:.1f}%")
        print("⚠️ No performance improvement from separators")
    
    # Test separator functionality
    print("\n=== Separator Analysis ===")
    print("CacheBlend separator found in blended prompt:", " # #<s> " in blended_prompt)
    separator_count = blended_prompt.count(" # #<s> ")
    print(f"Number of separators: {separator_count}")
    
    if separator_count > 0:
        print("✅ CacheBlend separators are correctly inserted")
        print("This enables the system to identify and blend cached chunks")
    else:
        print("❌ No CacheBlend separators found")
        
    return {
        'blended_time': blended_time,
        'regular_time': regular_time,
        'chunk_times': chunk_times,
        'separator_count': separator_count
    }

if __name__ == "__main__":
    results = test_cache_blend_functionality()
    print(f"\n=== Test completed ===")
    print("Results saved for analysis")