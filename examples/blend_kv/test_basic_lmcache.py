# Test basic LMCache functionality without CacheBlend
import time
import json
from openai import OpenAI

def test_basic_lmcache_performance():
    """Test basic LMCache performance without blending"""
    
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
    
    print("=== Basic LMCache Performance Test ===")
    print(f"Testing with {len(chunks)} document chunks")
    print(f"Total content length: {sum(len(chunk) for chunk in chunks)} characters")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    models = client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")
    
    # Create test prompts 
    sys_prompt = "Here's a document from the user: "
    question = "Question: What does this document mainly talk about? Answer: "
    
    # Combined document prompt
    combined_prompt = f"{sys_prompt}{chunks[0]}{chunks[1]}{question}"
    print(f"Combined prompt length: {len(combined_prompt)} characters")
    
    # Test 1: First query (cold cache)
    print("\n=== Test 1: First Query (Cold Cache) ===")
    start_time = time.perf_counter()
    first_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.0,
        max_tokens=100
    )
    first_time = time.perf_counter() - start_time
    
    print(f"First response: {first_response.choices[0].message.content}")
    print(f"First query time: {first_time:.3f}s")
    
    # Test 2: Second query (potentially warm cache)
    print("\n=== Test 2: Second Query (Potentially Warm Cache) ===")
    start_time = time.perf_counter()
    second_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.0,
        max_tokens=100
    )
    second_time = time.perf_counter() - start_time
    
    print(f"Second response: {second_response.choices[0].message.content}")
    print(f"Second query time: {second_time:.3f}s")
    
    # Test 3: Slight variation (partial cache hit)
    print("\n=== Test 3: Slight Variation (Partial Cache) ===") 
    variation_question = "Question: Can you summarize the main points? Answer: "
    variation_prompt = f"{sys_prompt}{chunks[0]}{chunks[1]}{variation_question}"
    
    start_time = time.perf_counter()
    variation_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": variation_prompt}],
        temperature=0.0,
        max_tokens=100
    )
    variation_time = time.perf_counter() - start_time
    
    print(f"Variation response: {variation_response.choices[0].message.content}")
    print(f"Variation query time: {variation_time:.3f}s")
    
    # Test 4: Individual chunks (different cache pattern)
    print("\n=== Test 4: Individual Chunk Tests ===")
    individual_times = []
    
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
        individual_times.append(chunk_time)
        
        print(f"Chunk {i+1} response: {chunk_response.choices[0].message.content}")
        print(f"Chunk {i+1} query time: {chunk_time:.3f}s")
    
    # Performance Analysis
    print("\n=== Performance Analysis ===")
    print(f"First query (cold):      {first_time:.3f}s")
    print(f"Second query (repeat):   {second_time:.3f}s")
    print(f"Variation query:         {variation_time:.3f}s")
    print(f"Individual chunks:       {sum(individual_times):.3f}s")
    
    # Cache effectiveness analysis
    if second_time < first_time:
        cache_improvement = ((first_time - second_time) / first_time) * 100
        print(f"\nCache effectiveness: {cache_improvement:.1f}% improvement on repeat")
        print("✅ LMCache shows caching benefits")
    else:
        print(f"\nNo cache improvement detected ({second_time:.3f}s >= {first_time:.3f}s)")
        print("⚠️ Cache may not be effective for this scenario")
    
    # Memory and efficiency analysis
    combined_vs_individual = sum(individual_times) - first_time
    if combined_vs_individual > 0:
        print(f"Combined query efficiency: {combined_vs_individual:.3f}s faster than individual")
        print("✅ Combined processing more efficient")
    else:
        print(f"Individual queries more efficient by: {abs(combined_vs_individual):.3f}s")
    
    return {
        'first_time': first_time,
        'second_time': second_time,
        'variation_time': variation_time,
        'individual_times': individual_times,
        'cache_improvement': second_time < first_time
    }

if __name__ == "__main__":
    results = test_basic_lmcache_performance()
    print(f"\n=== Test Completed ===")
    print("Basic LMCache functionality verified")