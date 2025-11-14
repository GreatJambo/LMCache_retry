# UUID-based KV Retrieval (LMCache v1, blocking, non-layerwise)

This minimal example demonstrates how to:
- Store KV for a synthetic prompt
- Register a UUID for that prompt (persisted in SQLite under the local disk path)
- Retrieve KV later by UUID (without tokens)

It uses a dummy GPU connector so it runs without vLLM. KV data is stored in CPU memory and persisted to a local disk backend.

## Prerequisites
- Python environment that can import this repository
- Torch installed (CUDA not required)

## Run
```bash
python examples/uuid_kv/uuid_kv.py \
  --disk file://examples/uuid_kv/local_disk/ \
  --model meta-llama/Meta-Llama-3-8B \
  --tokens 640 \
  --chunk 64 \
  --hidden 64 \
  --layers 2
```

Output should end with:
```
OK: full prefix loaded via UUID mapping.
```

Notes:
- The UUID mapping is stored in `examples/uuid_kv/local_disk/lmcache_uuid_index.sqlite`.
- KV chunks are stored under the same local disk backend.
- This example keeps `use_layerwise=false` and does a blocking retrieval.


