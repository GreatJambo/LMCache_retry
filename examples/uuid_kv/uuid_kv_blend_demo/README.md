# UUID KV Blend Demo (vLLM-integrated)

This demo shows how to:
- Precompute KV for multiple chunks with vLLM
- Register each chunk to a UUID in LMCache
- Blend multiple KV caches by reordering chunks in the final prompt

By default, the final prompt concatenation omits separators (recommended when composing from pre-tokenized pieces). You can enable separators in the final prompt with `--use-sep-in-final` if you are constructing prompts from strings and want stable token boundaries.

## Run

From the project root:

```bash
# Disk-backed (recommended for UUID verification)
python3 examples/uuid_kv/uuid_kv_blend_demo/uuid_kv_blend.py \
  --use-disk --chunks-json examples/uuid_kv/uuid_kv_blend_demo/chunks.json \
  --enable-async-loading --sleep-after-first-secs 2 --verify-uuid

# In-memory (CPU) only
python3 examples/uuid_kv/uuid_kv_blend_demo/uuid_kv_blend.py \
  --chunks-json examples/uuid_kv/uuid_kv_blend_demo/chunks.json

# Use separators in final prompt
python3 examples/uuid_kv/uuid_kv_blend_demo/uuid_kv_blend.py \
  --chunks-json examples/uuid_kv/uuid_kv_blend_demo/chunks.json --use-sep-in-final
```

Arguments:
- `--model`: vLLM model name (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `--tokenizer-mode`: vLLM tokenizer mode (default: `auto`)
- `--blend-check-layers`, `--blend-recompute-ratios`: override blending parameters
- `--use-disk`: enable LMCache disk backend at `local_disk/`
- `--enable-async-loading`: enable background KV loading
- `--verify-uuid`: check `retrieve_by_uuid` via a minimal LMCache engine (requires `--use-disk`)
- `--chunks-json`: path to a JSON file containing chunk texts
- `--sleep-after-first-secs`: wait time for KV persistence between steps
- `--use-sep-in-final`: insert separators between chunks in the final prompt

### Chunks JSON format

Example `chunks.json`:

```json
{
  "chunks": {
    "chunk1": "Hello, how are you? This is the first chunk.",
    "chunk2": "Hello, what's up? This is the second chunk.",
    "chunk3": "Hi, what are you up to? This is the third chunk."
  }
}
```

## Notes

- Blending requires `LMCACHE_ENABLE_BLENDING=True` and `LMCACHE_USE_LAYERWISE=True` (set by the script).
- Separators are not mandatory for blending; they are mainly helpful when stitching strings and re-tokenizing. If you already operate on token sequences per chunk, omitting separators is fine.
- UUID verification with a dummy engine is a convenience check to confirm UUID mapping correctness; it performs a no-op GPU transfer.

