# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import os
import time
import uuid as uuidlib

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


class DummyGPUConnector(GPUConnectorInterface):
    def __init__(self, hidden_dim_size: int, num_layers: int):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        return None

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.tensor is not None
        memory_obj.tensor.copy_(
            torch.randn_like(memory_obj.tensor).to(memory_obj.tensor.device)
        )
        memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        return None

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--disk", type=str, default="file://examples/uuid_kv/local_disk/",
                   help="Local disk path for LMCache (file://...) or plain path")
    p.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                   help="Model name recorded in keys (no weights needed)")
    p.add_argument("--tokens", type=int, default=640,
                   help="Number of prompt tokens to simulate")
    p.add_argument("--chunk", type=int, default=64, help="LMCache chunk size")
    p.add_argument("--hidden", type=int, default=64, help="Hidden dim size for KV")
    p.add_argument("--layers", type=int, default=2, help="Number of layers")
    p.add_argument("--uuid", type=str, default=None, help="UUID to use (optional)")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous KV and UUID index files under --disk before running",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Optionally clean previous data before engine starts
    if args.clean:
        # Normalize disk directory similar to engine/backend
        from urllib.parse import urlparse
        disk_dir = args.disk
        try:
            parsed = urlparse(disk_dir)
            if parsed.scheme == "file":
                netloc = parsed.netloc
                path = parsed.path or ""
                disk_dir = os.path.join(netloc, path.lstrip("/")) if netloc else path
        except Exception:
            pass
        os.makedirs(disk_dir, exist_ok=True)

        removed = 0
        for name in os.listdir(disk_dir):
            if name.endswith('.pt') or name.startswith('lmcache_uuid_index.sqlite'):
                try:
                    os.remove(os.path.join(disk_dir, name))
                    removed += 1
                except Exception:
                    pass
        print(f"Cleaned {removed} file(s) under {disk_dir}")

    # Configure LMCache v1
    cfg = LMCacheEngineConfig.from_defaults()
    cfg.chunk_size = args.chunk
    cfg.local_cpu = True
    cfg.max_local_cpu_size = 1.0
    # enable disk persistence
    cfg.local_disk = args.disk
    cfg.max_local_disk_size = 1.0
    cfg.pre_caching_hash_algorithm = "sha256"
    cfg.use_layerwise = False

    # Prepare metadata
    kv_dtype = torch.float16
    kv_shape = (args.layers, 2, args.chunk, 1, args.hidden)
    meta = LMCacheEngineMetadata(
        model_name=args.model,
        world_size=1,
        worker_id=0,
        fmt="vllm",
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        use_mla=False,
    )

    # Build a minimal engine with a dummy GPU connector
    connector = DummyGPUConnector(args.hidden, args.layers)

    def _bcast_fn(t: torch.Tensor, dst: int):
        return None

    def _bcast_obj_fn(obj, dst: int):
        return obj

    instance_id = "uuid_kv_demo"
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        cfg,
        meta,
        connector,
        _bcast_fn,
        _bcast_obj_fn,
    )
    engine.post_init()

    # Simulate tokens
    torch.manual_seed(0)
    tok = torch.randint(10, 10000, (args.tokens,), dtype=torch.long)

    # Store to LMCache (will allocate CPU memory and write to disk)
    print("Storing KV for tokens ...")
    engine.store(tokens=tok)
    time.sleep(0.2)

    # Register UUID mapping
    uid = args.uuid or str(uuidlib.uuid4())
    total = engine.register_uuid(uid, tokens=tok.tolist())
    print(f"Registered UUID={uid}, total_tokens={total}")

    # Retrieve by UUID (no tokens provided)
    print("Retrieving by UUID ...")
    ret_mask = engine.retrieve_by_uuid(uid)
    got = int(torch.sum(ret_mask).item())
    print(f"Retrieved tokens by UUID: {got} / {total}")

    assert got == total, "retrieve_by_uuid did not load full prefix"
    print("OK: full prefix loaded via UUID mapping.")

    engine.close()


if __name__ == "__main__":
    main()


