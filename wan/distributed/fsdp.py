# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload # scy modi
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage
from torch.distributed.fsdp import BackwardPrefetch # scy modi


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
    use_lora=False,
    cpu_offload=True # scy modi
):
    offload_config = CPUOffload(offload_params=True) if cpu_offload else None # scy modi

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        cpu_offload=offload_config,
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # scy modi
        forward_prefetch=True,  # scy modi
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False
)
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
