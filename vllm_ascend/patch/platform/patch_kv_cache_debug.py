# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Verbose KV cache debug logging for learning/debugging purposes.

Enabled by env var ``VLLM_ASCEND_KV_DEBUG=1`` (see vllm_ascend/envs.py).
This module is only imported when the env var is on, so it adds zero
overhead when disabled.

It wraps the upstream scheduler-side KV cache components and prints one
"[KV_DEBUG]" log line per event so that a single request's lifecycle can
be followed end to end in the EngineCore process:

- ``[KV_DEBUG][INIT]``: block pool size, block_size, page size per group.
- ``[KV_DEBUG][STEP]``: start of every scheduler step.
- ``[KV_DEBUG][PREFIX]``: prefix cache hit blocks for a waiting request.
- ``[KV_DEBUG][ALLOC]``: blocks handed to a request by ``allocate_slots``.
- ``[KV_DEBUG][BLOCK_POOL]``: physical block ids taken from the free pool.
- ``[KV_DEBUG][SCHED]``: per-step result: tokens scheduled, request block
  lists and free block accounting.
- ``[KV_DEBUG][FREE]``: blocks returned to the pool on finish/preemption.

Key relationship to keep in mind while reading the logs:
    slot (where one token's KV lives in the cache tensor)
        = block_id * block_size + offset_in_block
"""

from functools import wraps
from typing import Any

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.utils import kv_debug_format_ids


def _log_kv_debug_error(where: str, exc_info: bool = True) -> None:
    logger.debug("[KV_DEBUG] failed to build %s log", where, exc_info=exc_info)


def _req_blocks_snapshot(kv_cache_manager: KVCacheManager, request_id: str) -> tuple[list[int], list[list[int]]]:
    """Return (per-group block counts, per-group block id lists) of a request.

    ``req_to_blocks`` is ``dict[str, list[KVCacheBlock]]`` for the single
    group coordinator and ``dict[str, list[list[KVCacheBlock]]]`` for the
    hybrid coordinator, so both shapes are handled.
    """
    req_to_blocks = getattr(kv_cache_manager.coordinator, "req_to_blocks", None)
    if req_to_blocks is None:
        return [], []
    blocks = req_to_blocks.get(request_id)
    if not blocks:
        return [], []
    if isinstance(blocks[0], (list, tuple)):
        return [len(group) for group in blocks], [[b.block_id for b in group] for group in blocks]
    return [len(blocks)], [[b.block_id for b in blocks]]


_original_kv_cache_manager_init = KVCacheManager.__init__


@wraps(_original_kv_cache_manager_init)
def _patched_kv_cache_manager_init(self, *args: Any, **kwargs: Any) -> None:
    _original_kv_cache_manager_init(self, *args, **kwargs)
    try:
        config = getattr(self.coordinator, "kv_cache_config", None)
        if config is None:
            logger.info("[KV_DEBUG][INIT] KVCacheManager 初始化完成 (未获取到 kv_cache_config 详情)")
            return
        groups = config.kv_cache_groups
        logger.info("[KV_DEBUG][INIT] ============ KV Cache 块池初始化 (Scheduler 进程) ============")
        logger.info(
            "[KV_DEBUG][INIT] 物理块池大小 num_blocks=%d: 所有请求共享这 %d 个块, "
            "请求每写满一个块或需要新块时, 都从这里领取空闲块",
            config.num_blocks,
            config.num_blocks,
        )
        total_bytes = 0
        for gid, group in enumerate(groups):
            spec = group.kv_cache_spec
            block_size = int(getattr(spec, "block_size", 0) or 0)
            page_size = int(getattr(spec, "page_size_bytes", 0) or 0)
            total_bytes += page_size * config.num_blocks
            per_token_kb = (page_size / block_size / 1024) if block_size else 0
            logger.info(
                "[KV_DEBUG][INIT] group[%d]: block_size=%d (1个block存%d个token) | 本组层数=%d | "
                "1个block占显存 %.2f MiB(该组所有层合计) | 每个token的KV占 %.1f KB(该组所有层合计)",
                gid,
                block_size,
                block_size,
                len(group.layer_names),
                page_size / 1048576,
                per_token_kb,
            )
            logger.info("[KV_DEBUG][INIT] group[%d] 层列表(前5个): %s", gid, list(group.layer_names[:5]))
        logger.info(
            "[KV_DEBUG][INIT] KV池总显存 = %.2f GiB (= num_blocks × 各group每块page_size之和)",
            total_bytes / 1073741824,
        )
        logger.info(
            "[KV_DEBUG][INIT] 初始空闲块 = %d/%d",
            self.block_pool.get_num_free_blocks(),
            config.num_blocks,
        )
        logger.info(
            "[KV_DEBUG][INIT] 核心映射公式: 某token的KV写入位置 slot = block_id * block_size + 块内偏移; "
            "每个请求占用的block_id列表 = BlockTable中该请求对应的一行"
        )
    except Exception:
        _log_kv_debug_error("kv cache manager init")


_original_get_computed_blocks = KVCacheManager.get_computed_blocks


@wraps(_original_get_computed_blocks)
def _patched_get_computed_blocks(self, request, *args: Any, **kwargs: Any):
    blocks = _original_get_computed_blocks(self, request, *args, **kwargs)
    try:
        block_ids = blocks.get_block_ids()
        logger.info(
            "[KV_DEBUG][PREFIX] req=%s 前缀缓存查找: 命中块数=%s 命中块id=%s "
            "(这些块的KV内容已存在, 请求无需重新计算这部分token)",
            request.request_id,
            [len(x) for x in block_ids],
            [kv_debug_format_ids(x) for x in block_ids],
        )
    except Exception:
        _log_kv_debug_error("get_computed_blocks")
    return blocks


_original_allocate_slots = KVCacheManager.allocate_slots


@wraps(_original_allocate_slots)
def _patched_allocate_slots(self, request, num_new_tokens, *args: Any, **kwargs: Any):
    free_before = self.block_pool.get_num_free_blocks()
    result = _original_allocate_slots(self, request, num_new_tokens, *args, **kwargs)
    try:
        if result is None:
            logger.info(
                "[KV_DEBUG][ALLOC] req=%s 分配失败(返回None): 本步要为 %d 个新token找slot, "
                "但空闲块只有 %d 个 → Scheduler 将抢占(preempt)其他请求腾出空间后重试",
                request.request_id,
                num_new_tokens,
                free_before,
            )
            return result
        new_block_ids = result.get_block_ids()
        num_computed = kwargs.get("num_new_computed_tokens", 0)
        counts, cur_ids = _req_blocks_snapshot(self, request.request_id)
        logger.info(
            "[KV_DEBUG][ALLOC] req=%s 分配成功: 本步新token=%d (其中前缀命中可复用=%s) → 新分配块数=%s 新块id=%s",
            request.request_id,
            num_new_tokens,
            num_computed,
            [len(x) for x in new_block_ids],
            [kv_debug_format_ids(x) for x in new_block_ids],
        )
        logger.info(
            "[KV_DEBUG][ALLOC] req=%s 该请求累计占用块数=%s 块id=%s | 空闲块 %d → %d",
            request.request_id,
            counts,
            [kv_debug_format_ids(x) for x in cur_ids],
            free_before,
            self.block_pool.get_num_free_blocks(),
        )
    except Exception:
        _log_kv_debug_error("allocate_slots")
    return result


_original_free = KVCacheManager.free


@wraps(_original_free)
def _patched_free(self, requests, *args: Any, **kwargs: Any) -> None:
    request_list = requests if isinstance(requests, (list, tuple)) else [requests]
    snapshots = []
    for request in request_list:
        request_id = getattr(request, "request_id", None)
        if request_id is not None:
            snapshots.append((request_id, _req_blocks_snapshot(self, request_id)))
    free_before = self.block_pool.get_num_free_blocks()
    _original_free(self, requests, *args, **kwargs)
    try:
        for request_id, (counts, ids) in snapshots:
            if not counts:
                continue
            logger.info(
                "[KV_DEBUG][FREE] req=%s 请求完成/被抢占, 归还块数=%s 块id=%s | 空闲块 %d → %d",
                request_id,
                counts,
                [kv_debug_format_ids(x) for x in ids],
                free_before,
                self.block_pool.get_num_free_blocks(),
            )
    except Exception:
        _log_kv_debug_error("free")


_original_get_new_blocks = BlockPool.get_new_blocks


@wraps(_original_get_new_blocks)
def _patched_get_new_blocks(self, num_blocks, *args: Any, **kwargs: Any):
    free_before = self.get_num_free_blocks()
    blocks = _original_get_new_blocks(self, num_blocks, *args, **kwargs)
    try:
        logger.info(
            "[KV_DEBUG][BLOCK_POOL] 从空闲块池领取 %d 个物理块: 块id=%s | 空闲块 %d → %d",
            num_blocks,
            kv_debug_format_ids([b.block_id for b in blocks]),
            free_before,
            self.get_num_free_blocks(),
        )
    except Exception:
        _log_kv_debug_error("get_new_blocks")
    return blocks


_original_schedule = Scheduler.schedule


@wraps(_original_schedule)
def _patched_schedule(self, *args: Any, **kwargs: Any):
    self._kv_debug_step_count = getattr(self, "_kv_debug_step_count", 0) + 1
    try:
        logger.info(
            "[KV_DEBUG][STEP] ====== 调度步 #%d 开始: running=%d waiting=%d 空闲块=%d/%d ======",
            self._kv_debug_step_count,
            len(self.running),
            len(self.waiting),
            self.kv_cache_manager.block_pool.get_num_free_blocks(),
            self.kv_cache_manager.block_pool.num_gpu_blocks,
        )
    except Exception:
        _log_kv_debug_error("schedule")
    return _original_schedule(self, *args, **kwargs)


_original_update_after_schedule = Scheduler._update_after_schedule


@wraps(_original_update_after_schedule)
def _patched_update_after_schedule(self, scheduler_output, *args: Any, **kwargs: Any) -> None:
    _original_update_after_schedule(self, scheduler_output, *args, **kwargs)
    try:
        pool = self.kv_cache_manager.block_pool
        free_now = pool.get_num_free_blocks()
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id)
            request = self.requests.get(req_id)
            num_computed = getattr(request, "num_computed_tokens", None)
            new_block_ids = cached.new_block_ids[i] if i < len(cached.new_block_ids) else None
            new_flat = [b for group in (new_block_ids or []) for b in group]
            counts, cur_ids = _req_blocks_snapshot(self.kv_cache_manager, req_id)
            state = f"占用块数={counts} 块id={[kv_debug_format_ids(x) for x in cur_ids]}"
            if not counts:
                state = "请求本步已完成(块已归还块池)"
            logger.info(
                "[KV_DEBUG][SCHED] 步#%d 调度结果: req=%s 本步调度token=%s 已计算token=%s "
                "| 该请求%s 本步新增块id=%s | 空闲块=%d/%d",
                getattr(self, "_kv_debug_step_count", 0),
                req_id,
                num_scheduled,
                num_computed,
                state,
                kv_debug_format_ids(new_flat),
                free_now,
                pool.num_gpu_blocks,
            )
        logger.info(
            "[KV_DEBUG][SCHED] 本步汇总: 调度%d个请求 总token=%d running=%d waiting=%d",
            len(cached.req_ids),
            scheduler_output.total_num_scheduled_tokens,
            len(self.running),
            len(self.waiting),
        )
    except Exception:
        _log_kv_debug_error("update_after_schedule")


KVCacheManager.__init__ = _patched_kv_cache_manager_init
KVCacheManager.get_computed_blocks = _patched_get_computed_blocks
KVCacheManager.allocate_slots = _patched_allocate_slots
KVCacheManager.free = _patched_free
BlockPool.get_new_blocks = _patched_get_new_blocks
Scheduler.schedule = _patched_schedule
Scheduler._update_after_schedule = _patched_update_after_schedule
