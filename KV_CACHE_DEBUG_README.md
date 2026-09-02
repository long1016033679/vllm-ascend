# KV Cache Debug 日志使用指南（VLLM_ASCEND_KV_DEBUG）

本文档介绍如何使用 vLLM-Ascend 内置的 KV Cache 教学日志，理解**单个请求**在推理全过程中：

- KV Cache 块池如何分配 / 释放物理块
- 请求与物理块的映射（BlockTable）如何更新
- 每个 token 的写入位置（slot_mapping）如何计算
- 每层 Attention 如何使用这些映射读写 KV

所有日志以 `[KV_DEBUG]` 开头，通过环境变量 `VLLM_ASCEND_KV_DEBUG=1` 开启，默认关闭、零开销。仅用于调试和学习，日志量大且会拖慢服务，请勿在生产环境开启。

## 1. 快速开始

```bash
# 必须在启动 vllm serve 之前 export，子进程才能继承
export VLLM_ASCEND_KV_DEBUG=1

# 建议 --enforce-eager：decode 走 aclgraph 图回放时看不到每层写入日志
vllm serve /path/to/glm-model --enforce-eager ... 2>&1 | tee kv_debug.log
```

发送一个请求，然后在日志中跟踪：

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 20}'

grep "KV_DEBUG" kv_debug.log
```

日志全部为 INFO 级别，无需再设置 `VLLM_LOGGING_LEVEL=DEBUG`。

## 2. 日志来自两个进程

vLLM v1 架构分为调度器（EngineCore）和执行器（Worker）两个进程，日志从两边打印到同一控制台输出：

| 进程 | 职责 | 日志标签 |
| --- | --- | --- |
| EngineCore | 决定"给请求分哪些块" | `INIT` `STEP` `PREFIX` `BLOCK_POOL` `ALLOC` `SCHED` `FREE` |
| Worker | 决定"这些块怎么被 kernel 用" | `INIT` `BLK_TABLE` `PREPARE` `SLOT` `ATTN_META` `KV_WRITE` |

## 3. 日志标签总览

| 标签 | 打印时机 | 内容 |
| --- | --- | --- |
| `[INIT]` | 启动时各打印一次 | 块池大小 `num_blocks`、`block_size`、每块/每 token 显存占用、每层 cache tensor 形状 |
| `[STEP]` | 每个调度步开始 | running/waiting 请求数、当前空闲块数 |
| `[PREFIX]` | 新请求进入时 | 前缀缓存命中的块 id（第二次发相同 prompt 会命中） |
| `[BLOCK_POOL]` | 每次发放块时 | 从空闲块池领取的物理块 id、空闲块变化 |
| `[ALLOC]` | 每次调度分配 | 新分配块数与块 id、请求累计占用块列表、空闲块前后对比；失败时提示将抢占 |
| `[SCHED]` | 每步调度结束 | 本步调度 token 数、请求当前块列表、空闲块守恒 |
| `[FREE]` | 请求结束/被抢占 | 归还的块 id 列表、空闲块恢复情况 |
| `[BLK_TABLE]` | 模型执行前 | 调度器发来的块列表写入 BlockTable 对应行；CPU → NPU 同步 |
| `[PREPARE]` | 每步输入准备 | 各请求本步 token 数、已计算 token 数、`seq_lens`、`attn_state` |
| `[SLOT]` | 每步计算映射时 | positions 与 slot_mapping 的具体数值（核心验证点） |
| `[ATTN_META]` | 每步构建元数据 | attention kernel 实际拿到的 block_table 行与 slot_mapping |
| `[KV_WRITE]` | 每层写入时 | 每层把 K/V 写入 cache 的真实 slot、cache tensor 形状（GQA 与 MLA 后端均有） |

## 4. 常用查看命令

```bash
# 实时滚动（只看教学日志）
tail -f kv_debug.log | grep --line-buffered "KV_DEBUG"

# 滤掉每层写入（最密集的部分）
tail -f kv_debug.log | grep --line-buffered "KV_DEBUG" | grep -v "KV_WRITE"

# 只看某一类事件
grep "\[INIT\]" kv_debug.log        # 启动信息（看一次）
grep "\[ALLOC\]" kv_debug.log       # 分配
grep "\[FREE\]" kv_debug.log        # 释放
grep "\[SLOT\]" kv_debug.log        # slot 映射

# 只追一个请求（req_id 可从日志或 API 响应的 id 中获取）
grep "KV_DEBUG" kv_debug.log | grep "req=<request_id>"

# 只看第一层的写入
grep "KV_WRITE" kv_debug.log | grep "layers.0."
```

## 5. 单请求生命周期走读

一个请求从进入到结束，日志按以下顺序出现：

```text
启动阶段（一次）
  [INIT] KV Cache 块池初始化 (Scheduler 进程)    # num_blocks、block_size=128
  [INIT] KV Cache 张量初始化完成 (Worker 进程)   # 每层 cache 形状 [2, num_blocks, 128, ...]

请求进入（waiting -> running）
  [STEP] 新调度步开始
  [PREFIX] 命中块数=[0]                          # 第二次发相同 prompt 时 >0
  [BLOCK_POOL] 领取 N 个物理块                   # N = ceil(prompt 长度 / block_size)
  [ALLOC] 分配成功 -> 该请求累计占用块数=[N]
  [SCHED] 调度结果: req=xxx_0 本步调度token=...

prefill 执行（第一步，一次）
  [BLK_TABLE] add_row(新请求首次登记)            # 调度器的块列表写进映射表
  [BLK_TABLE] commit_block_table                 # CPU -> NPU 同步
  [PREPARE] attn_state=PrefillNoCache
  [SLOT] slot_mapping=[block_id*128+0, ...]      # 验证公式的关键位置
  [ATTN_META] block_table(每行=一个请求的块列表)
  [KV_WRITE] layer=...                           # 每层一条，写入相同的 slot 列表

decode 阶段（每步重复）
  [STEP] -> [SCHED] 本步调度token=1 -> [PREPARE] seq_lens 每步 +1
  每写满一个 block 出现一次:
  [BLK_TABLE] append_row(运行中请求追加新块) + [ALLOC] 新分配 1 块

请求结束（一次）
  [FREE] 归还块数=[N] | 空闲块 X -> X+N
```

## 6. 必须看懂的三条关系

1. **slot 公式**：`slot = block_id * block_size + 块内偏移`。
   拿 prefill 那步的 `[SLOT]` 日志验证：第一个 slot 应恰好等于第一个 block_id × block_size。
2. **块数守恒**：`[ALLOC]` 中"空闲块 100 -> 99"，请求结束 `[FREE]` 中"99 -> 100"，两边能对上。
3. **三方一致**：同一个块 id 列表依次出现在 `[ALLOC]`（调度器发放）、`[BLK_TABLE]`（写入映射表）、`[ATTN_META]`（kernel 使用）中，数字完全相同。

## 7. 常见问题

| 现象 | 原因与解决 |
| --- | --- |
| 完全没有 `[KV_DEBUG]` 日志 | 环境变量未在启动 server 之前 export；或日志重定向丢失了 stderr（记得 `2>&1`） |
| 没有 `KV_WRITE` | decode 走了 aclgraph 图回放，加 `--enforce-eager` |
| 没有 `SCHED` / `ALLOC` | 这些日志在 EngineCore 进程，确认该进程输出也被收集 |
| 两进程日志交错难读 | 按标签过滤，或结合日志中的进程标识区分 |
| 日志量太大 | 优先看 `[ALLOC]` / `[SLOT]` / `[FREE]`，并过滤掉 `KV_WRITE` |

## 8. 相关代码位置

| 文件 | 内容 |
| --- | --- |
| `vllm_ascend/envs.py` | `VLLM_ASCEND_KV_DEBUG` 开关定义 |
| `vllm_ascend/patch/platform/patch_kv_cache_debug.py` | 调度器侧日志：包装 `KVCacheManager` / `BlockPool` / `Scheduler` |
| `vllm_ascend/worker/model_runner_v1.py` | Worker 侧日志：初始化、输入准备、attention 元数据 |
| `vllm_ascend/worker/block_table.py` | BlockTable 行更新与 slot_mapping 计算日志 |
| `vllm_ascend/attention/attention_v1.py` | GQA 后端每层 KV 写入日志 |
| `vllm_ascend/attention/mla_v1.py` | MLA（DeepSeek/GLM-5 DSA 类模型）每层 KV 写入日志 |
| `vllm_ascend/utils.py` | `is_kv_cache_debug_enabled` / `kv_debug_format_ids` 辅助函数 |
| `tests/ut/test_kv_cache_debug.py` | 辅助函数单元测试 |
