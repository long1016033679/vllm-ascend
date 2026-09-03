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
| Worker | 决定"这些块怎么被 kernel 用" | `INIT`(+`TENSOR↔LAYER`/`BLOCK↔TENSOR`/`SUMMARY`) `BLK_TABLE` `PREPARE` `SLOT`(+`DERIVE`) `ATTN_META`(+`GROUP`/`ROW`/`SLOT`) `KV_WRITE` |

## 3. 日志标签总览

| 标签 | 打印时机 | 内容 |
| --- | --- | --- |
| `[INIT]` | 启动时各打印一次 | 块池大小 `num_blocks`、`block_size`、每块/每 token 显存占用 |
| `[INIT][TENSOR↔LAYER]` | 启动一次 | 每层 kv_cache 形状、层与 tensor 的对应关系（不同层同编号不同内存） |
| `[INIT][BLOCK↔TENSOR]` | 启动一次 | block_id 到 tensor 位置的映射（`kv_cache[layer][0][N][i]`） |
| `[INIT][SUMMARY]` | 启动一次 | 块池、每层 tensor、映射、写入、读取五条关系总结 |
| `[STEP]` | 每个调度步开始 | 步号 `#N`、running/waiting 请求数、当前空闲块数 |
| `[PREFIX]` | 新请求进入时 | 前缀缓存命中的块 id（第二次发相同 prompt 会命中） |
| `[BLOCK_POOL]` | 每次发放块时 | 从空闲块池领取的物理块 id、空闲块变化 |
| `[ALLOC]` | 每次调度分配 | 新分配块数与块 id、请求累计占用块列表、空闲块前后对比；失败时提示将抢占 |
| `[SCHED]` | 每步调度结束 | 步号 `#N`、本步调度 token 数、请求当前块列表、空闲块守恒 |
| `[FREE]` | 请求结束/被抢占 | 归还的块 id 列表、空闲块恢复情况 |
| `[BLK_TABLE]` | 模型执行前 | 调度器发来的块列表写入 BlockTable 对应行；CPU → NPU 同步 |
| `[PREPARE]` | 每步输入准备 | 各请求本步 token 数、已计算 token 数、`seq_lens`、`attn_state` |
| `[SLOT]` | 每步计算映射时 | positions 与 slot_mapping 的具体数值 |
| `[SLOT][DERIVE]` | 每步计算映射后 | 逐 token 推导：`pos → 逻辑块idx → block_table查表 → block_id → slot`，显示前 3 个 token |
| `[ATTN_META][GROUP]` | 每步构建元数据 | 本 group 含多少层、block_table 形状、slot_mapping 形状、层间共享关系 |
| `[ATTN_META][ROW]` | 每步构建元数据 | 每行有效块数（`num_blocks_per_row`）+ 实际块列表 |
| `[ATTN_META][SLOT]` | 每步构建元数据 | attention kernel 实际拿到的 slot_mapping |
| `[KV_WRITE]` | 每层写入时 | 前 3 个 slot 分解（`slot=块id*128+offset→kv_cache[0][bid][off]`）、cache 形状、跨层共享说明 |

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

## 5. 核心数据关系详解

这是理解 KV Cache 最关键的部分。五层数据从上到下逐层细化：

### 5.1 块池（BlockPool）— 最外层

```text
块池: num_blocks 个物理块, 编号 0 ~ num_blocks-1, 所有请求共享
调度器负责分配/释放; 调度器在 [ALLOC]/[FREE] 日志里记录
```

### 5.2 每层 kv_cache tensor — block 的容器

```text
每层有独立的 kv_cache tensor, 形状 [2, num_blocks, block_size, num_kv_heads, head_size]
  第 0 维 = 2: K 和 V 各一份
  第 1 维 = num_blocks: 每个元素对应块池中的一个物理块
  第 2 维 = block_size: 每个块存 128 个 token 的 KV
关键: 不同层的 block_id=N 是不同内存, 但编号相同
     如 block_id=3 → layer0 的 kv_cache_0[3] 和 layer1 的 kv_cache_1[3]
```

对应日志：`[INIT][TENSOR↔LAYER]`、`[INIT][BLOCK↔TENSOR]`

### 5.3 BlockTable — 请求到块的映射

```text
BlockTable 形状 [max_num_reqs, max_num_blocks_per_req]
  每行 = 一个请求, 如行 0 是 req_A 的块列表
  行内容 = [block_id_0, block_id_1, ...], 如 [3, 7, 12] 表示 req_A 的 KV 在 block 3, 7, 12
  num_blocks_per_row[r] = 该行实际用了几个块 (其余位置填 0)
关键: 所有层共享同一份 BlockTable (同一请求在所有层用相同的块)
```

对应日志：`[BLK_TABLE] add_row/append_row`、`[ATTN_META][ROW]`

### 5.4 slot_mapping — token 到写入位置的映射

```text
slot_mapping 形状 [num_tokens], 每个元素是一个 slot
slot = block_id * block_size + 块内偏移
推导: pos → 逻辑块idx = pos // block_size → block_table[req, idx] = block_id → slot = block_id * block_size + (pos % block_size)
关键: 所有层共享同一份 slot_mapping, 每层按 slot 写入自己的 kv_cache[layer]
```

对应日志：`[SLOT]`、`[SLOT][DERIVE]`（逐 token 推导）、`[ATTN_META][SLOT]`

### 5.5 reshape_and_cache — 最终写入

```text
reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
  把 token t 的 K 写到 key_cache.view(-1)[slot_mapping[t]]
  把 token t 的 V 写到 value_cache.view(-1)[slot_mapping[t]]
等价于: key_cache[0][block_id][offset] = K_t, 其中 block_id = slot // block_size, offset = slot % block_size
```

对应日志：`[KV_WRITE]`（含 slot 分解）

### 5.6 完整关系图

```text
请求 req_A
  │
  ├── 调度器分配 → [ALLOC] 分到物理块 [3, 7, 12]
  │
  ├── BlockTable 行 0 写入 [3, 7, 12] → [BLK_TABLE]
  │     ↓ 所有层共用这张表
  │
  ├── 每层 kv_cache[layer] 形如 [2, num_blocks, 128, ...]
  │     ↓ block_id=3 在 layer0 是 kv_cache_0[3], 在 layer1 是 kv_cache_1[3]
  │
  ├── slot_mapping[t] = block_id * 128 + offset → [SLOT][DERIVE]
  │     ↓ 所有层共用这份 mapping
  │
  └── reshape_and_cache → [KV_WRITE]
        token t 的 K → kv_cache[layer][0][block_id][offset]
        token t 的 V → kv_cache[layer][1][block_id][offset]
```

## 6. 单请求生命周期走读

一个请求从进入到结束，日志按以下顺序出现：

```text
启动阶段（一次）
  [INIT] 块池初始化: num_blocks=5000, block_size=128
  [INIT][TENSOR↔LAYER] 每层 kv_cache 形状、层间关系
  [INIT][BLOCK↔TENSOR] block_id → kv_cache[layer][0][N][i] 映射
  [INIT][SUMMARY] 五条关系总结

请求进入（waiting -> running）
  [STEP] 调度步 #1 开始: running=0 waiting=1 空闲块=5000/5000
  [PREFIX] 命中块数=[0]                          # 第二次发相同 prompt 时 >0
  [BLOCK_POOL] 领取 4 个物理块: 块id=[3, 7, 12, 15]  # ceil(512/128)=4
  [ALLOC] 分配成功 → 该请求累计占用块数=[4] 块id=[3, 7, 12, 15]
  [SCHED] 步#1 调度结果: req=xxx_0 本步调度token=512 已计算token=0

prefill 执行（第一步）
  [BLK_TABLE] add_row(新请求首次登记): row=0 各group块列表=[3, 7, 12, 15]
  [BLK_TABLE] commit_block_table: CPU → NPU 同步
  [PREPARE] attn_state=PrefillNoCache seq_lens=[512]
  [SLOT] positions=[0, 1, 2, ..., 511] slot_mapping=[384, 385, ..., 2047]
  [SLOT][DERIVE] 逐 token 推导:
    token#0: pos=0 → 逻辑块idx=0 → block_table[0,0]=物理块id=3 → slot=3*128+0=384
    token#1: pos=1 → 逻辑块idx=0 → block_table[0,0]=物理块id=3 → slot=3*128+1=385
    token#128: pos=128 → 逻辑块idx=1 → block_table[0,1]=物理块id=7 → slot=7*128+0=896
  [ATTN_META][GROUP] group=0: 含 28 层, 共享 block_table 和 slot_mapping
  [ATTN_META][ROW] row=0: 有效块数=4 块列表=[3, 7, 12, 15]
  [ATTN_META][SLOT] slot_mapping=[384, 385, ..., 2047]
  [KV_WRITE] layer=0 写入KV: slot1920=块3*128+0→kv_cache[0][3][0] | ...
  [KV_WRITE] layer=1 写入KV: (同一份 slot_mapping, 不同层的 cache)
  ... (每层一条)

decode 阶段（每步重复）
  [STEP] 步#2: 本步调度token=1
  [SCHED] 步#2: 已计算token=512→513
  [SLOT][DERIVE] token#0: pos=512 → 逻辑块idx=4 → block_table[0,4]=物理块id=21 → slot=21*128+0=2688
  [KV_WRITE] layer=0: slot2688=块21*128+0→kv_cache[0][21][0]
  ...
  每写满 128 个 token 出现一次:
  [BLK_TABLE] append_row(追加新块) + [ALLOC] 新分配 1 块

请求结束（一次）
  [FREE] 归还块数=[N] | 空闲块 X → X+N
```

## 7. 必须看懂的三条关系

1. **slot 公式**：`slot = block_id * block_size + 块内偏移`。
   拿 `[SLOT][DERIVE]` 日志验证：`token#0: pos=0 → block_table[0,0]=物理块id=3 → slot=3*128+0=384`。
2. **块数守恒**：`[ALLOC]` 中"空闲块 100 → 99"，请求结束 `[FREE]` 中"99 → 100"，两边能对上。
3. **三方一致**：同一个块 id 列表依次出现在 `[ALLOC]`（调度器发放）、`[BLK_TABLE]`（写入映射表）、`[ATTN_META][ROW]`（kernel 使用）中，数字完全相同。

## 8. 常见问题

| 现象 | 原因与解决 |
| --- | --- |
| 完全没有 `[KV_DEBUG]` 日志 | 环境变量未在启动 server 之前 export；或日志重定向丢失了 stderr（记得 `2>&1`） |
| 没有 `KV_WRITE` | decode 走了 aclgraph 图回放，加 `--enforce-eager` |
| 没有 `SCHED` / `ALLOC` | 这些日志在 EngineCore 进程，确认该进程输出也被收集 |
| 两进程日志交错难读 | 按标签过滤，或结合日志中的进程标识区分 |
| 日志量太大 | 优先看 `[ALLOC]` / `[SLOT][DERIVE]` / `[FREE]`，并过滤掉 `KV_WRITE` |
| 看不懂数据关系 | 先读 `[INIT][SUMMARY]` 五条总结，再看 `[SLOT][DERIVE]` 逐 token 推导 |

## 9. 相关代码位置

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
