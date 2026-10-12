# Unified memory: what a kernel gets from each kind, and the CPU-GPU handoff

Source: `src/kernels/unified.cu` (the memory kinds, the advice wrapper, the device caps and
the streaming read ladder), `src/kernels/unified_gemv.cu` (the decode GEMV over weights in
any kind of memory, and the signal kernels). Bench: `bench_unified`. Python:
`sk.unified_read_sum`, `sk.unified_gemv`, `sk.unified_device_caps`. Test:
`tests/test_unified.py`.

**Not measured.** Written on a machine without a GPU or nvcc, before the Spark arrived. The
kernels have not compiled, the bench has no rows, the test has not run. Every number below is
a byte count or a floor computed from a spec sheet; none is a time anyone observed. The
"what to run first" list at the end is the order to fill the table in.

## Why this matters on the Spark

Every other note in this repo treats memory as the GPU's own: a weight is `cudaMalloc`ed,
copied in once, and streamed by the kernel at whatever fraction of the copy roof the kernel
reaches. On the GB10 that model is incomplete in three ways.

1. **There is one pool.** The 128 GB of LPDDR5X is the CPU's memory and the GPU's memory,
   over a coherent fabric, and the spec's 273 GB/s is shared by both. A checkpoint `mmap`ed
   by the loader is already in the only memory there is; copying it into a `cudaMalloc`
   buffer doubles its footprint and moves bytes from one part of the pool to another. The
   question for a kernel writer is whether a kernel can read the mapped pages in place at
   the same GB/s it reads `cudaMalloc` pages, or whether the path through the fabric and the
   CPU's page tables costs something per byte or per page.
2. **The 1-bit DeepSeek V4.1 Flash experts are the use case.** [w1gemm](w1gemm.md) counts the
   model's routed experts at 1 bit plus scales as 76 GB for 40 layers, a set that does not
   fit a 32 GB card at any width and does fit the Spark's 128 GB with room for the rest of
   the model. A decode step touches 6 of the 384 experts per layer, 1.19 GB per token, which
   at 273 GB/s is a 4.4 ms floor, about 230 tokens/s from the expert bytes alone. If those
   76 GB can sit in system memory and be read by the GPU directly, the loader is a `mmap` and
   a quantize pass, and there is no copy anywhere in the decode loop. If they cannot, every
   layer's experts have to be `cudaMalloc`ed, and that is still fine on 128 GB, but it is a
   different engine.
3. **The CPU is close.** Twenty Arm cores share the pool with the GPU. A decode step has
   work that is awkward on the GPU and trivial on a core: the top-6 routing of the MoE gate
   over 384 scores, the sampling of one token from a few thousand logits after top-p, the
   scheduler's bookkeeping. Today each of those is either a kernel (`sample`, the routing
   sort in `moe.py`) or a `cudaMemcpy` and a stream sync. If a word the CPU writes reaches a
   running kernel in a few microseconds, the engine can route on a Grace core while the GPU
   streams the shared expert, and hand the expert indices over by writing them.

On the RTX 5090 the same calls are the same calls, over PCIe 5.0 x16, 64 GB/s each way on the
spec sheet against 1,792 GB/s of GDDR7. The bench builds and runs there, and the point of
running it there is the contrast: the 5090 column says what the API costs when the memory is
far, the GB10 column says what it costs when it is the same memory.

## The four kinds

`unified_alloc(kind, bytes)` returns a pointer the GPU can dereference for each:

| kind | call | on the 5090 | on the GB10 (expected, not measured) |
|---|---|---|---|
| `device` | `cudaMalloc` | GDDR7, the baseline | the pool, GPU page tables, the baseline |
| `managed` | `cudaMallocManaged` | pages migrate on first touch each way; advice and prefetch steer them | one pool, so "migration" is at most a page-table change; the question is whether the driver still does one |
| `host_mapped` | `cudaHostAlloc(cudaHostAllocMapped)` | pinned host pages read over PCIe every time, bounded by 64 GB/s | pinned pages read over the fabric, bounded by whatever the fabric path costs |
| `system` | `posix_memalign`, nothing else | readable in place only with HMM (open kernel modules, Linux 6.1+); otherwise `unified_alloc` falls back to `cudaHostRegister` and records `registered` | the GB10 is expected to report `pageableMemoryAccess` through ATS, so the GPU reads the CPU's pages through the CPU's page tables, no CUDA call at all |

`unified_device_caps()` reports the attributes that decide which of these work:
`pageableMemoryAccess` (and whether it uses the host's page tables),
`concurrentManagedAccess`, `directManagedMemAccessFromHost`, `hostNativeAtomicSupported`.
The first line the bench prints is these, and the first thing to look at on the Spark is
whether `pageableMemoryAccess` is 1; everything about the loader depends on it.

The `system` kind is the one that matters, because it is what every torch CPU tensor and
every `mmap`ed safetensors shard already is. The binding `sk.unified_gemv(a, w)` takes `w` as
a plain CPU tensor on a device with pageable access, so the test can ask the question from
Python with no allocator of its own.

## What each advice is expected to do

`unified_advise(ptr, bytes, advice, device, stream)` wraps `cudaMemAdvise` and
`cudaMemPrefetchAsync` (in CUDA 13's `cudaMemLocation` form) and returns false when the
runtime refuses, which it does for `device` and `host_mapped` memory and for plain pages on a
machine without HMM. The bench prints the refusal and moves on, since "the runtime does not
take advice on this memory" is one of the facts being collected.

| advice | call | what it should change |
|---|---|---|
| `none` | | on a discrete card the first GPU touch migrates the pages; the timed runs then read device-resident pages, and `first_ms` on stderr is the migration. On the GB10, nothing to migrate, so `first_ms` should be close to the steady state |
| `preferred_gpu` | `SetPreferredLocation(gpu)` | pages migrate to the GPU and stay there even when the CPU touches them. On the GB10 it may be a no-op or it may change which page tables map the range |
| `preferred_cpu` | `SetPreferredLocation(cpu)` | pages stay on the host; the GPU reads them remotely. On the 5090 this is the "read over PCIe" number for managed memory, to compare with `host_mapped`. On the GB10 it is the same pool |
| `accessed_by_gpu` | `SetAccessedBy(gpu)` | a mapping is kept for the GPU so a touch does not fault, without moving the pages. Used with `preferred_cpu` for the flag buffer so the spin does not thrash |
| `read_mostly` | `SetReadMostly` | read-duplicated: both sides keep a copy, a write invalidates. For weights that are only read, the GPU copy should behave like `device` after the first pass |
| `prefetch_gpu` | `cudaMemPrefetchAsync(gpu)` | the migration done up front on the stream; the timed runs should match `device`. The prefetch time itself is the loader's cost and is not in the rows |

The expectation to test on the Spark is that none of these change the steady-state GB/s,
because there is nowhere to migrate to. If `preferred_cpu` or `host_mapped` is slower than
`device` on the GB10, the fabric path costs something per byte and the loader should prefer
`cudaMalloc` plus a copy for anything read every token. If `system` matches `device`, the
76 GB expert set can be `mmap`ed.

## The read ladder

`unified_read_bf16` reduces the array into one fp32 partial per block, so the bytes are read
and the result can be checked against the CPU; the grid is `SMs x 8` blocks of 256 threads
for any `n`.

| variant | loads in flight per thread | what it tests |
|---|---|---|
| 0 | one 2-byte load | the LSU issue rate; the same baseline as `bandwidth_copy` variant 0 |
| 1 | one 16-byte load | the usual streaming kernel: one 128-bit load per lane per step |
| 2 | four 16-byte loads, four accumulators | the latency-hiding question |

Little's law is why variant 2 exists. Bytes in flight needed = bandwidth x latency. On the
5090, 1,532 GB/s of measured copy roof over 170 SMs: each SM needs to keep its share of
roughly 1.5 MB (at a 1 us round trip) in flight, and 170 SMs x 2048 threads x 16 bytes is
5.6 MB, so one load per thread is enough and variant 1 reaches the roof there
([bandwidth](bandwidth.md)). On the GB10, 273 GB/s over 48 SMs: 48 x 2048 x 16 bytes is
1.5 MB in flight with one load per thread, and at a 1 us latency the chip needs 273 KB, so
variant 1 should also be enough for `device` memory. The reason to have variant 2 is that
the latency to a far page (a `system` page through the CPU's page tables, a `host_mapped`
page across the fabric) is the unknown; if it is several times the local latency, or if the
GEMV's occupancy is half, variant 2 is where the difference shows. Which variant wins per
kind is the first row of the results table.

## The GEMV

`unified_gemv_bf16(A [K], W [N, K], y [N])` is the batch-1 decode projection with `W` in any
kind of memory: one warp per strip of rows, 16 bytes per lane per row per step, fp32
accumulation, bf16 out, checked against cuBLAS on a device copy of the same weight.

| variant | shape of the work | what it tests |
|---|---|---|
| 0 | one warp per row, one load per lane per step, grid `N / 4` | the baseline; at N = 2304 that is 576 blocks of 4 warps, 3.4 waves on 170 SMs and 12 on 48 |
| 1 | one warp per 4 rows, the next step's four loads in flight | 8 loads per lane in flight instead of 1 |
| 2 | variant 1's body on a persistent grid of `SMs x resident blocks` | the same kernel fills 48 and 170 SMs; on the GB10 a 2304-row expert and a 14336-row projection are the same launch |

Bytes per call and the floor at the spec bandwidth, for the shapes the bench runs:

| weight | bytes (bf16) | floor at 273 GB/s | floor at 1,532 GB/s (5090 copy roof) |
|---|---|---|---|
| 4096 x 4096 (Llama-3-8B q / o) | 33.6 MB | 123 us | 21.9 us |
| 14336 x 4096 (gate or up) | 117.4 MB | 430 us | 76.6 us |
| 4096 x 14336 (down) | 117.4 MB | 430 us | 76.6 us |
| 2304 x 5120 (DeepSeek V4.1 Flash expert gate or up) | 23.6 MB | 86 us | 15.4 us |

A Llama-3-8B decode step reads 32 layers x (2 x 33.6 + 2 x 8.4 + 3 x 117.4) MB = 14.0 GB
of bf16 weights; at 273 GB/s that is 51 ms, about 19 tokens/s, the number every memory kind
will be judged against on the Spark. The `host_mapped` row on the 5090 cannot beat 64 GB/s, so
that row's job is to show what a far read looks like, not to compete.

## The latency experiment

Three kernels, one thread each, with `cuda::atomic_ref<int, cuda::thread_scope_system>`
loads and stores:

- `unified_poll_flag`: spin until `*flag == expect`, write the spin count. Timed from the
  launch: launch latency plus the time to see a word the host wrote right after launching.
- `unified_set_flag`: write a word the host polls.
- `unified_ping_pong`: `rounds` of (wait `req == i`, store `ack = i`). Launched once; the
  host then does the trips: store `req = i` with release, spin on `ack == i` with acquire,
  `std::chrono` around each. Median of 1000 trips is the row, the min is next to it.

The flag lives in `host_mapped`, `system` or `managed` memory (the last only with
`concurrentManagedAccess`, and once more with `preferred_cpu + accessed_by_gpu` so the pages
do not migrate under the spin). Every spin is bounded (`kMaxSpins` on the GPU, 5 s per round
on the host), so a machine where the flag never becomes visible fails a row instead of
hanging the device. The baseline row is what a decode loop does today to read a sampled
token: a 4-byte `cudaMemcpyAsync` up, one down, and a stream sync.

On the 5090 a system-scope load from the GPU is a PCIe read of host memory per spin, and the
host's store becomes visible after a PCIe write; the round trip is a few of those. On the GB10
both are reads and writes to one pool through a coherent fabric. The repo does not guess the
numbers; the table has the cells.

**What a decode engine would do with a sub-10 us handoff.** A Llama-3-8B decode step on the
Spark is at least 51 ms of weight streaming at bf16 (above), a DeepSeek V4.1 Flash step at
1 bit at least 4.4 ms of expert bytes plus the dense parts. If the CPU can hand the GPU a
word in under 10 us, then per layer:

1. the GPU writes the router logits (384 bf16, one line) to a flag-guarded buffer and sets
   the flag, then starts the shared expert's GEMMs, which do not depend on the routing;
2. a Grace core, spinning on the flag, does the sigmoid, the biased top-6, the
   renormalization and the expert sort (the whole of `route` and `sort_by_expert` in
   `moe.py`, a few hundred operations on 384 numbers), writes the six indices and the
   offsets, and sets the answer flag;
3. the grouped `w1gemm_moe` launch, already queued behind the shared expert, reads the
   offsets the kernel's blocks were waiting on.

That replaces a `cudaMemcpy` of the logits, a sync, a torch sort and a copy back per layer,
or a chain of small kernels and a CUDA graph to absorb their launches, with two flag writes.
The same shape serves sampling: the GPU writes the top-p-filtered logits, a core samples and
writes the token, the next step's embedding lookup reads it. Whether 10 us is the right
threshold depends on the per-layer time it is hidden behind: 4.4 ms over 40 layers is 110 us
per layer at the floor, so a 10 us handoff is a tenth of a layer's expert stream, and the
routing work on the core has to finish inside the shared expert's time to cost nothing. All
of that is conditional on the measured round trip, and the measured round trip is the first
row to fill.

## Results

`bench_unified`, 1 GB buffer, median of 20 launches per row, 1000 round trips per signal row.
GB/s is the bytes read once over the time. Unmeasured on both machines.

### Streaming read, 1 GB bf16

| kind | advice | GB10 v0 GB/s | GB10 v1 | GB10 v2 | GB10 first touch ms | 5090 v0 | 5090 v1 | 5090 v2 | 5090 first touch ms |
|---|---|---|---|---|---|---|---|---|---|
| device | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | preferred_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | preferred_cpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | accessed_by_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | read_mostly | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | prefetch_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| host_mapped | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| system (hmm / ats) | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| system | preferred_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| system | read_mostly | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| system | prefetch_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### Decode GEMV, best variant, GB/s of the weight

| kind | advice | 4096x4096 GB10 | 14336x4096 GB10 | 4096x14336 GB10 | 2304x5120 GB10 | 4096x4096 5090 | 14336x4096 5090 | 4096x14336 5090 | 2304x5120 5090 |
|---|---|---|---|---|---|---|---|---|---|
| device | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | preferred_cpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| managed | prefetch_gpu | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| host_mapped | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| system | none | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| cuBLAS, device copy | | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

### CPU-GPU round trip, median of 1000 (min in parentheses)

| path | GB10 us | 5090 us |
|---|---|---|
| cudaMemcpyAsync 4 B up + 4 B down + sync | not measured | not measured |
| poll_flag, launch + one trip | not measured | not measured |
| ping_pong, host_mapped | not measured | not measured |
| ping_pong, system | not measured | not measured |
| ping_pong, managed, no advice | not measured | not measured |
| ping_pong, managed, preferred_cpu + accessed_by | not measured | not measured |

## What to run first

In this order, on the Spark, then the same on the 5090 for the other column:

1. `./build/bench_unified --kinds=device --skip-signal --mb=256` and read the caps line.
   `pageableMemoryAccess` decides whether the `system` kind is the real thing or the
   `cudaHostRegister` fallback (the `path` field of every row says which ran). The device rows
   must validate; they are the baseline.
2. `./build/bench_unified --kinds=system,managed --advice=none,prefetch_gpu`. If `system`
   with no advice matches `device` on the read and the GEMV, the 1-bit expert set can be
   `mmap`ed and read in place, and `deepseek_flash.py`'s loader should stop copying. If it is
   slower, `first_ms` and the `managed` rows say whether the gap is per page (migration, a
   one-time cost a prefetch removes) or per byte (the fabric path, which nothing removes).
3. `./build/bench_unified --skip-read --skip-gemv --trips=1000`, the signal rows. A round
   trip under 10 us makes the routing-on-a-core engine above worth writing; one over 50 us
   means the GPU-side routing stays.
4. The full default run for the table, then `pytest tests/test_unified.py`, which checks the
   GEMV and the read sum through the Python bindings on CUDA and pinned tensors, and on plain
   CPU tensors where the device reports pageable access.
5. `bench_bandwidth` next to it: the read rows on `device` memory should sit near the copy
   roof `bandwidth_copy` measures, and if they do not, the read ladder is the problem, not
   the memory kind.
