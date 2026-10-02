# More models, and long context

The engine ([serving.md](serving.md)) was written for Llama-3-8B and first ran a real
checkpoint at 2K tokens. This note covers what it took to run two more checkpoints of the
same shape, Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, and to run Llama 3.1 at the
context it was trained for: needle retrieval up to 100K tokens, time to first token and decode
rate against context length, and perplexity at 64K, each next to Hugging Face transformers.

Every number is from the RTX 5090 (sm_120, torch 2.14, transformers 5.18, sdpa attention on
the transformers side), bf16 weights, greedy decoding. Rows: `results/llm_ppl.jsonl` and
`results/llm_models.json`. Scripts: `scripts/eval_ppl.py`, `scripts/needle.py`,
`scripts/generate.py`.

## What each model needed

**Llama-3-8B-Instruct** needed nothing new. It is the shape the kernels were built for.

**Llama-3.1-8B-Instruct** has the same weights layout and a `rope_scaling` block of type
`llama3`: factor 8, low and high frequency factors 1 and 4, original context 8192. The loader
refused it before. `layer.llama3_inv_freq` now applies the adjustment to the inverse
frequencies before the cos and sin tables are built: a frequency whose wavelength is shorter
than 8192 / 4 = 2048 positions stays, one whose wavelength is longer than 8192 positions is
divided by 8, and the band between is a linear blend. The kernels never see this. They read
the tables. `tests/test_hf.py` checks `inv_freq` against transformers' own
`ROPE_INIT_FUNCTIONS["llama3"]` to 1e-6 relative. Other rope types (yarn, dynamic, longrope)
are still refused.

**Mistral-7B-Instruct-v0.3** has the same hidden size, heads, K/V heads and MLP width, a
32768 vocabulary (a multiple of 64, so hgemm takes the lm_head as is), rope theta 1e6 and
rms_norm_eps 1e-5. Theta already came from the config, and eps is the engine's constant, so I
did not thread eps through the models. `sliding_window` is null in this release, so the
attention is plain causal. The tensor names are Llama's. The only fix was in `generate.py`:
it looked up `<|eot_id|>` as a stop token, which Mistral's tokenizer maps to the unknown id 0.
It now stops on the tokenizer's eos, the generation config's eos ids and `<|eot_id|>` only
when the vocabulary has it. The chat template and the BOS token come from the tokenizer and
work unchanged.

## Perplexity, short context

wikitext-2 test, the first 16 windows of 2048 tokens (32,752 scored tokens):

| model                    | spark  | transformers |
|--------------------------|-------:|-------------:|
| Llama-3-8B-Instruct      | 8.5298 | 8.5311       |
| Llama-3.1-8B-Instruct    | 7.4370 | 7.4386       |
| Mistral-7B-Instruct-v0.3 | 5.5235 | 5.5240       |

The engine is within 0.03% of transformers on all three, and slightly lower on all three. I
take that to be the fp32 RoPE tables and fp32 accumulation everywhere against transformers'
bf16 cos and sin. I did not check that.

## Long prompts: chunked prefill

The kernels handle 128K-token sequences as they are. Every offset in `attention_varlen`,
`paged_decode`, `rope_append_paged_` and the GEMM epilogues is computed in 64 bits, and
`tests/test_paged.py` now runs all three at 128K keys (decode at 131072 and 70001 keys, a
256-token prefill chunk at the end of a 131072-token sequence, rope at positions 65535 to
131071) against SDPA. They pass without changes.

What does not fit is a whole long prompt in one forward. The engine prefilled each prompt
in one packed batch, so a 100K-token prompt meant 100K rows of every activation: the
SwiGLU output alone is 2.9 GB, and all-position logits for a perplexity window of 32K tokens
would be 32K x 128256 bf16 = 8.4 GB. The engine now splits the prefill into forwards of at
most `prefill_tokens` tokens. A forward packs the next pieces of the admitted prompts in
order, a prompt that does not fit what is left of the budget is cut, and its next piece goes
in the next forward. A piece attends to the keys the earlier pieces put in the cache: that is
`attention_varlen`'s bottom-right causal mask, which was there for this case and needed no
change. `Engine.prompt_logits_chunks` gives the logits of each piece as it is computed, so a
perplexity never holds more than one chunk of logits. In a mixed step ([llm_serving.md](llm_serving.md)) the
running sequences' decode rows ride in the first forward of the admitted prompts' chunks
and the later chunks are prefill only, so each running sequence still gets one token per
step. `tests/test_engine.py` checks that a chunked, mixed run gives the same tokens as
whole-prompt prefills with separate decode steps.

Two smaller fixes came with it. The engine now refuses a `max_seq` longer than the RoPE
tables: the rope kernel reads the table at whatever position it is given, and `generate.py`
built 8192-row tables, so a longer prompt would have read past them without an error.
`TorchModel` raises on a chunked prefill instead of attending only to the chunk.

The chunk size costs little. Needle prompt of 32,769 tokens, Llama 3.1:

| chunk  | TTFT (s) |
|-------:|---------:|
| 2048   | 3.647    |
| 4096   | 3.410    |
| 8192   | 3.320    |
| 16384  | 3.257    |
| 32768  | 3.230    |

From 32K down to 8K-token forwards the prefill is 3% slower. At 2K it is 13% slower: the
GEMMs get short and each chunk re-reads all the earlier keys.

## The memory limit at 100K tokens

On this card 29.75 GiB is free at start (another tenant held 1.1 GiB during these runs).
The weights take 14.96 GiB, and 16.9 GiB are reserved once `hf.load` returns. A cache of
100K tokens is 12.2 GiB (32 layers x 8 K/V heads x 128 x 2 x 2 bytes = 128 KB a token).
That leaves about 1.3 GiB, and the first prefill failed with an out-of-memory error inside
hgemm.

The cause was not the GEMM workspaces. I measured `mem_get_info` around single calls: the
first launch of the TMA GEMM (variants 5 and 6, the default for prefill shapes) takes
3.4 GiB of device memory outside torch's allocator, at M = 1024 as at M = 8192. The
pre-TMA Stream-K kernel (variant 4) takes 45 MB, and the decode path (M <= 64) never
launches the TMA kernel. `cuobjdump` shows 8 bytes of stack and no local memory for the TMA
kernels, so it is not a local-memory reservation, and it is not module loading: with
`CUDA_MODULE_LOADING=EAGER` the first launch still takes 3.3 GiB. I did not find what
allocates it. It is a fixed cost, so it only matters when the cache is sized to the last
gigabyte.

The fix on the engine side: `SparkModel(prefill_variant=4)` runs the prefill GEMMs on
variant 4 and leaves decode on the default. With that, a 4096-token chunk and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, a 100,000-token prompt fits. It costs
time: at 64K the same prompt takes 9.77 s with the default GEMMs and 10.42 s on variant 4,
both at chunk 4096, against 9.32 s at the default chunk of 8192.

## Needle in a haystack

`scripts/needle.py` hides "The special magic number is N." (N a random seven-digit number)
at 10%, 50% and 90% of the text of War and Peace (Project Gutenberg), cut to the target
length, asks for the number through the chat template, and checks the greedy reply. Llama
3.1:

| prompt tokens | spark found | transformers found              |
|--------------:|:-----------:|:--------------------------------|
| 2,049         | 3/3         | 3/3                             |
| 8,193         | 3/3         | 3/3                             |
| 32,769        | 3/3         | 3/3                             |
| 65,537        | 3/3         | out of memory in `generate`     |
| 100,000       | 3/3         | out of memory in `generate`     |

Each spark reply is the number and nothing else, and where transformers ran, its replies are
the same strings. Transformers' `generate` keeps a dynamic cache plus the full-length MLP
activations of the prefill, which does not fit next to the weights at 64K. Its
forward without a cache does fit, which gives the TTFT rows below.

## TTFT and decode against context length

TTFT is the whole prefill of one prompt to its first token. For the engine that is the
chunked prefill (chunk 8192 up to 64K; chunk 4096 and variant 4 at 100K). For transformers
it is one forward with `use_cache=False` and `logits_to_keep=1`. Decode is 63 greedy tokens
at batch 1 after the prompt, the engine on its CUDA graph, transformers in `generate` with
its time to prefill subtracted. Mean of the three depths:

| prompt tokens | spark TTFT (s) | HF TTFT (s) | spark prefill tok/s | spark decode tok/s | HF decode tok/s |
|--------------:|---------------:|------------:|--------------------:|-------------------:|----------------:|
| 2,049         | 0.133          | 0.157       | 15,400              | 99.7               | 83.1            |
| 8,193         | 0.583          | 0.680       | 14,050              | 94.1               | 75.9            |
| 32,769        | 3.320          | 3.658       | 9,870               | 79.9               | 52.5            |
| 65,537        | 9.331          | 9.830       | 7,020               | 66.5               | out of memory   |
| 100,000       | 20.57          | 19.10       | 4,860               | 56.5               | out of memory   |

The engine's prefill is 10% to 15% faster up to 32K, 5% faster at 64K, and 8% slower at
100K, where it runs on the slower GEMM and smaller chunks to fit, and transformers runs one
forward with no cache to keep. Prefill throughput falls with length on both because the
attention is quadratic: at 100K the attention FLOPs are larger than the GEMM FLOPs.

Decode is bandwidth: every step reads the 16.1 GB of weights plus the cache. At 2K that is
99.7 tokens/s, 10.0 ms a step, against 10.5 ms to stream the weights at the 1,532 GB/s copy
rate (a read-only stream beats the copy rate by up to 10%, [attention.md](attention.md)). At
100K the cache adds 13.1 GB a step and the rate drops to 56.5 tokens/s, 17.7 ms: 29.2 GB in
17.7 ms is 1,650 GB/s, so the split in `paged_decode` keeps one 100K sequence at the same
bandwidth as the weights. Transformers' decode falls faster, from 83.1 to 52.5 tokens/s
between 2K and 32K. I did not profile it.

## Perplexity at long context

Llama 3.1, the first 131,072 tokens of each text cut into windows of the given length, so
every row scores the same tokens (one fewer per window):

| text          | ctx   | spark  | transformers |
|---------------|------:|-------:|-------------:|
| wikitext-2    | 2048  | 7.1164 | 7.1176       |
| wikitext-2    | 8192  | 6.3828 | 6.3841       |
| wikitext-2    | 32768 | 6.2371 | 6.2374       |
| wikitext-2    | 65536 | 6.2332 | 6.2322       |
| War and Peace | 2048  | 9.5210 | 9.5254       |
| War and Peace | 8192  | 8.8523 | 8.8539       |
| War and Peace | 32768 | 8.8034 | 8.8063       |
| War and Peace | 65536 | 8.7634 | 8.7671       |

The two agree to 0.05% at every length, so the scaled RoPE, the chunked prefill and the
kernels at 64K keys match transformers. The 64K spark rows use a 4096-token chunk: at 8192
the 2.1 GB chunk of logits did not fit next to an 8 GiB cache and the 3.4 GiB above.

Longer windows help, most of it by 8K. The effect of the rope scaling shows when I turn it
off (`eval_ppl.py --no-rope-scaling`) on the same 32K windows: wikitext-2 goes from 6.2371 to
8.3968 and War and Peace from 8.8034 to 10.542. Without the adjustment the model sees
rotation angles past position 8192 that it was not trained on.

## What I did not do

- Mistral at long context. Its config allows 32K and it should run as it is, but I only
  measured it at 2K.
- Find what the first TMA GEMM launch allocates. A fix there would let the default GEMMs run
  at 100K and close the 8% gap to transformers at that length.
- A fair transformers decode at 64K and 100K. It needs a static or offloaded cache.
