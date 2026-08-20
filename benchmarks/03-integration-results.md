# 03 - Integrate: RAG pipeline run

Host `Windows-AMD64` · llama.cpp `b10488` ·
retrieval backend: **keyword overlap** · 3 queries

| Query | Contexts retrieved | embed (ms) | retrieve (ms) | llm (ms) | total (ms) |
|:--|--:|--:|--:|--:|--:|
| Why is goodput more useful than raw throughp... | goodput, paged, radix | 0.0 | 0.0 | 823.5 | 823.5 |
| What problem does PagedAttention actually so... | paged, radix, disagg | 0.0 | 0.0 | 458.1 | 458.2 |
| When does splitting prefill and decode help?... | disagg, radix, batching | 0.0 | 0.0 | 466.6 | 466.6 |

Mean per stage (ms): embed **0.0** · retrieve **0.0** ·
llm **582.7** · total **582.8**
Dominant stage: **llm** (100% of total)

## Answers returned

**Why is goodput more useful than raw throughput?**

> Goodput@SLO counts only the requests per second that met the TTFT and TPOT targets. Throughput at saturation ignores SLOs.

**What problem does PagedAttention actually solve?**

> PagedAttention stores the KV cache in non-contiguous pages, which removes the internal fragmentation that wasted most GPU memory.

**When does splitting prefill and decode help?**

> Splitting prefill and decode helps because prefill is compute-bound and decode is memory-bandwidth-bound.


## Which N16-N19 pieces are real

Honest inventory of this run. Only the serving layer is real; everything upstream is the
shipped stub, and I have not wired in my own N16-N19 work.

| Day | Piece | Real or stub? | What actually ran |
|:--|:--|:--|:--|
| N16 Cloud/IaC | - | **stub** | No cluster, no Compose. `llama-server` bound to `127.0.0.1:8080` on my laptop. |
| N17 Data pipeline | - | **stub** | No DAG, no batch job. The corpus is the in-memory `TOY_DOCS` list literal in `pipeline.py`. |
| N18 Lakehouse | - | **stub** | No Delta/Iceberg, not even SQLite. Same 6-element Python list. |
| N19 Vector + features | - | **stub** | No vector index and **no embedding model**: `embed()` returned `None` because I ran without `--embed-url`, so `retrieve()` fell back to keyword overlap (set intersection on words longer than 3 chars). Retrieval backend reported as `keyword overlap`. |
| N20 Serving | `llama-server` | **real** | Gemma 4 E2B `UD-Q4_K_XL` on llama.cpp `b10488`, CUDA, `--parallel 4 --cont-batching --metrics`, reached over OpenAI-compatible HTTP. |

Retrieval quality was adequate for these three queries only because the toy corpus is
tiny and the query wording overlaps the documents almost verbatim - the top hit scored
1.0-2.0 while the rest scored 0.0. That is keyword matching getting lucky on a 6-document
set, not retrieval working. It would not survive a real corpus.

## Is the dominant stage what I expected?

**Yes in ranking, no in magnitude - and chasing the magnitude found a real bug in how I
was measuring.**

Expected: `llm` dominates. Confirmed, and overwhelmingly - `llm` is 582.7 ms of a 582.8
ms mean total, i.e. **100%**. `retrieve` is 0.0-0.1 ms and `embed` is exactly 0.0 ms
because no embedding model ran at all. With a stubbed retriever this ratio is close to
tautological: the only stage doing work is the one calling the GPU.

**The magnitude is where it got interesting.** My first two runs reported `llm` at
**2863 ms mean** - 4.9x higher than the 582.7 ms above - while the server's own `timings`
block in the same responses reported only ~85 ms prefill + ~340 ms decode. A ~2.4 s gap
per request between what the client measured and what the server admitted to. I first
assumed leftover queueing from the `load-50` run, but `/metrics` showed
`requests_processing = 0` and `requests_deferred = 0`, so the server was genuinely idle.

The cause is name resolution, not inference:

```
getaddrinfo('localhost', 8080)  ->  [('::1', 8080), ('127.0.0.1', 8080)]
raw socket connect to localhost   : 2050.5 ms
raw socket connect to 127.0.0.1   :    0.7 ms
```

`llama-server` is started with `--host 127.0.0.1` (see `labkit.server_cmd`), so it binds
**IPv4 only**. On this Windows box `localhost` resolves to `::1` first, and the connection
to `::1` is not refused instantly - it stalls for ~2 s of SYN retries before falling back
to IPv4. `pipeline.py` defaults to `--base-url http://localhost:8080` and calls
`httpx.post(...)` per query, which opens a **new connection every request**, so it paid
that ~2.4 s toll three times over.

Confirmed by holding one connection open - only the first request pays:

```
reused httpx.Client to localhost:  2516.7 ms -> 288.7 ms -> 277.9 ms
```

The numbers in the table above were therefore produced with
`--base-url http://127.0.0.1:8080`, which is the honest measurement of the pipeline
itself.

**Scope of the bug - which of this repo's other numbers are affected: none.**
`make bench` is clean because `labkit.serve_bg()` yields `http://127.0.0.1:{port}`, not
`localhost` - its TTFT P50 of 424 ms could not have hidden a 2 s stall anyway. The locust
runs are clean because geventhttpclient keeps connections alive per simulated user, so the
toll is paid once at spawn and amortised across a 60 s run. `make smoke` uses `localhost`
but only asserts non-zero counters, making no timing claim. So the defect was confined to
exactly the stage whose timing I was trying to attribute - which is the sort of thing that
would have led me to "optimise" the LLM stage when 80% of what I was measuring was a
TCP timeout.

## If I had to halve this pipeline's latency

I would **not** start with the LLM even though it is 100% of the total, because "100% of
582 ms" hides the fact that the pipeline is barely doing anything. Ranked:

1. **Prompt caching, which is already paying and should be protected.** Query 1 prefills
   149 tokens in 256 ms; queries 2 and 3 prefill **5 tokens in 31-33 ms** because the
   byte-identical system prompt and overlapping context let the server reuse the cached
   prefix. That is a ~7x cut in prefill on the repeat queries, for free. The way to lose
   it is to make the system prompt dynamic - injecting a timestamp or reordering retrieved
   chunks would invalidate the prefix on every call.
2. **Cut output tokens.** Decode is 218-284 ms of the ~460 ms warm request at only 24-30
   tokens - the dominant term once prefill is cached. `max_tokens` is 200 here; these
   answers used ~25. Tighter instructions and a lower cap move latency more than any
   serving knob at this scale.
3. **Only then the serving layer** - and per `02-server-results.md`, under concurrency
   the win there is `--parallel`, not per-request speed.

The stage-split as measured is not actionable guidance for a real RAG system, and I want
to be explicit about that: with a real N19 vector index and a real embedding model, `embed`
and `retrieve` would stop being 0.0 ms, and prefill would grow with genuinely retrieved
context rather than shrinking to 5 cached tokens. The honest conclusion from this run is
"my retrieval is free because it does nothing", not "retrieval is cheap".
