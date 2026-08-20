# 01 - Measure: latency baseline

Model `Gemma 4 E2B` · host `Windows-AMD64` · llama.cpp `b10488`
Settings: `threads=8` `ngl=99` `ctx=2048`
`max_tokens=64` · warm-up discarded
Completed requests: `UD-Q4_K_XL` 10/10 · `UD-Q2_K_XL` 10/10

| Quantization | Size (GB) | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode (tok/s) |
|:--|--:|--:|--:|--:|--:|--:|
| UD-Q4_K_XL | 2.97 | 5755 | 424 / 778 | 10.0 / 10.7 | 1052 / 1436 / 1436 | 100.3 |
| UD-Q2_K_XL | 2.24 | 4314 | 305 / 465 | 8.5 / 9.2 | 844 / 998 / 998 | 117.9 |

- **TTFT** = prefill. Short prompts keep it small; long-context RAG is where it explodes.
- **TPOT** = per-output-token decode cost, bounded by memory bandwidth. `decode tok/s = 1000 / TPOT_p50`.
- `UD-Q2_K_XL` decodes **1.18x faster** than `UD-Q4_K_XL` here, for 0.73 GB less on disk.

## Your observation

**Verdict: on this machine the 2-bit build is not worth it, and the reason is that the
thing it optimises is not the thing I am short of.**

The speed difference is real but modest. TPOT P50 goes 10.0 -> 8.5 ms, i.e. 100.3 ->
117.9 tok/s decode, a **1.18x** speedup, and the file is 0.73 GB smaller (2.97 -> 2.24
GB). TTFT improves more in relative terms (424 -> 305 ms P50, 1.39x) and the P95 tail
tightens noticeably (778 -> 465 ms).

Why the decode gain is only 1.18x when the weights shrank by 25%: decode here runs on
the RTX 4060 (`ngl=99`), and GDDR6 bandwidth is not the scarce resource for a model this
small. Fewer bits per weight buys speed in proportion to how bandwidth-bound you already
are, and on GPU I am much less bandwidth-bound than a CPU-only run would be. The same
25% size cut should pay off more on the CPU side - see `01-tuning-ngl.md`, where CPU
decode saturates its memory channels at 8 threads and then regresses.

**The size argument does not apply to me either.** Both quantizations fit in 8187 MiB of
VRAM with room to spare, so 2.24 GB vs 2.97 GB buys me nothing operationally: no extra
slots, no larger context, no avoided spill to system RAM. On an 8 GB laptop with no
discrete GPU that 0.73 GB could be the difference between fitting and swapping, and then
the answer flips. This is a machine-specific verdict, not a general one.

**Quality check - I served both and asked the same three questions.** `UD-Q4_K_XL` on
:8080, `UD-Q2_K_XL` on :8090, same prompts, `temperature=0.3`, `max_tokens=220`.

- On the queueing question ("4 decode slots, 50 users, P95 3s -> 15s, RPS flat") both
  answered correctly and both named queueing delay as the cause. No visible gap.
- On "explain continuous batching" the 4-bit answer is meaningfully more correct: it
  says a request "is placed into the next available processing slot" as soon as it
  arrives. The 2-bit answer drifts into generic pipelining - "overlapping the setup,
  execution, and teardown phases", "while one batch is being processed the system is
  preparing the next batch" - which describes double-buffering, not continuous batching.
  Right shape of words, wrong mechanism.
- On "TTFT vs TPOT" **both** refused, claiming the acronyms are not standard. That is a
  model-knowledge limit, not a quantization artifact, and I am separating it out rather
  than counting it against either build.

So the degradation I could actually observe is subtle: not broken output, but a slide
from a specific mechanism to a plausible-sounding generic one. That is the failure mode
hardest to catch in production, because it never looks like an error. Three prompts is
far too small a sample to quantify it - I am reporting a direction, not a score.

**Conclusion:** I keep `UD-Q4_K_XL`. I would be paying 1.18x decode throughput and
0.73 GB - neither of which is my binding constraint - to accept a quality regression I
can see but cannot bound. If I were RAM-limited rather than GPU-backed, I would take the
opposite trade, and then I would spend the effort to measure quality properly instead of
eyeballing three answers.
