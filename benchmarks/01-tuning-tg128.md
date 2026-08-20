# 01 - Tune: thread-count sweep

Model `gemma-4-E2B-it-UD-Q4_K_XL.gguf` · host `Windows-AMD64` · llama.cpp `b10488`
CPU: **8 physical · 16 logical** cores · `ngl=99` · metric `tg128`

| threads (-t) | tg128 (tok/s) | vs best |
|:--|--:|--:|
| 1 | 106.2 | 95% |
| 4 | 106.5 | 95% |
| 8 | 111.6 | 100% |
| 16 | 112.0 | 100% |
| 32 | 110.6 | 99% |

**Best**: `-t 16` at 112.0 tok/s
**Slowest tested**: `-t 1` at 106.2 tok/s (1.05x spread)
**Against the physical-core default** (`-t 8`, 111.6 tok/s): 1.00x

Use this in your run:

```bash
LAB_N_THREADS=16 make bench
```

## Your explanation

**There is no knee. The curve is flat, and that is the finding.**

Best (`-t 16`, 112.0 tok/s) beats worst (`-t 1`, 106.2 tok/s) by 1.05x, and against the
physical-core default `-t 8` the "best" setting is 1.00x - inside run-to-run noise. The
deck predicts a rise to the physical core count and then a drop; I got neither the rise
nor the drop.

**Why:** this sweep ran with `ngl=99`. The lab sets that because `llama-server
--list-devices` really does enumerate the RTX 4060, so all of Gemma 4 E2B's layers are
resident in VRAM. During decode the CPU threads are not doing the matrix multiplies -
they enqueue CUDA kernels and run the sampler, and then wait. `-t` sizes a thread pool
that is no longer on the critical path. Setting it to 1 or to 32 cannot move a number
that is determined by how fast the 4060 streams its own weights out of GDDR6. The knob
is not merely weak here; it is **disconnected**.

Note `-t 32` (2x logical cores, deliberate oversubscription) costs only 1.2% versus the
peak. On a CPU-bound run that point is where you would expect the clearest damage. Its
near-absence is more evidence for the same conclusion: there is barely any CPU-side work
left to contend over.

**I checked this rather than assuming it.** `01-tuning-ngl.md` re-runs exactly this
sweep with `-ngl 0` so decode falls back to the Ryzen 7 8745H. That curve *does* have
the textbook shape - 5.88 tok/s at 1 thread, peaking at **22.98 at `-t 8`** (exactly the
physical core count), then **dropping to 19.26 at `-t 16`** as SMT siblings contend for
the same load/store units and DDR5 channels. Same binary, same model, same session; the
only change is where decode runs. So the deck's model of thread scaling is not wrong -
it is a model of *CPU* decode, and this sweep was not measuring CPU decode.

**What I would report to someone reusing this number:** `-t` is worth tuning on this
machine only if you are running `ngl=0` (no GPU, or offload disabled). With the GPU in
play, any value from 1 to 32 is fine and I left it at the physical-core default of 8.
The tuning effort belongs elsewhere - `--parallel`, context size, or quantization - as
the load tests in `02-server-results.md` show.
