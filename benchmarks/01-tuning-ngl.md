# 01 - Tune (supplementary): what the thread sweep could not show

Model `gemma-4-E2B-it-UD-Q4_K_XL.gguf` · host `Windows-AMD64` · llama.cpp `b10488`
CPU: **8 physical · 16 logical** (AMD Ryzen 7 8745H) · GPU: NVIDIA RTX 4060 Laptop, 8187 MiB

`make tune` sweeps `-t` at whatever `ngl` the lab picked, and on this machine the lab
picks `ngl=99` because the CUDA runtime really does enumerate the 4060. The resulting
curve (`01-tuning-tg128.md`) is flat — 1.05x between the best and worst thread count.
That is a true result, but it is flat for a reason the sweep itself cannot show: the
thread count was never the binding constraint.

So I ran the same sweep twice, once with decode on the CPU and once with it on the GPU.
Both tables are raw `llama-bench` output, same binary, same model, same session:

```
runtime\b10488\llama-bench.exe -m models\gemma-4-E2B-it-UD-Q4_K_XL.gguf \
    -t 1,4,8,16 -ngl <0|99> -p 0 -n 128 -r 2 -o md
```

## `-ngl 0` — decode on the Ryzen 7 8745H

| threads (-t) | tg128 (tok/s) | vs best |
|:--|--:|--:|
| 1 | 5.88 ± 0.07 | 26% |
| 4 | 21.28 ± 4.02 | 93% |
| 8 | **22.98 ± 2.21** | **100%**  ← knee, = physical cores |
| 16 | 19.26 ± 1.30 | 84%  ← SMT siblings, slower |

## `-ngl 99` — decode on the RTX 4060

| threads (-t) | tg128 (tok/s) | vs best |
|:--|--:|--:|
| 1 | 110.23 ± 1.18 | 99% |
| 4 | **111.78 ± 0.63** | **100%** |
| 8 | 111.41 ± 0.45 | 100% |
| 16 | 111.55 ± 0.81 | 100% |

## The before/after

At a fixed `-t 8` (the physical-core default the lab would otherwise use):

```
before (-ngl 0,  CPU decode):   22.98 tok/s
after  (-ngl 99, GPU decode):  111.41 tok/s
speedup:                         4.85x
```

## My reading

The two tables are the same experiment with one variable changed, and they disagree
about whether thread count matters — which is the point.

**On CPU the curve has the shape the deck predicts, and the knee is exactly at 8.**
Going 1 -> 4 threads buys 3.6x, because at one thread decode is starved of issue width
and nowhere near the memory system's limit. 4 -> 8 buys only 8% more. 8 -> 16 *loses*
16%. The 16-thread point is the informative one: those extra 8 threads are not new
cores, they are SMT siblings sharing one physical core's load/store units and L2 slice
with the thread already there. Decode reads the weight matrices once per token and does
almost no arithmetic per byte read, so once 8 cores are issuing loads the DDR5 channels
are already the constraint; adding 8 more requesters only adds contention and cache
thrash. That is the classic memory-bandwidth-bound signature: throughput saturates at
the point where you run out of *channels*, not out of *cores*, and then regresses.

Note also the error bars: ±4.02 at `-t 4` and ±2.21 at `-t 8` versus ±0.07 at `-t 1`.
The run-to-run spread grows exactly where threads start contending — the variance is
itself evidence of contention rather than noise I should average away.

**On GPU the curve is flat within noise (110.2-111.8, ±1.2 at worst), including at
`-t 1`.** This is not the GPU being "fast enough that threads don't matter" — it is
that with all layers resident in VRAM, the CPU threads have almost no decode work left
to do. They launch kernels and run the sampler; the matrix multiplies and the weight
traffic happen entirely on the 4060 against its own GDDR6. `-t` sizes a thread pool that
is no longer on the critical path, so setting it to 1 or to 16 changes nothing
measurable. The knob did not get better — it got *disconnected*.

I deliberately do not convert these into a GB/s figure. Gemma 4 **E2B** is a
MatFormer-style model: the file is 2.95 GiB and 4.65 B parameters, but only a subset is
activated per token, so `tok/s x file size` would overstate the bytes actually moved and
give a bandwidth number that is not real. The thread curve is the stronger evidence
anyway, and it does not depend on knowing the activated-parameter count: a workload that
peaks at the physical core count and then *loses* throughput to SMT is bandwidth-bound
regardless of how many bytes per token it turns out to be.

**What this changes about how I read the rest of the lab.** Every base-track number in
this repo — the `make bench` table, the load tests, the pipeline timings — was produced
with `ngl=99`. They are GPU-serving numbers, not laptop-CPU numbers, and the 4.85x above
is the size of that difference. A classmate on a CPU-only machine is not measuring a
slower version of my setup; they are measuring the left-hand table, where `-t` is the
knob that matters and mine is the one that does not.
