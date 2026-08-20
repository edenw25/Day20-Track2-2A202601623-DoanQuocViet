# 02 - Serve: load test + saturation reading

Host `Windows-AMD64` · llama.cpp `b10488` ·
`--parallel 4` · `ctx=2048` · `threads=8` ·
`ngl=99`

| Users | Reqs | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Eff. concurrency | Failures |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 10 | 202 | 3.46 | 1900 | 3000 | 3800 | 6.9 | 0.0% |
| 50 | 194 | 3.25 | 14000 | 15000 | 16000 | 41.6 | 0.0% |

*Effective concurrency = RPS x average latency (Little's Law) -- how many requests were
really in flight, regardless of how many users locust simulated. It counts queued requests
too, so the occupancy/slot ratio can legitimately exceed 1.0; it is occupancy, not
utilisation. For true slot utilisation use the server's own gauges (`make metrics`).*

## What these two runs say

| Going from 10 to 50 users | |
|:--|--:|
| Offered load | 5x |
| Throughput actually delivered | **0.94x** (19% of linear) |
| P95 latency | **5.00x** |
| Effective concurrency at 50 users | 41.6 vs `--parallel 4` slots (occupancy/slot ratio 10.39) |

**Saturated.** Throughput delivered only 0.94x for 5x the offered load, and effective concurrency (41.6) is at or above all 4 decode slots. Saturation sets in somewhere at or below 50 users; the load you added beyond that point became queue time rather than throughput.

Throughput moved 0.94x while P95 moved 5.00x. That gap is the goodput argument: past saturation you buy throughput by spending latency, and if your SLO is a P95 target then the requests you added are no longer being served within it. (This lab does not fix an SLO number for you -- pick one in your write-up and state how much goodput you keep at it.)

## Your reading

**The server saturates below 10 users, and the number that settles it is that RPS went
*down*.**

Offered load went up 5x (10 -> 50 users) and delivered throughput went **0.94x**: 3.46
-> 3.25 RPS. Not "grew slowly" - it fell slightly. Meanwhile P95 went 3000 -> 15000 ms,
a clean **5.00x**, and P50 went 1900 -> 14000 ms (7.4x). Failures stayed at 0.0% in both
runs, so nothing was shed; every one of those extra requests was accepted and then made
to wait.

When throughput is flat, latency rising in near-exact proportion to offered load is the
signature of a queue, not of slower work. Little's Law makes it explicit: effective
concurrency is 3.25 x 12.78 s = **41.6 requests in the system against `--parallel 4`
slots**, an occupancy/slot ratio of 10.4. At any instant roughly 4 requests are decoding
and roughly 38 are waiting for a slot. So about **(41.6 - 4) / 41.6 = 90% of the time a
request spends in the system at 50 users is queue time**, and the residual ~1.4 s of
service time matches what the 10-user run measured end to end.

**How I know it is queue time and not compute time**, independently of Little's Law:
`make metrics`, sampled while `load-50` was running, shows
`n_busy_slots_per_decode = 3.97 of 4` and `requests_deferred` sitting at **42-45** for
the whole 60 s window (`02-server-batching-u50.md`). The server told me directly that
its 4 slots were ~99% packed and that ~43 more requests were parked outside them. Per-
slot decode speed had not degraded - there were simply only ever 4 slots. Two independent
instruments, same answer.

The 10-user run was already past the knee, not below it: 6.9 effective concurrency
against 4 slots means requests were queueing there too, which is why its P50 (1900 ms)
is already well above the ~500 ms a single unloaded request takes (`03-integration-
results.md`, and the `make bench` E2E P50 of 1052 ms at 64 output tokens). **True
saturation is somewhere near 4-6 concurrent requests** - i.e. at the slot count, exactly
where the model predicts.

**Pick an SLO and the picture gets blunt.** At a P95 <= 3 s target: the 10-user run just
meets it (P95 = 3000 ms) and delivers ~3.46 RPS of goodput. The 50-user run delivers
**0 RPS of goodput** - not one request in the 95th percentile met the target, and even
P50 (14 s) misses it by ~5x. Throughput barely changed; goodput@SLO went to zero. That
gap is the whole argument for not reporting peak throughput.

**What I would change first: `--parallel`, raised from 4 to 8-12.** Reasons, in order:

1. It targets the thing I actually measured as the constraint. Slots are pegged at 3.97
   of 4 with 43 requests deferred; every other knob leaves that queue in place.
2. I have headroom to spend. The 4060 shows ~7.1 GB free of 8187 MiB with the model
   loaded, and `n_ctx=2048` split across 4 slots is only 512 tokens per slot. More slots
   is mostly more KV cache, and I can afford it.
3. Decode is bandwidth-bound, so a wider batch is close to free per step: adding a
   sequence to a decode step reuses the same weight read. That is exactly why continuous
   batching raises throughput faster than it raises latency - and my `n_busy_slots` of
   3.97 shows the batcher is already doing this correctly, just against too small a
   ceiling.

**Why not the alternatives.** Raising `-t` does nothing (`01-tuning-tg128.md`: the
thread curve is flat at 1.05x spread, because decode is on the GPU). Switching to the
2-bit quantization buys 1.18x decode and does not change the slot count, so it moves P95
by far less than the 5x I need. Raising `--ctx-size` makes things *worse* at fixed
`--parallel` by consuming the VRAM I want to spend on slots.

**The honest caveat:** `--parallel 8` will not give 2x goodput. Beyond the point where
the GPU's decode step is itself saturated, more slots convert queue time into slower
per-token decode for everyone - the wait moves rather than disappears. I would sweep
`--parallel` at 1, 4, 8, 12 against a fixed P95 <= 3 s SLO and take the value that
maximises goodput, not the one that maximises RPS. I have not run that sweep, so I am
naming it as the next experiment rather than claiming its result.
