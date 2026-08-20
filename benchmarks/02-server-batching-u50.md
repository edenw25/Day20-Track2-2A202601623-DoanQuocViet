# 02 - Continuous batching under load (u50)

Host `Windows-AMD64` · `--parallel 4` · 14 samples over
60s at 2.0s intervals · raw CSV: `02-server-metrics-u50.csv`

| Gauge | Peak observed |
|:--|--:|
| `n_busy_slots_per_decode` (avg/decode) | 3.97 of 4 slots (99%) |
| `requests_processing` | 4 |
| `requests_deferred` | 45 |
| `kv_cache_usage_ratio` | n/a — not exported by llama.cpp `b10488` |
| `tokens_predicted_total` (final) | 32128 |

Highest sampled value was **3.97 of 4** slots. Note this gauge is llama.cpp's *average* busy slots per decode step, so the number below is the highest average we sampled, not an instantaneous maximum batch width. A peak near 1 means
requests were served one at a time -- either the load was too light to overlap, or
they arrived too far apart. A peak approaching `--parallel` means the scheduler was
genuinely packing concurrent requests into shared decode steps.
`requests_deferred` went above zero: more requests arrived than there were slots, so some waited. That wait is the queue time in your P95.

## Your observation

**Peak batch width: 3.97 of 4 slots (99%).** Continuous batching is working - the
scheduler packed essentially every decode step with all four slots for the whole 60 s
window. A peak near 1 would have meant serialization; 3.97 means shared decode steps.

The gauge held at 3.96-3.97 across all 14 samples with no dip, which matters: it is not
a lucky peak, it is a plateau. `requests_processing` was pinned at exactly 4 - the slot
count - for every sample, and `requests_deferred` sat at 42-45 throughout, only falling
to 18 in the final sample as locust stopped spawning and the backlog began to drain.

**Does it match the effective concurrency in `02-server-results.md`? No - and they are
not supposed to match.** That report gives 41.6 effective concurrency against 4 slots
(ratio 10.4). These two numbers disagree by ~10x because they count different things:

- `n_busy_slots_per_decode` = 3.97 counts requests **being decoded**. It is bounded above
  by `--parallel`, so it can never exceed 4 no matter how much load arrives.
- Little's Law concurrency = 41.6 counts requests **in the system**, which includes every
  request queued outside a slot.

So they are not in conflict; they are the two halves of one picture, and the difference
between them *is* the queue: 41.6 - 3.97 = **~37.6 requests waiting at any instant**.
That is independently corroborated by the server's own `requests_deferred` gauge, which
reported 42-45 - the same quantity, measured directly rather than inferred. Two
instruments agreeing on the queue depth to within about 10% is the strongest evidence in
this run.

**Which do I trust for which question?** For "is batching working" I trust
`n_busy_slots_per_decode`, because it is the server's own count of what it did per
decode step rather than something derived from client-side timing. For "is the server
saturated" I trust the Little's Law number plus `requests_deferred`, because slot
occupancy alone cannot distinguish a server that is comfortably full from one that is
drowning - both read 4 of 4. It is precisely the pair of them that tells the story:
**fully batched and badly oversubscribed at the same time.**

One caveat I am keeping in view: this gauge is llama.cpp's *average* busy slots per
`llama_decode()` call, so 3.97 is the highest average I sampled at a 2 s interval, not an
instantaneous maximum. Given it never dropped below 3.96, the distinction does not change
any conclusion here. Also `kv_cache_usage_ratio` is not exported by build `b10488`, so I
could not check how close the KV cache came to full - that would have told me whether
raising `--parallel` is limited by memory as well as by compute, and it is the one number
I am missing for the recommendation in `02-server-results.md`.
