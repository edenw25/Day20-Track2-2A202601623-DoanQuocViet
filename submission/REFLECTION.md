# Reflection — Day 20 Lab (Personal Report)

> **Đây là báo cáo cá nhân.** Số liệu của tôi **không** so sánh được với bạn cùng lớp
> — chỉ so **before vs after trên chính máy tôi**.

**Họ Tên:** Đoàn Quốc Việt (2A202601623)
**Cohort:** K3
**Ngày submit:** 2026-08-20

---

## 1. Hardware & runtime  *(rubric 1, 2 — 10 điểm)*

- **OS:** Windows 11 Home 10.0.26200 (Python `platform` báo "Windows 10", xem `hardware.json`)
- **CPU:** AMD Ryzen 7 8745H with Radeon 780M Graphics
- **Cores:** 8 physical / 16 logical
- **CPU extensions:** AVX2 (llama.cpp tự chọn backend `ggml-cpu-zen4.dll` cho Zen 4)
- **RAM:** 15.3 GB
- **Accelerator:** NVIDIA GeForce RTX 4060 Laptop GPU, 8187 MiB — **CUDA, offload ACTIVE**
- **llama.cpp asset đã tải:** `llama-b10488-bin-win-cuda-12.4-x64.zip` (+ `cudart-llama-bin-win-cuda-12.4-x64.zip`)
- **Model đã dùng:** Gemma 4 E2B (`LAB_MODEL=gemma4-e2b`)
- **Quantization:** UD-Q4_K_XL (primary) + UD-Q2_K_XL (compare) (từ `models/active.json`)

**Chạy ở đâu:** laptop của tôi. Không dùng Colab/Kaggle. Đủ RAM (15.3 GB > 8 GB) nên chạy
model mặc định, không phải fallback.

**Setup story:**

Setup chạy đúng như tài liệu: `.\lab.ps1 probe` → venv → `setup.py` tải runtime CUDA
(1.1 GB DLL) + 5.2 GB weights, không compile gì. Hai điều đáng ghi lại:

**(1) Lab này chạy trên GPU, không phải CPU.** `probe` báo `GPU offload : ACTIVE`, nên
`labkit.n_gpu_layers()` trả về 99 và **mọi số liệu base track trong repo này là số
GPU-serving**. Điều đó thay đổi hẳn cách đọc §5 — xem ở đó.

**(2) `make verify` không thể exit 0 trên Windows nếu không sửa — đây là bug có sẵn của
lab, không phải lỗi máy tôi.** `scripts/verify.py` đọc file bằng `path.read_text()` không
truyền encoding, nên trên Windows nó dùng locale mặc định (`cp1252`). Nhưng
`submission/REFLECTION.md`, `GUIDE.md`, `rubric.md` nguyên bản trong git đều là UTF-8 có
tiếng Việt, chứa các byte `0x81/0x8d/0x8f/0x90/0x9d` — vốn **không được định nghĩa** trong
cp1252. Kết quả là verify crash ngay trên repo chưa sửa gì:

```
File "scripts/verify.py", line 187, in check_reflection
    text = path.read_text()
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 54
```

Tôi sửa tối thiểu 4 chỗ `read_text()` thành `read_text(encoding="utf-8", errors="replace")`.
Dùng `errors="replace"` chứ không phải UTF-8 thuần là có chủ ý: các file `benchmarks/*.md`
do chính lab sinh ra lại được ghi bằng cp1252 (ký tự `·`), nên đọc UTF-8 nghiêm ngặt sẽ
hỏng theo chiều ngược lại. `errors="replace"` chịu được cả hai, và các chuỗi verify đi tìm
(`"required -- replace this line"`, các placeholder) đều là ASCII nên không bị ảnh hưởng.
Tôi cũng convert các `benchmarks/*.md` sang UTF-8 để GitHub render đúng.

Sửa xong lỗi encoding thì lộ ra **lỗi thứ hai, cũng chỉ xảy ra trên Windows**: verify báo
mọi file trong thư mục con là "NOT committed" dù đã commit, trong khi `hardware.json` ở
thư mục gốc lại pass. Nguyên nhân là `is_committed()` so
`str(path.relative_to(repo_root()))` — trên Windows cho ra `modelsctive.json` — với tập
hợp lấy từ `git ls-files`, vốn **luôn dùng dấu gạch chéo xuôi** (`models/active.json`).
File ở thư mục gốc không có dấu phân cách nên vô tình khớp; mọi file khác thì không. Tôi
đổi sang `.as_posix()`. Sau hai sửa đổi này `make verify` exit 0.

---

## 2. Đo lường  *(rubric 3, 4, 5 — 20 điểm)*

> Từ `benchmarks/01-quickstart-results.md` (`make bench`). 10 request/quantization,
> warm-up đã bỏ, `max_tokens=64`, `threads=8`, `ngl=99`, `ctx=2048`.

| Quantization | Size (GB) | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode (tok/s) |
|---|--:|--:|--:|--:|--:|--:|
| UD-Q4_K_XL | 2.97 | 5755 | 424 / 778 | 10.0 / 10.7 | 1052 / 1436 / 1436 | 100.3 |
| UD-Q2_K_XL | 2.24 | 4314 | 305 / 465 | 8.5 / 9.2 | 844 / 998 / 998 | 117.9 |

**TTFT và TPOT được báo riêng**, không gộp thành end-to-end: TTFT là prefill — thời gian
tới token đầu tiên; TPOT là chi phí mỗi token sau đó, `decode tok/s = 1000 / TPOT_p50`.

**Quan sát:** 2-bit nhanh hơn **1.18×** decode (100.3 → 117.9 tok/s) và nhẹ hơn 0.73 GB.
**Nhưng trên máy này nó không đáng**, vì cả hai thứ nó tối ưu đều không phải thứ tôi đang
thiếu. Decode chạy trên GPU nên tôi ít bị chặn bởi bandwidth hơn nhiều so với chạy CPU —
đó là lý do cắt 25% số bit chỉ đổi được 1.18× chứ không hơn. Về dung lượng, cả hai đều nằm
gọn trong 8187 MiB VRAM, nên 2.24 GB so với 2.97 GB **không mua cho tôi thêm slot, thêm
context, hay tránh được lần swap nào**.

Tôi đã thử chất lượng thật: serve 4-bit ở :8080, 2-bit ở :8090, hỏi **cùng 3 câu**,
`temperature=0.3`, `max_tokens=220`. Câu về queueing (4 slot / 50 user) thì **cả hai trả
lời đúng như nhau**. Câu "giải thích continuous batching" thì 4-bit đúng cơ chế ("request
được đưa vào slot còn trống ngay khi nó đến"), còn 2-bit trôi sang mô tả pipelining chung
chung ("overlap giai đoạn setup / execution / teardown") — **đúng dạng chữ, sai cơ chế**.
Câu TTFT vs TPOT thì **cả hai đều từ chối**, nói đây không phải thuật ngữ chuẩn — đó là
giới hạn kiến thức của model chứ không phải do quantization, nên tôi tách nó ra.

Kiểu suy giảm tôi quan sát được là loại tinh vi nhất: không phải output hỏng, mà là trượt
từ cơ chế cụ thể sang thứ nghe hợp lý nhưng chung chung. 3 prompt là mẫu quá nhỏ để định
lượng — tôi báo **chiều hướng, không phải điểm số**. Kết luận: giữ UD-Q4_K_XL. Nếu tôi bị
giới hạn RAM thay vì có GPU thì tôi sẽ chọn ngược lại.

---

## 3. Serving under load  *(rubric 8, 9, 10 — 20 điểm)*

> Từ `benchmarks/02-server-results.md` (`make load-report`). `--parallel 4`,
> `--cont-batching`, `ctx=2048` (512/slot), 60 s mỗi lần.

| Users | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Eff. concurrency | Failures |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 3.46 | 1900 | 3000 | 3800 | 6.9 | 0 (0.0%) |
| 50 | 3.25 | 14000 | 15000 | 16000 | 41.6 | 0 (0.0%) |

- **Offered load tăng 5×, throughput thực tăng:** **0.94×** (tức là **giảm** nhẹ)
- **P95 tăng:** **5.00×**
- **Effective concurrency ở 50 users:** **41.6** so với `--parallel` = **4** slots (tỉ lệ 10.4)

**Peak `llamacpp:n_busy_slots_per_decode`** (từ `make metrics` chạy **chồng thời gian** với
`make load-50`): **3.97** / **4** slots (99%), giữ 3.96–3.97 suốt cả 14 mẫu.

**Saturation reading:** Server bão hoà **dưới 10 users**, và con số quyết định là **RPS đi
xuống** chứ không phải đi ngang: 3.46 → 3.25 khi tải tăng 5×. Không request nào fail (0.0%
cả hai lần) — nghĩa là toàn bộ tải thêm được **nhận vào rồi bắt xếp hàng**, chứ không bị
từ chối.

Phần latency tăng thêm là **queue time, không phải compute time**, và tôi biết bằng **hai
dụng cụ độc lập**. Thứ nhất, Little's Law: 41.6 request nằm trong hệ thống trên 4 slot →
tại mọi thời điểm ~4 cái đang decode và ~38 cái đang chờ, tức **~90% thời gian một request
sống trong hệ thống là thời gian chờ**. Thứ hai, chính server tự khai: `requests_deferred`
đứng ở **42–45** suốt 60 s và `requests_processing` bị ghim đúng ở 4. Hai nguồn khớp nhau
trong khoảng 10%. Tốc độ decode mỗi slot không hề tệ đi — chỉ là **chưa bao giờ có quá 4
slot**.

Lưu ý `n_busy_slots = 3.97` và `effective concurrency = 41.6` **không mâu thuẫn**: cái đầu
đếm request **đang được decode** (bị chặn trên bởi `--parallel`), cái sau đếm request
**trong hệ thống** (gồm cả hàng đợi). Hiệu của chúng chính là độ sâu hàng đợi.

Nếu đặt SLO **P95 ≤ 3 s**: lần 10 users vừa đúng chạm ngưỡng (P95 = 3000 ms) → goodput
~3.46 RPS. Lần 50 users → **goodput = 0 RPS**, vì đến cả P50 (14 s) cũng trượt ngưỡng ~5×.
Throughput gần như không đổi nhưng goodput@SLO về 0. Đó là toàn bộ lý do không nên báo cáo
peak throughput.

**Knob tôi đổi trước tiên: `--parallel` 4 → 8–12.** Vì (a) nó nhắm đúng thứ tôi đo được là
nút thắt — slot ghim 3.97/4 với 43 request bị hoãn, mọi knob khác đều để nguyên hàng đợi
đó; (b) tôi còn dư tài nguyên: ~7.1 GB VRAM trống, và mỗi slot mới chủ yếu tốn thêm KV
cache; (c) decode bị chặn bởi bandwidth, nên thêm một sequence vào cùng một decode step
gần như miễn phí — nó **dùng lại chính lần đọc trọng số đó**. Không chọn `-t` vì thread
curve phẳng (§5). Không chọn 2-bit vì nó chỉ được 1.18× và không đổi số slot. Không tăng
`--ctx-size` vì ở `--parallel` cố định nó **ăn mất** đúng phần VRAM tôi muốn dành cho slot.

**Cảnh báo trung thực:** `--parallel 8` sẽ **không** cho gấp đôi goodput. Quá điểm decode
step của GPU bão hoà, thêm slot chỉ biến queue time thành decode chậm hơn cho tất cả — chỗ
chờ dịch đi chứ không biến mất. Cách đúng là sweep `--parallel` ở 1/4/8/12 với SLO
P95 ≤ 3 s cố định rồi lấy giá trị **tối đa hoá goodput**, không phải tối đa hoá RPS. Tôi
chưa chạy sweep đó nên tôi nêu nó là thí nghiệm tiếp theo, không tuyên bố kết quả.

---

## 4. Integration  *(rubric 12, 13 — 15 điểm)*

> Từ `make pipeline` (3 query chạy hết, có in context retrieve được).
> Chi tiết: `benchmarks/03-integration-results.md`.

| Day | Piece | Real hay stub? |
|---|---|---|
| N16 Cloud/IaC | không cluster / Compose — `llama-server` bind `127.0.0.1:8080` trên laptop | **stub** |
| N17 Data pipeline | không DAG / batch job — corpus là list `TOY_DOCS` in-memory | **stub** |
| N18 Lakehouse | không Delta/Iceberg, không cả SQLite — vẫn là list 6 phần tử đó | **stub** |
| N19 Vector + features | không vector index và **không embedding model**; `embed()` trả `None` nên `retrieve()` rơi về keyword overlap | **stub** |
| N20 Serving | `llama-server` Gemma 4 E2B UD-Q4_K_XL, CUDA, OpenAI-compat HTTP | **real** |

**Latency split** (mean của 3 query, đo với `--base-url http://127.0.0.1:8080`):

- embed: **0.0 ms**
- retrieve: **0.0 ms** (0.0–0.1)
- llm: **582.7 ms**
- **stage chiếm nhiều nhất:** **llm** (**100%** của total 582.8 ms)

**Reflection:** Thứ hạng đúng như kỳ vọng (llm áp đảo), nhưng **độ lớn thì sai, và đuổi
theo cái sai đó lôi ra một bug thật trong cách tôi đo**.

Hai lần chạy đầu báo `llm` = **2863 ms**, gấp 4.9× con số ở trên, trong khi `timings` của
chính server trong cùng response chỉ báo ~85 ms prefill + ~340 ms decode. Chênh ~2.4 s mỗi
request. Tôi đoán đầu tiên là hàng đợi còn sót từ `load-50`, nhưng `/metrics` cho thấy
`requests_processing = 0` và `requests_deferred = 0` — server rảnh thật.

Nguyên nhân là **phân giải tên, không phải inference**:

```
getaddrinfo('localhost', 8080)  ->  [('::1', 8080), ('127.0.0.1', 8080)]
raw socket connect to localhost   : 2050.5 ms
raw socket connect to 127.0.0.1   :    0.7 ms
```

`llama-server` được khởi động với `--host 127.0.0.1` (xem `labkit.server_cmd`) nên chỉ
bind **IPv4**. Trên máy Windows này `localhost` phân giải ra `::1` **trước**, và kết nối
tới `::1` không bị từ chối ngay mà **treo ~2 s retry SYN** rồi mới fallback sang IPv4.
`pipeline.py` mặc định `--base-url http://localhost:8080` và gọi `httpx.post(...)` cho mỗi
query — tức **mở connection mới mỗi request** — nên nó trả khoản phí đó ba lần.

Kiểm chứng bằng cách giữ một connection: chỉ request đầu phải trả —
`2516.7 ms → 288.7 ms → 277.9 ms`.

**Phạm vi ảnh hưởng — các số khác trong repo: không cái nào dính.** `make bench` sạch vì
`labkit.serve_bg()` trả về `http://127.0.0.1:{port}`. Load test sạch vì geventhttpclient
giữ keep-alive theo từng user nên phí chỉ trả 1 lần lúc spawn rồi phân bổ đều trên 60 s.
`make smoke` dùng `localhost` nhưng không tuyên bố gì về thời gian. Nói cách khác, lỗi rơi
đúng vào **cái stage duy nhất mà tôi đang cố quy trách nhiệm** — loại bug sẽ khiến tôi đi
"tối ưu LLM" trong khi 80% thứ tôi đo là một TCP timeout.

**Nếu phải giảm latency pipeline này 2×:** tôi **không** bắt đầu từ LLM dù nó chiếm 100%.
(1) **Bảo vệ prompt caching đang có sẵn** — query 1 prefill 149 token/256 ms, query 2 và 3
chỉ prefill **5 token/31–33 ms** vì system prompt giống hệt từng byte nên server tái dùng
cached prefix; cách nhanh nhất để mất nó là nhét timestamp vào system prompt. (2) **Cắt số
token output** — decode là 218–284 ms của request ~460 ms khi đã ấm, với chỉ 24–30 token,
trong khi `max_tokens` để tận 200. (3) Sau đó mới tới serving layer, và lúc đó là
`--parallel` chứ không phải tốc độ mỗi request.

Tôi nói rõ giới hạn: stage-split này **không phải hướng dẫn dùng được cho RAG thật**. Với
N19 thật, `embed` và `retrieve` sẽ thôi bằng 0.0 ms, và prefill sẽ **phình theo context
thật** thay vì co lại còn 5 token cached. Kết luận trung thực của lần chạy này là
"retrieval của tôi miễn phí vì nó không làm gì", chứ không phải "retrieval rẻ".

---

## 5. The single change that mattered most  *(rubric 11 — 10 điểm)*

**Change:** chuyển decode từ CPU sang GPU — `-ngl 0` → `-ngl 99` (giữ nguyên `-t 8`, cùng
binary, cùng model, cùng phiên đo). Bằng chứng: `benchmarks/01-tuning-ngl.md`.

```
before:  22.98 tok/s   (llama-bench tg128, -ngl 0,  -t 8)
after:  111.41 tok/s   (llama-bench tg128, -ngl 99, -t 8)
speedup: 4.85x
```

**Tại sao tôi báo con số này chứ không phải kết quả `make tune`:** vì `make tune` trên máy
tôi cho **1.00×**, và cái *tại sao nó bằng 1.00×* mới là nội dung đáng viết.

`make tune` sweep `-t` ở 1/4/8/16/32 và ra một đường **phẳng**: 106.2 → 112.0 tok/s, spread
1.05×, so với default `-t 8` thì "best" chỉ 1.00× — nằm trong nhiễu. Deck dự đoán throughput
leo tới số core vật lý rồi tụt. Tôi **không có cả phần leo lẫn phần tụt**.

Lý do là sweep đó chạy với `ngl=99`. Toàn bộ layer của Gemma 4 E2B nằm trong VRAM, nên khi
decode các CPU thread **không làm phép nhân ma trận** — chúng enqueue kernel CUDA, chạy
sampler, rồi đợi. `-t` đang chỉnh kích thước một thread pool **không còn nằm trên critical
path**. Đặt nó bằng 1 hay 32 đều không dịch nổi một con số vốn do tốc độ 4060 kéo trọng số
ra khỏi GDDR6 quyết định. Knob không "yếu" — nó bị **ngắt kết nối**.

Nên tôi chạy lại đúng sweep đó với `-ngl 0` để decode rơi về Ryzen 7 8745H. Lúc đó đường
cong **hiện đúng hình dạng sách giáo khoa**:

| threads (-t) | -ngl 0 (CPU decode) | -ngl 99 (GPU decode) |
|--:|--:|--:|
| 1 | 5.88 | 110.23 |
| 4 | 21.28 | 111.78 |
| 8 | **22.98** ← knee | 111.41 |
| 16 | 19.26 ← tụt | 111.55 |

Phía CPU: 1 → 4 thread được 3.6× (một thread thì đói issue width, còn xa mới chạm giới hạn
memory). 4 → 8 chỉ thêm 8%. 8 → 16 **mất 16%**. Điểm 16-thread mới là điểm cung cấp thông
tin: 8 thread thêm đó không phải core mới, chúng là **SMT sibling** dùng chung load/store
unit và lát L2 với thread đã ngồi sẵn ở đó. Decode đọc ma trận trọng số một lần mỗi token
và làm rất ít phép tính trên mỗi byte đọc được, nên khi 8 core đã phát lệnh load thì kênh
DDR5 đã là ràng buộc rồi; thêm 8 requester nữa chỉ thêm tranh chấp và thrash cache. Đó
đúng là chữ ký của **memory-bandwidth-bound**: throughput bão hoà ở chỗ **hết kênh nhớ**
chứ không phải hết core, rồi đi lùi.

Một chi tiết củng cố: error bar là ±0.07 ở `-t 1` nhưng **±4.02 ở `-t 4` và ±2.21 ở `-t 8`**.
Độ lệch giữa các lần chạy phình lên đúng chỗ thread bắt đầu tranh chấp — bản thân phương
sai là bằng chứng của contention, không phải nhiễu cần bình quân đi.

Chuyển sang GPU đổi luôn *chỗ* nút thắt nằm: từ vài chục GB/s của DDR5 dual-channel chia
cho 8 core, sang GDDR6 riêng của 4060 với memory controller rộng hơn nhiều và hàng nghìn
thread che được latency. 4.85× là kích thước của bước nhảy đó.

**Điều tôi cố ý KHÔNG làm:** tôi không quy đổi các con số này ra GB/s. Gemma 4 **E2B** là
model kiểu MatFormer — file 2.95 GiB / 4.65 B tham số nhưng **chỉ một phần được kích hoạt
mỗi token**, nên `tok/s × kích thước file` sẽ thổi phồng số byte thực sự di chuyển và cho
ra một con số bandwidth không có thật. Đường cong thread là bằng chứng mạnh hơn và nó
**không phụ thuộc** vào việc biết chính xác bao nhiêu tham số được kích hoạt: một workload
đạt đỉnh ở đúng số core vật lý rồi **mất** throughput vì SMT thì là bandwidth-bound, bất kể
mỗi token đọc bao nhiêu byte.

**Ý nghĩa cho phần còn lại của repo:** mọi số base track ở đây đều sinh ra với `ngl=99`.
Chúng là số **GPU-serving**, không phải số laptop-CPU, và 4.85× ở trên là khoảng cách đó.
Bạn cùng lớp chạy CPU-only không phải đang đo một phiên bản chậm hơn của setup tôi — họ
đang đo **cột bên trái**, nơi `-t` là knob quan trọng còn của tôi thì không.

---

## 6. Bonus  *(optional — tối đa 20 điểm)*

Không làm bonus track. Toàn bộ báo cáo trên là base track.

_(Phần so sánh `-ngl 0` vs `-ngl 99` ở §5 được sinh bằng `llama-bench` chạy tay để giải
thích kết quả `make tune`, nên tôi tính nó vào base track (rubric 11) và **không** khai nó
là bonus B2/B3 — rubric yêu cầu số bonus phải đến từ `make sweep-*` hoặc
`make compare-builds`, mà tôi chưa chạy các lệnh đó.)_

---

## 7. Điều làm bạn ngạc nhiên nhất  *(optional)*

Hai lần, và cả hai đều là lúc **các con số không khớp nhau** chứ không phải lúc chúng đẹp.
Lần một: `make tune` cho 1.00× và tôi suýt ghi "thread count không quan trọng" — trong khi
sự thật là nó rất quan trọng, chỉ là không phải trên cấu hình tôi đang chạy. Lần hai:
`llm` = 2863 ms trong khi server tự khai 425 ms; nếu tôi tin client mà không đối chiếu với
`timings` của server thì tôi đã đi tối ưu inference để chữa một **TCP timeout 2 giây**.
Bài học chung: hai dụng cụ đo bất đồng nhau thì đó là thông tin, không phải phiền toái.

---

## 8. Self-check trước khi push

- [x] `hardware.json` committed
- [x] `models/active.json` committed
- [x] `benchmarks/01-quickstart-results.md` committed (`make bench`)
- [x] `benchmarks/01-tuning-tg128.md` committed (`make tune`)
- [x] `benchmarks/02-server-results.md` committed (`make load-report`)
- [x] `benchmarks/02-server-batching-u50.md` + `-metrics-u50.csv` committed (`make metrics`)
- [x] `benchmarks/locust-10_stats.csv` + `locust-50_stats.csv` committed (`make load-10` / `load-50`)
- [x] `benchmarks/03-integration-results.md` committed (`make pipeline`)
- [x] Mọi section **"required — replace this line"** trong các file `benchmarks/*.md` đã
      được thay bằng nhận xét của tôi
- [x] 5 screenshots trong `submission/screenshots/`
- [x] `make verify` → **exit 0**
- [ ] Repo GitHub ở chế độ **public**
- [ ] Đã paste public URL vào VinUni LMS
- [x] **Không** commit `models/*.gguf` hay `runtime/` (đã có trong `.gitignore`)
