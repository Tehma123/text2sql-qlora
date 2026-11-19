# SQLocal — Hỏi đáp với file CSV ngay trong trình duyệt (Spec v1)

Nov 17, 2026 · @Tehma

## Mục tiêu & người dùng

SQLocal cho người dùng kéo thả một file CSV, hỏi bằng ngôn ngữ tự nhiên và nhận lại bảng kết quả. Mọi thứ chạy trong trình duyệt, dữ liệu không rời khỏi máy.

- **Người dùng:** dân văn phòng, sinh viên, người làm phân tích không viết SQL, có file không muốn upload lên dịch vụ cloud.
- **Vì sao chạy local:** (1) riêng tư — điểm bán hàng chính; (2) deploy free trên GitHub Pages vì không cần server compute; (3) chạy offline sau lần tải đầu.
- **Với portfolio:** nâng dự án Text-to-SQL fine-tune QLoRA có sẵn thành sản phẩm deploy được, không tốn thêm slot CV.

## Phạm vi v1

v1 chỉ hỏi đáp trên một file CSV, một lượt mỗi câu, chỉ đọc dữ liệu, câu hỏi bằng tiếng Anh.

**Làm trong v1**

- Nhận một file CSV; tự đọc tên cột, kiểu dữ liệu và vài dòng mẫu.
- Câu hỏi trên một bảng: lọc, nhóm, tổng hợp, sắp xếp, top-k.
- Hiện SQL sinh ra; người dùng sửa tay được rồi chạy lại.
- Tự sửa lỗi: SQL lỗi thì đưa thông báo lỗi lại cho model, tối đa 2 lần.
- Biểu đồ tự động (cột hoặc đường) khi kết quả có một cột phân loại và một cột số.
- Xuất kết quả ra CSV.

**Không làm trong v1**

- JOIN nhiều file, file Excel nhiều sheet (để v2).
- Ghi hoặc sửa dữ liệu: chỉ cho SELECT.
- Hội thoại nhiều lượt có ngữ cảnh (để v2).
- Tài khoản, lưu lịch sử trên server.
- Mobile: chỉ hỗ trợ chính thức Chrome/Edge trên desktop.
- Câu hỏi tiếng Việt: v1 chỉ xử lý tiếng Anh (dữ liệu train Spider toàn tiếng Anh).

## Kiến trúc

Không có server compute: GitHub Pages phát app tĩnh, HF Hub phát trọng số model, mọi tính toán diễn ra trên máy người dùng.

&#91;embedded content: kiến trúc SQLocal · 2 nguồn tĩnh, 4 thành phần trong trình duyệt\]

Model và DuckDB-WASM cùng nằm trong một Web Worker, nên SQL chuyển thẳng từ model sang database mà không đi qua giao diện. Model chỉ tải một lần rồi được cache cho các lần sau.

## Model

Mục tiêu: Qwen2.5-Coder-1.5B-Instruct fine-tune QLoRA, export sang ONNX q4f16 (khoảng 1,3 GB) và chạy bằng Transformers.js trên WebGPU.

Vì sao chọn 1.5B: bản q4f16 của base model nặng 1,34 GB ([nguồn](https://huggingface.co/onnx-community/Qwen2.5-Coder-1.5B-Instruct/tree/main/onnx)); bản 3B phải tải hơn 2 GB; 7B không thực tế trong trình duyệt.

Model hiện tại: Qwen2.5-Coder-3B-Instruct fine-tune QLoRA trên Spider (bản v2, `checkpoint-600`: 71.7% execution accuracy zero-shot trên test). Kế hoạch là làm cả hai bản, theo thứ tự này:

1. **Dùng ngay bản 3B đã fine-tune cho M1.** Merge, export ONNX, quantize model của bạn trước. Bước export là rủi ro lớn nhất của dự án, và bạn kiểm tra được nó ngay mà không cần train lại gì.
2. **Train thêm bản 1.5B song song**, cùng data, pipeline và eval set. Bản này nhẹ hơn nên train trên 3060 nhanh hơn bản 3B bạn đã train.
3. **Để số liệu quyết định.** So sánh 3B và 1.5B trên cả PyTorch lẫn ONNX: accuracy, thời gian tải, latency.
4. **Sản phẩm:** mặc định dùng 1.5B (nhẹ, Apache 2.0). Thêm tùy chọn “chính xác cao” tải bản 3B cho máy mạnh, nếu chênh lệch accuracy đủ lớn để đáng làm.

Quy trình export, áp dụng cho cả hai bản:

1. Merge adapter vào base ở fp16 (`merge_and_unload`), không merge vào bản 4-bit.
2. Export ONNX (`text-generation-with-past`) theo đúng cấu trúc repo onnx-community. Chạy thử bản fp32 bằng Transformers.js trên Node trước khi quantize.
3. Quantize q4f16 (cho WebGPU) và q4 (cho WASM dự phòng).
4. Đẩy lên model repo trên HF Hub, trọng số đặt trong thư mục `onnx/`.
5. Đánh giá lại bản ONNX trên cùng eval set (xem Đánh giá & số đo).

Format prompt lúc train (`src/text2sql/prompts/templates.py`): chat template của Qwen; system prompt yêu cầu viết "SQLite query", chỉ trả SQL; user message là `### Schema:\n{CREATE TABLE ...}\n\n### Question:\n{câu hỏi}\n\n### SQL:`. DDL hiện có đủ khoá chính, chưa có dòng mẫu.

Prompt khi chạy phải giữ đúng format lúc train. Schema đưa vào dưới dạng `CREATE TABLE` rút gọn (tên cột, kiểu) kèm 3 dòng mẫu, để bước prefill trong trình duyệt ngắn.

## Luồng xử lý một câu hỏi

Mỗi câu hỏi đi qua một vòng sinh, chạy, sửa. Người dùng luôn thấy được SQL, kể cả khi nó lỗi.

&#91;embedded content: luồng xử lý một câu hỏi · 6 bước, 2 điểm rẽ nhánh\]

- **Đọc schema:** DuckDB trả về tên cột, kiểu dữ liệu và 3 dòng mẫu ngay khi nạp file, không phải đọc lại ở mỗi câu hỏi.
- **Kiểm tra an toàn:** chỉ câu bắt đầu bằng SELECT hoặc WITH mới được chạy; câu khác được tính là lỗi và đi vào vòng tự sửa.
- **Tự sửa:** prompt sửa gồm câu hỏi gốc, SQL vừa sinh và thông báo lỗi của DuckDB. Sau 2 lần vẫn lỗi thì dừng và cho người dùng sửa tay.

## Đánh giá & số đo

Mục đích là có một bảng số trả lời được câu: chạy local mất bao nhiêu độ chính xác, đổi lại được gì. Đo trên 2 máy: máy của bạn (RTX 3060, Chrome) và một laptop không có GPU rời.

| Chỉ số | Cách đo | Mục tiêu |
| --- | --- | --- |
| Execution accuracy, PyTorch fp16 | Eval set hiện có | Baseline |
| Execution accuracy, ONNX q4f16 | Cùng eval set, chạy bằng Transformers.js | Chốt sau khi có baseline |
| Accuracy trên CSV thật | 50 câu tự viết trên 5 file CSV thật, tách riêng khỏi dữ liệu train | Chốt sau khi có baseline |
| Tỉ lệ cứu được nhờ tự sửa lỗi | Số câu sai ở lần 1, đúng sau lần sửa 1–2 | Ghi lại |
| Thời gian tải model | Lần đầu và lần sau (đã cache) | Ghi lại |
| Latency | Time-to-first-token và tổng thời gian sinh một câu SQL | Ghi lại |
| Test với người dùng | 5–10 người dùng thử, ghi chỗ họ bị kẹt | Sửa các vấn đề gặp nhiều nhất |

## Xử lý lỗi & UX

Nguyên tắc: không bao giờ im lặng trả về kết quả sai. Mọi SQL đều được hiện ra, và mọi lỗi đều có lời nhắn rõ ràng. Model và DuckDB chạy trong Web Worker để giao diện không bị đơ.

| Tình huống | Xử lý |
| --- | --- |
| Máy không có WebGPU | Chuyển sang WASM với bản q4, báo trước là sẽ chậm |
| Lần đầu phải tải khoảng 1,3 GB | Hỏi trước khi tải, có thanh tiến trình, cache lại cho lần sau |
| SQL sai cú pháp hoặc sai tên cột | Gửi lỗi của DuckDB lại cho model, tối đa 2 lần; vẫn lỗi thì hiện SQL kèm lỗi và cho sửa tay |
| SQL không phải SELECT/WITH | Chặn trước khi chạy, tính là một lỗi trong vòng tự sửa |
| Kết quả 0 dòng | Nói rõ "0 dòng", hiện vài giá trị thật của cột đang lọc để người dùng kiểm tra |
| Tên cột có dấu, khoảng trắng hoặc ký tự đặc biệt | Luôn đặt tên cột trong ngoặc kép trong schema đưa vào prompt |
| File lớn | DuckDB đọc trực tiếp; giao diện chỉ hiện 100 dòng đầu, xuất CSV để lấy đủ |
| Câu hỏi mơ hồ | v1 không hỏi lại; hiện SQL kèm một dòng diễn giải để người dùng tự kiểm tra |

## Lộ trình triển khai

Sáu mốc, mỗi mốc có điều kiện hoàn thành rõ ràng. Ngày cụ thể chưa chốt (xem Câu hỏi mở).

1. **M0 — Thử tính khả thi:** model gốc onnx-community 1.5B và DuckDB-WASM chạy trong một trang tĩnh, sinh và chạy được một câu SQL. *Xong khi:* đo được thời gian tải và latency trên 2 máy.
2. **M1 — Model của bạn chạy trong trình duyệt:** export bản 3B trước, train 1.5B song song, rồi merge, export, quantize, đánh giá cả hai (xem Model). *Xong khi:* có bảng accuracy PyTorch so với ONNX q4f16 cho cả 3B và 1.5B.
3. **M2 — MVP:** kéo thả CSV, tự đọc schema, sinh SQL, hiện SQL và bảng, chặn non-SELECT. *Xong khi:* đã deploy lên GitHub Pages.
4. **M3 — Độ bền:** tự sửa lỗi, dự phòng WASM, tên cột đặc biệt, kết quả rỗng, biểu đồ, xuất CSV. *Xong khi:* mọi tình huống trong bảng Xử lý lỗi đều đã test.
5. **M4 — Người dùng thật:** 5–10 người dùng thử, làm bộ test trên CSV thật. *Xong khi:* đã sửa các vấn đề gặp nhiều nhất.
6. **M5 — Đóng gói portfolio:** README (kiến trúc, bảng số đo, các đánh đổi), video demo 60 giây, cập nhật CV.

## Việc cần chạy cho M1

Ghi lại ngày 17/11/2026, chưa chạy. Repo vừa clone lại: `checkpoints/` và `data/raw/` đang trống, adapter 3B v2 (`checkpoint-600`) không còn trên máy và chưa từng được đẩy lên HF Hub. Vì vậy bước 1 của kế hoạch Model cần train lại bản 3B trước.

- [x] **Môi trường** (18/11): `.venv` Python 3.11 (tạo bằng `uv`). `pip install -r requirements.txt` không đủ: nó kéo torch bản CPU và `unsloth 2025.7.2` không chạy được với `trl` mới. Bộ đang chạy: `torch 2.14.1+cu132`, `torchvision 0.29.1+cu132` (index `https://download.pytorch.org/whl/cu132`), `unsloth`/`unsloth-zoo 2026.11.6`, `trl 1.15.0`, `transformers 4.57.6`. xformers không khớp torch nên Unsloth dùng attention của PyTorch. Bản Unsloth mới tự bật padding-free, khác lúc train v2 cũ, nên kết quả train lại có thể lệch nhẹ. `requirements.txt` giờ đã ghim bộ này (cài torch CUDA trước, `onnxruntime-genai` cài riêng với `--no-deps`).
- [x] **Dữ liệu** (18/11): `python scripts/download_data.py`, rồi `python scripts/prepare_data.py`. Ra train 8.575 / dev 1.034 / test 2.147, khớp `reports/data_preparation.md`.
- [x] **Smoke test** (18/11): `train.py --limit 20 --max-steps 2` (khoảng 6,5 s/bước) và `evaluate.py --limit 5` đều chạy được; output ghi ra ngoài repo nên `checkpoints/v2` vẫn trống.
- [x] **Train lại 3B v2** (18/11): dừng tay sau `checkpoint-600` (khoảng 9 s/bước). eval_loss thấp nhất ở bước 200 (0.2345) rồi tăng dần, lên 0.3036 ở bước 600, cùng hình dạng với lần train cũ. Dev execution accuracy (1.034 câu, zero-shot, `results/finetuned/v2_retrain/`): **checkpoint-600 75,1%** (exact match 45,7%), checkpoint-500 72,3% (44,4%); 600 hơn 500 có ý nghĩa (McNemar chính xác, p=0,004). Thấp hơn bản v2 cũ 1,3 điểm (76,4%); không so cặp được vì kết quả cũ không còn. Chọn checkpoint-600. Bước 200 không chấm. Lệnh: `python scripts/train.py` (config hiện tại), rồi `python scripts/evaluate.py --checkpoints checkpoints/v2/checkpoint-600 checkpoints/v2/checkpoint-500 --split dev --output-dir results/finetuned/v2_retrain` (khoảng 30 phút mỗi checkpoint). Có thể chạy tiếp từ bước 600 bằng `--resume-from-checkpoint checkpoints/v2/checkpoint-600`.
- [x] **Sao lưu adapter** (18/11): `checkpoint-600` đã lên repo private [`Tehma/text2sql-qwen2.5-coder-3b-lora`](https://huggingface.co/Tehma/text2sql-qwen2.5-coder-3b-lora) (adapter, tokenizer, `trainer_state.json`; không có `optimizer.pt`/`scheduler.pt`/`rng_state.pth`, nên muốn train tiếp phải dùng bản trên máy). Checksum adapter khớp bản local.
- [x] **Merge và export 3B** (18/11): bỏ fp32 vì máy chỉ còn khoảng 16 GB RAM trống; đi thẳng fp16 → q4f16 (bản q4 cho WASM bỏ với 3B, làm với 1.5B).
  - `python scripts/merge_adapter.py --adapter checkpoints/v2/checkpoint-600 --output checkpoints/v2/merged-600-fp16 --verify 200 --reference results/finetuned/v2_retrain/checkpoint-600_dev_zero_shot.json`: merge vào `Qwen/Qwen2.5-Coder-3B-Instruct` fp16 (6,2 GB), khoảng 3 phút.
  - Lần export đầu đi qua optimum/torch: export fp16 trên GPU (khoảng 1 phút, `no_post_process` vì protobuf mới không serialize được graph >2 GB), rồi lượng tử hoá 4-bit cả MatMul lẫn embedding (`GatherBlockQuantized`) bằng `MatMulNBitsQuantizer`. Nhánh code này đã bị bỏ khi dọn repo ngày 19/11 (chưa từng commit) vì bản build bằng onnxruntime-genai bên dưới nhanh hơn khoảng 6 lần.
  - Dung lượng: `model_q4f16.onnx_data` **1,92 GB** (embedding để fp16 thì 2,37 GB; embedding chiếm 622 MB). `model_fp16` 7,1 GB không chấm được trên 12 GB VRAM (tràn bộ nhớ), chỉ là bước trung gian.
  - 200 câu dev đầu, execution accuracy: 4-bit + adapter 67,0% · PyTorch fp16 đã merge 66,0% · **ONNX q4f16 64,0%** (exact match 36,0%). q4f16 so với 4-bit + adapter: McNemar p=0,29, không có ý nghĩa. Lượng tử hoá embedding không đổi kết quả (64,0% cả hai bản, 187/200 câu SQL trùng hệt).
- [x] **Chạy thử q4f16 trong trình duyệt** (18/11): `web/spike/` (`serve.py` phục vụ model ở 127.0.0.1, `run_spike.py` dùng Playwright điều khiển Edge 157, không cần Node). Transformers.js 4.3.2 (onnxruntime-web 1.31), WebGPU trên RTX 3060 (`shader-f16` có).
  - **Chạy được, kể cả `GatherBlockQuantized`.** 5/5 câu dev đầu sinh SQL trùng hệt onnxruntime CUDA.
  - Cần sửa: Transformers.js chỉ đọc chat template trong `tokenizer_config.json`, không đọc `chat_template.jinja`; `export_onnx.py` giờ tự ghép vào.
  - Số đo (server local, nên chưa tính thời gian tải qua mạng): nạp model 11,5 s lần đầu, 10,6 s lần sau; TTFT câu đầu 2,5–3,3 s (biên dịch shader), các câu sau khoảng 1,1 s với prompt khoảng 320 token; sinh **khoảng 6,5–7 token/s** (125–172 ms/token). Một câu SQL 15–28 token mất 3–5,8 s.
- [x] **Tìm nguyên nhân sinh token chậm** (18/11): so với `onnx-community/Qwen2.5-Coder-3B-Instruct` q4f16 (base, chưa fine-tune) trên cùng trang và máy: **bản đó sinh khoảng 43 token/s (23 ms/token), bản của mình khoảng 7 token/s (140–154 ms/token), chậm khoảng 6 lần**; TTFT 0,75 s so với 1,1 s; nạp 7,5 s so với 11 s. Nguyên nhân là graph: bản onnx-community tạo bằng **onnxruntime-genai model builder** (1.920 node, có `MultiHeadAttention`, `RotaryEmbedding`, `SkipSimplifiedLayerNormalization` đã gộp); bản của mình export qua optimum/torch (9.100 node, không gộp gì, 412 Cast, 3.247 Constant). `config.json` của họ còn có `transformers.js_config` (`kv_cache_dtype: float16`, `use_external_data_format: true`). Ghi chú thêm: base model hay bọc SQL trong khung ```sql, bản fine-tune thì không.
- [x] **Build lại bằng onnxruntime-genai model builder** (19/11): `python scripts/export_onnx.py --merged checkpoints/v2/merged-600-fp16 --output checkpoints/v2/onnx-600-genai --verify 200 --reference results/finetuned/v2_retrain/checkpoint-600_dev_zero_shot.json` (khoảng 1,5 phút; cài `onnxruntime-genai==0.20.0` với `--no-deps` để không đè `onnxruntime-gpu`).
  - Hai lỗi của builder 0.20 phải vá lúc chạy (lớp bọc `_BUILDER_SHIM` trong `export_onnx.py`): `rope_scaling=None` làm builder crash; và với WebGPU builder chọn `GroupQueryAttention`, vốn cần KV cache cấp phát sẵn của onnxruntime-genai nên **lỗi trong Transformers.js** ("past_key dimension 3 ... got 0 expected 128"). Tắt GQA để builder dùng `MultiHeadAttention` + `RotaryEmbedding` + `position_ids`, giống bản onnx-community.
  - Kết quả: 2.601 node, `shared_embeddings` (embedding và lm_head dùng chung một bản 4-bit), **1,74 GB**. Accuracy 200 câu dev: **64,0%**, 197/200 câu SQL trùng hệt bản optimum.
  - Trình duyệt (Edge, WebGPU, RTX 3060): **khoảng 45 token/s** (22–23 ms/token, nhanh khoảng 6 lần bản optimum, ngang bản onnx-community); TTFT 0,47 s (câu đầu 1,4–1,7 s); nạp 6,6–7,0 s; **một câu SQL mất 0,5–1,0 s**. 5/5 câu SQL trùng onnxruntime CUDA.
  - Bản dùng: `checkpoints/v2/onnx-600-genai/`. Kết quả đo của cả ba bản nằm trong `web/spike/results/`.
- [x] **Dọn repo** (19/11): xoá các bản build không dùng nữa (`onnx-600` bằng optimum, 8,5 GB; `onnx-600-genai-gqa`, 1,7 GB), model tham chiếu onnx-community (tải lại được), `checkpoint-100`…`400` (chưa từng chấm accuracy; eval_loss của chúng vẫn còn trong `trainer_state.json` của `checkpoint-600`) và log cũ. Còn giữ: `checkpoint-500`, `checkpoint-600`, `merged-600-fp16`, `onnx-600-genai`.
- [x] **Đẩy bản ONNX lên HF Hub** (19/11): [`Tehma/text2sql-qwen2.5-coder-3b-onnx`](https://huggingface.co/Tehma/text2sql-qwen2.5-coder-3b-onnx), **public**, theo cấu trúc onnx-community. Checksum `model_q4f16.onnx_data` khớp bản local.
  - Giấy phép Qwen Research cho phép phân phối lại (phi thương mại) nếu kèm bản giấy phép (`LICENSE`), có file `NOTICE` với câu ghi công bắt buộc và danh sách file đã sửa, và ghi "Built with Qwen" trong tài liệu (đã thêm vào model card).
- [x] **Chạy thử trang tải trực tiếp từ Hub** (19/11): `python web/spike/run_spike.py --n 5 --model Tehma/text2sql-qwen2.5-coder-3b-onnx` (model id có `/` thì trang tải từ Hub, không có thì tải từ `serve.py`). Không cần token.
  - Tải lần đầu (profile trống): **53 s** và 90 s ở hai lần chạy, phụ thuộc mạng (1,74 GB). Tải lại khi đã có cache: **3,0 s** (nhanh hơn bản localhost 7 s, vì `serve.py` gửi `no-store`). Sinh token và 5/5 câu SQL giống hệt khi chạy local.
  - Transformers.js lưu cả model lẫn file wasm của onnxruntime-web vào Cache API (`transformers-cache`, 1,7 GB; quota của profile là 11,9 GB).
  - **Rủi ro cho SQLocal:** ở cửa sổ ẩn danh (và context mặc định của Playwright), quota quá nhỏ nên không lưu được cache, lần nào vào trang cũng tải lại cả 1,74 GB (lần chạy đầu đo được 48 s cho "lần tải lại"). App nên kiểm tra `navigator.storage.estimate()` trước khi tải, báo cho người dùng nếu quota không đủ, và gọi `navigator.storage.persist()` để trình duyệt không tự xoá cache.
- [ ] **Train 1.5B:** sửa `QUANTIZED_MODEL_ID` trong `src/text2sql/models/load_model.py` (đang gán cứng bản 3B) thành tham số theo config, tạo config riêng cho 1.5B với `output_dir` riêng, rồi train với cùng data, pipeline và eval set.
- [ ] **So sánh:** bảng 3B và 1.5B × PyTorch và ONNX q4f16: accuracy, thời gian tải, latency.

## Rủi ro & câu hỏi mở

Rủi ro lớn nhất là bước export ONNX từ model tự fine-tune, nên M0 và M1 được xếp lên trước mọi việc làm giao diện.

| Rủi ro | Ảnh hưởng | Phương án |
| --- | --- | --- |
| Tự convert ONNX bị lỗi: đã có người gặp với Qwen2.5-Coder ([issue #1415](https://github.com/huggingface/transformers.js/issues/1415)) | Chặn M1 | Theo đúng cấu trúc repo onnx-community, test fp32 trước. Dự phòng: build MLC cho WebLLM, đã có người làm với bản 1.5B fine-tune ([ví dụ](https://huggingface.co/bernardw/qwen2.5-coder-1.5b-codereview-taid-q4-mlc)) |
| Dữ liệu train theo dialect khác DuckDB (hàm ngày tháng, cách quote) | SQL sai khi chạy | Thêm mẫu theo cú pháp DuckDB vào dữ liệu train; vòng tự sửa bắt phần còn lại |
| Bản 1.5B kém hơn bản lớn | Demo yếu | Đo và ghi rõ đánh đổi; giới hạn v1 ở câu hỏi trên một bảng |
| Người xem CV ngại tải 1,3 GB | Ít người dùng thử | Video demo trong README; trang có sẵn ví dụ mẫu để xem trước khi tải |
| Máy người dùng yếu, không có WebGPU | Rất chậm | WASM với bản q4, cảnh báo trước |

**Câu hỏi mở**

- [x] Model hiện tại cỡ bao nhiêu B, train trên dataset nào? **3B (Qwen2.5-Coder-3B-Instruct), Spider.**
- [x] Format prompt lúc train cụ thể ra sao? **Xem cuối mục Model.**
- [x] Dữ liệu train có câu hỏi tiếng Việt không? **Không; v1 chỉ xử lý tiếng Anh.**
- [ ] Mốc thời gian và deadline cho từng milestone?
- [ ] Tên chính thức (SQLocal là tên tạm).

## Nguồn

- [onnx-community/Qwen2.5-Coder-1.5B-Instruct — các file ONNX](https://huggingface.co/onnx-community/Qwen2.5-Coder-1.5B-Instruct/tree/main/onnx)
- [Hugging Face Spaces overview — phần cứng và gói](https://huggingface.co/docs/hub/en/spaces-overview)
- [transformers.js issue #1415](https://github.com/huggingface/transformers.js/issues/1415)
- [Qwen2.5-Coder-1.5B build MLC cho WebLLM](https://huggingface.co/bernardw/qwen2.5-coder-1.5b-codereview-taid-q4-mlc)
