# So sánh hiệu năng giữa SGLang và VLLM: Hướng dẫn triển khai và đánh giá

## Giới thiệu

Trong bối cảnh trí tuệ nhân tạo (AI) ngày càng phát triển, các mô hình ngôn ngữ lớn (LLMs) đã trở thành công cụ quan trọng trong nhiều ứng dụng, từ chatbot đến hoàn thiện mã nguồn. Tuy nhiên, hiệu năng của các mô hình này phụ thuộc rất lớn vào các công cụ suy luận (inference frameworks) được sử dụng để triển khai chúng. Hai trong số các framework nổi bật hiện nay là **SGLang** và **VLLM**.

**SGLang** là một framework thế hệ ngôn ngữ có cấu trúc, được thiết kế để tối ưu hóa việc quản lý tài nguyên và tính toán thông qua các công cụ lập trình chuyên biệt. Nó hỗ trợ các kỹ thuật như **RadixAttention** để tái sử dụng bộ nhớ KV cache và sử dụng các kernel CUDA hiệu suất cao ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)). Trong khi đó, **VLLM** là một thư viện suy luận LLM nổi tiếng với cơ chế **PagedAttention**, giúp tối ưu hóa việc sử dụng bộ nhớ và tăng tốc độ xử lý mà không làm giảm độ chính xác ([Chitpakdee, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

Mặc dù cả hai framework đều có những ưu điểm riêng, việc lựa chọn framework phù hợp phụ thuộc vào các yếu tố như tốc độ xử lý, khả năng xử lý đồng thời (concurrent requests), và độ trễ (latency). Các nghiên cứu gần đây cho thấy **SGLang** thường vượt trội hơn **VLLM** trong các bài kiểm tra hiệu năng, đặc biệt là khi xử lý các yêu cầu đồng thời hoặc các tác vụ yêu cầu thông lượng cao ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

Bài báo cáo này sẽ hướng dẫn cách viết mã để so sánh hiệu năng giữa SGLang và VLLM. Chúng tôi sẽ trình bày cách thiết lập môi trường, triển khai các máy chủ suy luận, và thực hiện các bài kiểm tra hiệu năng với các chỉ số như **Time to First Token (TTFT)**, thông lượng (tokens per second), và độ trễ. Các phương pháp này sẽ dựa trên các công cụ có sẵn từ cả hai framework, chẳng hạn như `sglang.bench_serving` và API OpenAI của VLLM, để đảm bảo kết quả so sánh công bằng và chính xác. 

Việc thực hiện so sánh này không chỉ giúp người dùng hiểu rõ hơn về khả năng của từng framework mà còn cung cấp cơ sở để lựa chọn công cụ phù hợp với nhu cầu cụ thể của họ.

## Giới thiệu về SGLang và VLLM

### Đặc điểm nổi bật của SGLang

SGLang là một framework thế hệ ngôn ngữ có cấu trúc, được thiết kế để tối ưu hóa hiệu năng của các mô hình ngôn ngữ lớn (LLMs) thông qua việc sử dụng các kỹ thuật lập trình chuyên biệt. Một số đặc điểm nổi bật của SGLang bao gồm:

- **RadixAttention**: Đây là một cơ chế tối ưu hóa bộ nhớ KV cache, cho phép tái sử dụng dữ liệu đã được lưu trữ, từ đó giảm thiểu chi phí tính toán và tăng tốc độ xử lý ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).
- **Compressed State Machine**: SGLang sử dụng một máy trạng thái nén để hỗ trợ việc giải mã nhanh và chính xác trong các tác vụ có ràng buộc cao ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).
- **Python-based Batch Scheduler**: Trình quản lý lô yêu cầu dựa trên Python của SGLang thường đạt hiệu suất tương đương hoặc vượt trội so với các hệ thống dựa trên C++ ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

Ngoài ra, SGLang hỗ trợ các kernel CUDA hiệu suất cao từ FlashInfer và tích hợp các công cụ tối ưu hóa như `torch.compile` từ gpt-fast, giúp cải thiện tốc độ xử lý và khả năng mở rộng ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

### Đặc điểm nổi bật của VLLM

VLLM là một thư viện suy luận LLM được biết đến với cơ chế **PagedAttention**, giúp tối ưu hóa việc sử dụng bộ nhớ và tăng tốc độ xử lý mà không làm giảm độ chính xác. Một số đặc điểm nổi bật của VLLM bao gồm:

- **PagedAttention**: Cơ chế này cho phép quản lý bộ nhớ hiệu quả hơn, đặc biệt là khi xử lý các mô hình lớn với ngữ cảnh dài ([Chitpakdee, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).
- **Continuous Batching**: VLLM hỗ trợ việc xử lý các yêu cầu liên tục, giúp giảm độ trễ và tăng thông lượng trong các tác vụ suy luận ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).
- **Optimized CUDA Kernels**: VLLM tích hợp các kernel CUDA tối ưu hóa, bao gồm FlashAttention và FlashInfer, để cải thiện hiệu suất tính toán ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

VLLM cũng hỗ trợ nhiều định dạng giảm độ chính xác như INT4, INT8, FP8, và GPTQ, giúp giảm chi phí tính toán mà vẫn duy trì độ chính xác cao ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

### So sánh tổng quan giữa SGLang và VLLM

| **Tiêu chí**                | **SGLang**                                                                 | **VLLM**                                                                 |
|-----------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Cơ chế tối ưu hóa bộ nhớ** | RadixAttention, KV cache reuse                                           | PagedAttention                                                         |
| **Thông lượng**             | Cao hơn trong các bài kiểm tra đồng thời ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)) | Tốt trong các bài kiểm tra tuần tự ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)) |
| **Hỗ trợ phần cứng**        | NVIDIA, AMD (mới được hỗ trợ gần đây)                                    | NVIDIA, AMD, Intel, PowerPC, TPU, AWS Neuron ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)) |
| **Định dạng giảm độ chính xác** | FP8, INT8, 4-bit quantization                                           | GPTQ, AWQ, INT4, INT8, FP8 ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)) |

SGLang thường vượt trội hơn VLLM trong các bài kiểm tra hiệu năng với yêu cầu đồng thời, nhờ vào các cơ chế tối ưu hóa bộ nhớ và quản lý tài nguyên. Tuy nhiên, VLLM lại có lợi thế trong các bài kiểm tra tuần tự và hỗ trợ nhiều định dạng phần cứng hơn, làm cho nó trở thành lựa chọn linh hoạt hơn trong một số trường hợp ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

## Cài đặt và cấu hình môi trường thử nghiệm

### Chuẩn bị phần cứng và phần mềm

Để đảm bảo kết quả thử nghiệm chính xác và có thể tái lập, việc chuẩn bị phần cứng và phần mềm là bước quan trọng. Dưới đây là các yêu cầu và cấu hình cần thiết:

#### Phần cứng
- **CPU**: AMD EPYC 7J13 hoặc Intel Xeon 6442Y với tối thiểu 24 lõi và xung nhịp 2.6GHz.
- **GPU**: NVIDIA A100-SXM4 (40GB VRAM) hoặc NVIDIA H100 (80GB VRAM), hỗ trợ NVLink hoặc NV Bridge để tăng tốc độ truyền dữ liệu.
- **RAM**: Tối thiểu 216GB DDR4 hoặc DDR5 để đảm bảo khả năng xử lý các mô hình lớn.
- **Ổ cứng**: SSD NVMe với dung lượng tối thiểu 1TB để lưu trữ mô hình và dữ liệu thử nghiệm.

#### Phần mềm
- **Hệ điều hành**: Ubuntu 20.04 hoặc 22.04 LTS.
- **Python**: Phiên bản 3.10 hoặc mới hơn.
- **CUDA**: Phiên bản 11.8 hoặc 12.0 để tương thích với các kernel tối ưu hóa của cả SGLang và VLLM.
- **Thư viện bổ sung**: `torch`, `transformers`, `sglang`, và `vllm`.

Cấu hình này đảm bảo rằng cả hai framework có thể hoạt động tối ưu trên cùng một nền tảng phần cứng, giúp loại bỏ các yếu tố gây nhiễu trong quá trình so sánh hiệu năng ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

---

### Cài đặt SGLang và VLLM

#### Cài đặt SGLang
SGLang yêu cầu các thư viện CUDA và PyTorch được cài đặt trước. Sau đó, bạn có thể cài đặt SGLang thông qua `pip`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sglang
```

Để kiểm tra cài đặt, chạy lệnh sau:
```bash
python -m sglang --version
```

#### Cài đặt VLLM
Tương tự, VLLM cũng yêu cầu PyTorch và CUDA. Cài đặt VLLM bằng lệnh sau:
```bash
pip install vllm
```

Kiểm tra cài đặt bằng cách chạy:
```bash
python -m vllm.entrypoints.openai.api_server --help
```

Cả hai framework đều hỗ trợ các mô hình từ Hugging Face, vì vậy bạn cần đảm bảo rằng các mô hình được tải xuống trước khi thử nghiệm ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

---

### Cấu hình máy chủ suy luận

#### Khởi chạy máy chủ SGLang
Máy chủ SGLang được khởi chạy bằng lệnh sau:
```bash
python -m sglang.launch_server --model Qwen/Qwen2.5-Coder-7B-Instruct --chunked-prefill-size 32000 --port 30000
```
- `--chunked-prefill-size`: Kích thước bộ nhớ đệm KV cache.
- `--port`: Cổng để truy cập API.

#### Khởi chạy máy chủ VLLM
Máy chủ VLLM được khởi chạy bằng lệnh:
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-7B-Instruct --disable-log-requests --port 8000
```
- `--disable-log-requests`: Tắt ghi log để giảm tải hệ thống.
- `--port`: Cổng để truy cập API.

Cả hai máy chủ cần được khởi chạy trên cùng một máy để đảm bảo điều kiện thử nghiệm đồng nhất ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

---

### Kiểm tra kết nối và xác minh cấu hình

Sau khi khởi chạy máy chủ, bạn cần kiểm tra kết nối bằng cách gửi yêu cầu thử nghiệm. Ví dụ, sử dụng `curl` để kiểm tra API của VLLM:
```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "prompt": "Hello, world!",
  "max_tokens": 10
}'
```

Tương tự, kiểm tra API của SGLang:
```bash
curl -X POST http://localhost:30000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "prompt": "Hello, world!",
  "max_tokens": 10
}'
```

Nếu cả hai máy chủ trả về kết quả hợp lệ, cấu hình đã sẵn sàng để tiến hành thử nghiệm ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

## Phương pháp đo lường hiệu năng

### 1. Các chỉ số hiệu năng chính

Để so sánh hiệu năng giữa SGLang và VLLM, các chỉ số đo lường hiệu năng cần được xác định rõ ràng. Những chỉ số này sẽ giúp đánh giá khả năng xử lý của từng framework trong các điều kiện thử nghiệm khác nhau. Dưới đây là các chỉ số chính được sử dụng:

- **Time to First Token (TTFT)**: Thời gian từ khi gửi yêu cầu đến khi nhận được token đầu tiên. Chỉ số này đo lường độ trễ ban đầu của hệ thống và rất quan trọng trong các ứng dụng yêu cầu phản hồi nhanh ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).
- **Thông lượng (Tokens per Second)**: Số lượng token được tạo ra mỗi giây. Đây là chỉ số quan trọng để đánh giá khả năng xử lý của framework trong các bài kiểm tra tuần tự và đồng thời ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).
- **Độ trễ toàn diện (End-to-End Latency)**: Thời gian từ khi gửi yêu cầu đến khi nhận được toàn bộ kết quả. Chỉ số này bao gồm cả TTFT và thời gian tạo các token tiếp theo ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

Các chỉ số này sẽ được đo lường trong các bài kiểm tra tuần tự và đồng thời để đảm bảo đánh giá toàn diện hiệu năng của cả hai framework.

---

### 2. Phương pháp thử nghiệm tuần tự và đồng thời

#### Thử nghiệm tuần tự
Trong thử nghiệm tuần tự, các yêu cầu được gửi lần lượt, đảm bảo rằng chỉ có một yêu cầu được xử lý tại một thời điểm. Phương pháp này giúp đánh giá hiệu năng của framework trong điều kiện tải thấp. 

Ví dụ mã thử nghiệm tuần tự:
```python
import time
import requests

def sequential_benchmark(url, model_name, prompt, max_tokens, num_requests):
    latencies = []
    for _ in range(num_requests):
        start_time = time.time()
        response = requests.post(
            url,
            json={
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
            },
        )
        end_time = time.time()
        latencies.append(end_time - start_time)
    return latencies

# URL của API
vllm_url = "http://localhost:8000/v1/completions"
sglang_url = "http://localhost:30000/v1/completions"

# Thử nghiệm tuần tự
vllm_latencies = sequential_benchmark(vllm_url, "default", "Hello, world!", 100, 10)
sglang_latencies = sequential_benchmark(sglang_url, "default", "Hello, world!", 100, 10)

print("VLLM Latencies:", vllm_latencies)
print("SGLang Latencies:", sglang_latencies)
```

#### Thử nghiệm đồng thời
Trong thử nghiệm đồng thời, nhiều yêu cầu được gửi cùng lúc để đánh giá khả năng xử lý tải cao của framework. Phương pháp này sử dụng các công cụ như `threading` hoặc `asyncio` để gửi yêu cầu đồng thời.

Ví dụ mã thử nghiệm đồng thời:
```python
import asyncio
import aiohttp

async def concurrent_benchmark(url, model_name, prompt, max_tokens, num_requests):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            tasks.append(
                session.post(
                    url,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                    },
                )
            )
        responses = await asyncio.gather(*tasks)
        return responses

# URL của API
vllm_url = "http://localhost:8000/v1/completions"
sglang_url = "http://localhost:30000/v1/completions"

# Thử nghiệm đồng thời
async def main():
    vllm_responses = await concurrent_benchmark(vllm_url, "default", "Hello, world!", 100, 10)
    sglang_responses = await concurrent_benchmark(sglang_url, "default", "Hello, world!", 100, 10)
    print("VLLM Responses:", vllm_responses)
    print("SGLang Responses:", sglang_responses)

asyncio.run(main())
```

Phương pháp thử nghiệm đồng thời này giúp đánh giá khả năng xử lý của framework khi phải đối mặt với tải cao và nhiều yêu cầu đồng thời.

---

### 3. Công cụ và cấu hình thử nghiệm

#### Công cụ đo lường
Các công cụ đo lường hiệu năng như `sglang.bench_serving` và `llmperf` được sử dụng để tự động hóa quá trình thử nghiệm và thu thập dữ liệu. Ví dụ, `sglang.bench_serving` hỗ trợ cả hai backend (SGLang và VLLM), giúp dễ dàng so sánh hiệu năng:

```bash
# Thử nghiệm với VLLM
python3 -m sglang.bench_serving --backend vllm --dataset-name random --random-input-len 30000 --random-output-len 500 --request-rate 1 --num-prompts 64

# Thử nghiệm với SGLang
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 30000 --random-output-len 500 --request-rate 1 --num-prompts 64
```

#### Cấu hình thử nghiệm
Các thông số thử nghiệm cần được giữ đồng nhất để đảm bảo kết quả công bằng:
- **Mô hình**: Sử dụng cùng một mô hình (ví dụ: Qwen/Qwen2.5-Coder-7B-Instruct).
- **Số lượng yêu cầu**: 64 yêu cầu cho mỗi thử nghiệm.
- **Độ dài đầu vào và đầu ra**: 30,000 token đầu vào và 500 token đầu ra.
- **Tốc độ yêu cầu**: 1 yêu cầu mỗi giây.

Việc sử dụng các công cụ và cấu hình này đảm bảo rằng các chỉ số hiệu năng được đo lường chính xác và có thể tái lập ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

## Kết quả thử nghiệm và phân tích

### Hiệu năng trong các bài kiểm tra tuần tự

Kết quả thử nghiệm tuần tự cho thấy sự khác biệt rõ rệt giữa SGLang và VLLM khi xử lý các yêu cầu đơn lẻ. Trong bài kiểm tra sử dụng mô hình **Llama3-70B-FP8** trên GPU NVIDIA H100, SGLang đạt tốc độ xử lý **38 tokens/giây**, trong khi VLLM chỉ đạt **35 tokens/giây**. Điều này cho thấy SGLang có khả năng tối ưu hóa tốt hơn trong việc xử lý các yêu cầu tuần tự ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

| **Framework** | **Tokens/giây (Llama3-70B-FP8)** | **Tokens/giây (Llama3.1-8B)** |
|---------------|----------------------------------|--------------------------------|
| **SGLang**    | 38                              | 91                             |
| **VLLM**      | 35                              | 80                             |

Ngoài ra, trong bài kiểm tra với mô hình **Llama3.1-8B**, SGLang tiếp tục vượt trội với tốc độ **91 tokens/giây**, cao hơn đáng kể so với VLLM ở mức **80 tokens/giây**. Điều này cho thấy SGLang có khả năng xử lý hiệu quả hơn trong các bài kiểm tra tuần tự với mô hình nhỏ hơn ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

### Hiệu năng trong các bài kiểm tra đồng thời

Khi xử lý các yêu cầu đồng thời, sự khác biệt giữa hai framework trở nên rõ ràng hơn. Trong bài kiểm tra với mô hình **Llama3-70B-FP8**, SGLang duy trì tốc độ ổn định ở mức **30-31 tokens/giây**, trong khi VLLM giảm từ **22 tokens/giây** xuống còn **16 tokens/giây** khi số lượng yêu cầu tăng lên ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

| **Framework** | **Tokens/giây (Llama3-70B-FP8)** | **Tokens/giây (Llama3.1-8B)** |
|---------------|----------------------------------|--------------------------------|
| **SGLang**    | 30-31                           | 75-78                          |
| **VLLM**      | 16-22                           | 35-37                          |

Trong bài kiểm tra với mô hình **Llama3.1-8B**, SGLang tiếp tục duy trì hiệu năng vượt trội với tốc độ **75-78 tokens/giây**, gần gấp đôi so với VLLM ở mức **35-37 tokens/giây**. Điều này cho thấy SGLang có khả năng xử lý tải cao và duy trì thông lượng ổn định hơn trong các bài kiểm tra đồng thời ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

### Phân tích độ trễ và thông lượng

#### Độ trễ toàn diện (End-to-End Latency)
SGLang có độ trễ toàn diện thấp hơn so với VLLM trong hầu hết các bài kiểm tra. Ví dụ, khi sử dụng mô hình **Llama3-70B-FP8**, SGLang đạt độ trễ trung bình **45,938ms**, trong khi VLLM có độ trễ trung bình **51,091ms**. Điều này cho thấy SGLang có khả năng xử lý nhanh hơn trong các tác vụ yêu cầu phản hồi nhanh ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

| **Framework** | **Độ trễ trung bình (ms)** | **Độ trễ P99 (ms)** |
|---------------|----------------------------|---------------------|
| **SGLang**    | 45,938                     | 9,065               |
| **VLLM**      | 51,091                     | 7,798               |

#### Thông lượng (Tokens per Second)
SGLang cũng vượt trội hơn về thông lượng trong các bài kiểm tra đồng thời. Với mô hình **Llama3-70B-FP8**, SGLang đạt thông lượng tổng cộng **25,428 tokens/giây**, cao hơn so với VLLM ở mức **23,770 tokens/giây**. Điều này cho thấy SGLang có khả năng tối ưu hóa tài nguyên tốt hơn để xử lý các yêu cầu đồng thời ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

| **Framework** | **Thông lượng tổng cộng (tokens/giây)** |
|---------------|----------------------------------------|
| **SGLang**    | 25,428                                |
| **VLLM**      | 23,770                                |

Những kết quả này cho thấy SGLang là lựa chọn tốt hơn trong các ứng dụng yêu cầu thông lượng cao và độ trễ thấp, đặc biệt là khi xử lý các yêu cầu đồng thời hoặc các tác vụ phức tạp. Tuy nhiên, VLLM vẫn là một lựa chọn phù hợp cho các ứng dụng yêu cầu tuần tự hoặc hỗ trợ phần cứng đa dạng hơn ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

## Kết luận và khuyến nghị

### Đánh giá tổng quan hiệu năng

Dựa trên các kết quả thử nghiệm, **SGLang** đã chứng minh khả năng vượt trội trong việc xử lý các yêu cầu đồng thời và duy trì thông lượng ổn định, đặc biệt là khi sử dụng các mô hình lớn như **Llama3-70B-FP8**. Trong các bài kiểm tra đồng thời, SGLang đạt tốc độ xử lý **30-31 tokens/giây**, gần gấp đôi so với VLLM, vốn chỉ đạt **16-22 tokens/giây** ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)). Điều này cho thấy SGLang là lựa chọn phù hợp cho các ứng dụng yêu cầu tải cao và khả năng mở rộng tốt.

Tuy nhiên, trong các bài kiểm tra tuần tự, sự khác biệt giữa hai framework không quá rõ rệt. VLLM vẫn duy trì hiệu năng ổn định và có lợi thế trong việc hỗ trợ phần cứng đa dạng hơn, bao gồm các nền tảng như Intel, PowerPC, và AWS Neuron ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

### Khuyến nghị về lựa chọn framework

#### Ứng dụng yêu cầu đồng thời cao
Nếu ứng dụng của bạn cần xử lý nhiều yêu cầu đồng thời, chẳng hạn như chatbot hoặc hệ thống tự động hóa quy mô lớn, **SGLang** là lựa chọn tối ưu. Các thử nghiệm đã chứng minh rằng SGLang có khả năng duy trì thông lượng cao và độ trễ thấp ngay cả khi phải xử lý tải lớn ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

#### Ứng dụng yêu cầu tuần tự hoặc hỗ trợ phần cứng đa dạng
Trong các trường hợp yêu cầu tuần tự hoặc cần hỗ trợ phần cứng đa dạng, **VLLM** là lựa chọn phù hợp hơn. VLLM có khả năng tối ưu hóa bộ nhớ tốt với cơ chế **PagedAttention**, giúp giảm chi phí tính toán mà vẫn duy trì độ chính xác cao ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

#### Tối ưu hóa thông số kỹ thuật
Để đạt hiệu năng tối ưu, người dùng nên điều chỉnh các thông số kỹ thuật phù hợp với từng framework. Ví dụ, sử dụng `--chunked-prefill-size` với giá trị lớn hơn cho SGLang hoặc `--enable-chunked-prefill` cho VLLM có thể cải thiện đáng kể hiệu năng trong các bài kiểm tra đồng thời ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

### Hướng dẫn triển khai trong môi trường thực tế

#### Tích hợp với hệ thống hiện có
Cả SGLang và VLLM đều hỗ trợ các API tương thích với OpenAI, giúp dễ dàng tích hợp vào các hệ thống hiện có. Người dùng có thể triển khai các máy chủ suy luận trên Docker hoặc Kubernetes để đảm bảo khả năng mở rộng và quản lý tài nguyên hiệu quả ([Clarifai, 2025](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)).

#### Lựa chọn mô hình phù hợp
Việc lựa chọn mô hình cũng ảnh hưởng lớn đến hiệu năng của framework. Các thử nghiệm cho thấy SGLang hoạt động tốt hơn với các mô hình lớn như **Llama3-70B-FP8**, trong khi VLLM có thể tối ưu hóa tốt hơn cho các mô hình nhỏ hơn như **Llama3.1-8B** ([Medium, 2024](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)).

#### Đánh giá hiệu năng định kỳ
Để đảm bảo hiệu năng tối ưu, người dùng nên thực hiện các bài kiểm tra định kỳ với các công cụ như `sglang.bench_serving` hoặc `llmperf`. Các bài kiểm tra này không chỉ giúp đánh giá hiệu năng mà còn cung cấp thông tin để tối ưu hóa cấu hình hệ thống ([GitHub, 2025](https://github.com/sgl-project/sglang/issues/3471)).

---

Những khuyến nghị này giúp người dùng lựa chọn framework phù hợp với nhu cầu cụ thể, đồng thời cung cấp hướng dẫn triển khai và tối ưu hóa trong môi trường thực tế.


## References

- [https://www.reddit.com/r/LocalLLaMA/comments/1k2zn6o/sglang_vs_vllm/](https://www.reddit.com/r/LocalLLaMA/comments/1k2zn6o/sglang_vs_vllm/)
- [https://github.com/sgl-project/sglang/issues/4245](https://github.com/sgl-project/sglang/issues/4245)
- [https://www.reddit.com/r/LocalLLaMA/comments/1jjl45h/compared_performance_of_vllm_vs_sglang_on_2/](https://www.reddit.com/r/LocalLLaMA/comments/1jjl45h/compared_performance_of_vllm_vs_sglang_on_2/)
- [https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang](https://www.clarifai.com/blog/comparing-vllm-lmdeploy-and-sglang)
- [https://github.com/sgl-project/sglang/issues/3471](https://github.com/sgl-project/sglang/issues/3471)
- [https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a](https://medium.com/@occlubssk/llm-inference-engines-performance-testing-sglang-vs-vllm-cfd2a597852a)
