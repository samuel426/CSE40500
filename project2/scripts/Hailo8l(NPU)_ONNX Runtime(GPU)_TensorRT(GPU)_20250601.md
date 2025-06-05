## NPU, ONNX Runtime(GPU), TensorRT(GPU) 성능 비교 분석 (`GRU_Apple` 및 `LSTM_Apple` 모델)

제공된 3개의 JSON 파일 ([`npu_performance_20250528_132327.json`](npu_performance_20250528_132327.json ) (NPU), [`onnx_performance_20250601_071411.json`](onnx_performance_20250601_071411.json ) (ONNX Runtime GPU), [`tensorrt_performance_20250601_070219.json`](tensorrt_performance_20250601_070219.json ) (TensorRT GPU))을 기반으로 `GRU_Apple` 및 `LSTM_Apple` 모델의 성능을 비교 분석합니다.

---

### 1. `GRU_Apple` 모델 성능 비교

#### 1.1. Latency (지연 시간, 단위: ms)

| 구분     | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :------- | :------------- | :----------------- | :------------- |
| **mean** | 1.326          | 0.186              | 0.072          |
| **median**| 1.319          | 0.183              | 0.071          |
| **min**  | 1.278          | 0.179              | 0.066          |
| **max**  | 1.417          | 0.299              | 0.084          |
| **std**  | 0.031          | 0.013              | 0.003          |
| **p95**  | 1.397          | 0.204              | 0.076          |
| **p99**  | 1.413          | 0.226              | 0.080          |

*   **분석:**
    *   TensorRT가 모든 지연 시간 지표에서 가장 우수한 성능을 보이며, NPU 대비 약 18.4배, ONNX Runtime 대비 약 2.6배 빠릅니다 (평균 기준).
    *   ONNX Runtime도 NPU에 비해 평균적으로 약 7.1배 빠른 지연 시간을 제공합니다.
    *   표준편차(std)는 TensorRT가 가장 작아 지연 시간 변동성이 가장 적었고, 그 다음이 ONNX Runtime, NPU 순입니다.

#### 1.2. Throughput (초당 처리량, 단위: FPS)

| 구분             | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :--------------- | :------------- | :----------------- | :------------- |
| **throughput_fps** | 764.05         | 5363.08            | 13942.90       |

*   **분석:**
    *   TensorRT가 가장 높은 처리량을 보이며, NPU 대비 약 18.2배, ONNX Runtime 대비 약 2.6배 높습니다.

#### 1.3. RMSE (Root Mean Squared Error, 예측 오차)

| 구분   | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :----- | :------------- | :----------------- | :------------- |
| **rmse** | 54.905         | 52.141             | 52.141         |

*   **분석:**
    *   ONNX Runtime과 TensorRT의 RMSE 값은 거의 동일하며, NPU보다 약간 낮은 오차율을 보입니다.

#### 1.4. Utilization & Resource Usage

| 지표                         | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :--------------------------- | :------------- | :----------------- | :------------- |
| **avg_device_utilization (%)**| 14.10          | 0.00               | 0.00           |
| **avg_model_utilization (%)** | 14.10          | 0.00               | 0.00           |
| **avg_gpu_memory_mb**        | N/A            | 1387.19            | 1393.69        |
| **avg_gpu_power_watts**      | N/A            | 27.57              | 68.33          |

*   **분석:**
    *   NPU는 약 14.10%의 사용률을 보인 반면, GPU 플랫폼에서는 `GRU_Apple` 모델 실행 시 GPU 사용률이 거의 0%로 측정되었습니다. 이는 해당 모델이 GPU 자원에 비해 매우 가볍다는 것을 의미합니다.
    *   GPU 메모리 사용량은 두 GPU 플랫폼에서 유사하게 약 1.39GB 수준입니다.
    *   전력 소비는 ONNX Runtime (약 27.6W)이 TensorRT (약 68.3W)보다 낮게 측정되었습니다. 이는 TensorRT가 더 많은 연산을 빠르게 처리하면서 순간적인 전력 소모가 높았을 수 있음을 시사합니다 (단, 이 값은 평균값이며, 실제 작업량 대비 효율은 다를 수 있습니다).

---

### 2. `LSTM_Apple` 모델 성능 비교

#### 2.1. Latency (지연 시간, 단위: ms)

| 구분     | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :------- | :------------- | :----------------- | :------------- |
| **mean** | 0.999          | 0.176              | 0.041          |
| **median**| 1.003          | 0.157              | 0.040          |
| **min**  | 0.956          | 0.153              | 0.039          |
| **max**  | 1.036          | 0.633              | 0.048          |
| **std**  | 0.015          | 0.069              | 0.002          |
| **p95**  | 1.022          | 0.282              | 0.045          |
| **p99**  | 1.035          | 0.513              | 0.048          |

*   **분석:**
    *   `LSTM_Apple` 모델에서도 TensorRT가 모든 지연 시간 지표에서 가장 우수한 성능을 보입니다. NPU 대비 평균 약 24.4배, ONNX Runtime 대비 평균 약 4.3배 빠릅니다.
    *   ONNX Runtime은 NPU에 비해 평균적으로 약 5.7배 빠른 지연 시간을 제공합니다.
    *   표준편차는 TensorRT가 가장 작아 매우 안정적인 성능을 보였고, 그 다음이 NPU, ONNX Runtime 순입니다.

#### 2.2. Throughput (초당 처리량, 단위: FPS)

| 구분             | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :--------------- | :------------- | :----------------- | :------------- |
| **throughput_fps** | 990.76         | 5675.65            | 24386.91       |

*   **분석:**
    *   TensorRT가 가장 높은 처리량을 보이며, NPU 대비 약 24.6배, ONNX Runtime 대비 약 4.3배 높습니다.

#### 2.3. RMSE (Root Mean Squared Error, 예측 오차)

| 구분   | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :----- | :------------- | :----------------- | :------------- |
| **rmse** | 76.580         | 64.950             | 64.950         |

*   **분석:**
    *   ONNX Runtime과 TensorRT의 RMSE 값은 거의 동일하며, NPU보다 낮은 오차율을 보입니다. `LSTM_Apple` 모델에서는 GPU 플랫폼의 정확도가 NPU보다 더 두드러지게 좋았습니다.

#### 2.4. Utilization & Resource Usage

| 지표                         | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) |
| :--------------------------- | :------------- | :----------------- | :------------- |
| **avg_device_utilization (%)**| 12.57          | 0.00               | 0.00           |
| **avg_model_utilization (%)** | 12.57          | 0.00               | 0.00           |
| **avg_gpu_memory_mb**        | N/A            | 1394.13            | 1385.63        |
| **avg_gpu_power_watts**      | N/A            | 84.52              | 86.93          |

*   **분석:**
    *   NPU는 약 12.57%의 사용률을 보였습니다. GPU 플랫폼에서는 `LSTM_Apple` 모델 실행 시에도 GPU 사용률이 거의 0%로 측정되었습니다. 이는 `LSTM` 모델 역시 GPU 자원에 비해 가볍다는 것을 의미합니다.
    *   GPU 메모리 사용량은 두 GPU 플랫폼에서 유사하게 약 1.39GB 수준입니다.
    *   전력 소비는 ONNX Runtime (약 84.5W)과 TensorRT (약 86.9W)가 유사한 수준으로 측정되었습니다. `GRU_Apple` 모델보다 `LSTM_Apple` 모델 실행 시 ONNX Runtime의 전력 소비가 더 높게 나타났습니다.

---

### 종합 결론

*   **성능 (Latency & Throughput):**
    *   두 모델(`GRU_Apple`, `LSTM_Apple`) 모두에서 **TensorRT(GPU)**가 압도적으로 가장 우수한 추론 성능(가장 낮은 지연 시간 및 가장 높은 처리량)을 제공합니다.
    *   **ONNX Runtime(GPU)**도 NPU에 비해 훨씬 뛰어난 성능을 보여주며, TensorRT보다는 느리지만 준수한 GPU 가속 성능을 제공합니다.
    *   **NPU(Hailo-8L)**는 절대적인 속도에서는 GPU 플랫폼에 비해 느리지만, 엣지 환경에서의 전력 효율성(이번 측정에서는 직접 비교 불가)과 특정 작업 부하에 대한 최적화를 목표로 합니다.

*   **정확도 (RMSE):**
    *   `GRU_Apple` 모델에서는 GPU 플랫폼(ONNX, TensorRT)이 NPU보다 약간 더 나은 RMSE를 보였습니다.
    *   `LSTM_Apple` 모델에서는 GPU 플랫폼의 RMSE가 NPU보다 눈에 띄게 더 좋았습니다. 이는 모델 복잡도나 양자화 민감도 등에서 차이가 있을 수 있음을 시사합니다.

*   **자원 사용률 및 소비 전력 (GPU):**
    *   테스트된 `GRU_Apple` 및 `LSTM_Apple` 모델은 사용된 GPU(RTX 3070 Ti로 추정)의 전체 연산 능력에 비해 매우 가벼워, GPU 사용률이 거의 0%로 측정되었습니다. 이는 해당 모델들에 대해 GPU가 과도한 사양일 수 있음을 의미합니다.
    *   GPU 메모리 사용량은 ONNX Runtime과 TensorRT 간에 큰 차이가 없었습니다.
    *   평균 전력 소비는 모델 및 실행 환경(ONNX vs TensorRT)에 따라 다소 변동성을 보였습니다. `GRU_Apple`에서는 ONNX Runtime이 더 낮았으나, `LSTM_Apple`에서는 두 GPU 플랫폼이 유사한 수준의 전력을 소비했습니다. 일반적으로 TensorRT는 더 높은 연산 강도로 인해 순간 전력이 높을 수 있으나, 총 작업 시간 단축으로 전체 에너지 소비는 다를 수 있습니다.

*   **선택 가이드:**
    *   **최고의 추론 속도와 처리량**이 필요하고 전력 소비에 큰 제약이 없다면 **TensorRT(GPU)**가 최적의 선택입니다.
    *   **범용성 및 개발 편의성**을 고려하면서 GPU 가속을 원한다면 **ONNX Runtime(GPU)**이 좋은 대안입니다.
    *   **저전력, 소형 폼팩터의 엣지 환경**에서 AI 추론을 구현해야 한다면 **NPU(Hailo-8L)**가 적합합니다. (단, 이번 분석에서는 NPU의 전력 소비 데이터는 제공되지 않았습니다.)

각 플랫폼은 서로 다른 설계 목표와 최적화 지점을 가지고 있으므로, 특정 애플리케이션의 요구사항(성능, 정확도, 전력, 비용 등)을 종합적으로 고려하여 적합한 하드웨어 및 실행 환경을 선택해야 합니다.