## NPU, ONNX Runtime(GPU), TensorRT(GPU) 성능 비교 분석 (`LSTM_Apple` 모델)

제공된 3개의 JSON 파일 ([`npu_performance_20250528_132327.json`](npu_performance_20250528_132327.json ), [`onnx_performance_20250529_134235.json`](onnx_performance_20250529_134235.json ), [`tensorrt_performance_20250529_135451.json`](tensorrt_performance_20250529_135451.json ))을 기반으로 NPU(Hailo-8L), ONNX Runtime(GPU), TensorRT(GPU) 환경에서의 `LSTM_Apple` 모델 성능을 비교 분석합니다.

---

### 1. Latency (지연 시간, 단위: ms) - `LSTM_Apple` 모델 기준

| 구분     | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) | TensorRT 우위 (NPU 대비) | TensorRT 우위 (ONNX 대비) |
| :------- | :------------- | :----------------- | :------------- | :----------------------- | :------------------------ |
| **mean** | 1.330          | 0.182              | 0.047          | **약 28.3배 빠름**       | **약 3.87배 빠름**        |
| **median**| 1.324          | 0.162              | 0.041          | **약 32.3배 빠름**       | **약 3.95배 빠름**        |
| **min**  | 1.278          | 0.159              | 0.038          | **약 33.6배 빠름**       | **약 4.18배 빠름**        |
| **max**  | 1.413          | 0.630              | 0.316          | **약 4.5배 빠름**        | **약 1.99배 빠름**        |
| **std**  | 0.031          | 0.072              | 0.038          | NPU가 더 안정적         | TensorRT가 더 안정적      |
| **p95**  | 1.397          | 0.222              | 0.049          | **약 28.5배 빠름**       | **약 4.53배 빠름**        |
| **p99**  | 1.409          | 0.624              | 0.304          | **약 4.6배 빠름**        | **약 2.05배 빠름**        |

*   **분석:**
    *   `LSTM_Apple` 모델에서도 **TensorRT가 모든 지연 시간 지표에서 가장 우수한 성능**을 보였습니다. NPU 대비 평균 약 28배, ONNX Runtime 대비 평균 약 3.9배 더 빠릅니다.
    *   ONNX Runtime 역시 NPU에 비해 평균적으로 약 7.3배 빠른 지연 시간을 제공합니다.
    *   표준편차(std)의 경우, NPU가 가장 작은 값을 보여 지연 시간의 변동성이 가장 적었고, 그 다음이 TensorRT, ONNX Runtime 순이었습니다. 이는 NPU가 일관된 속도로 처리했음을 의미하며, GPU 플랫폼에서는 일부 특이값의 영향이 있었을 수 있습니다.

---

### 2. Throughput (초당 처리량, 단위: FPS) - `LSTM_Apple` 모델 기준

| 구분             | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) | TensorRT 우위 (NPU 대비) | TensorRT 우위 (ONNX 대비) |
| :--------------- | :------------- | :----------------- | :------------- | :----------------------- | :------------------------ |
| **throughput_fps** | 751.71         | 5505.20            | 21185.49       | **약 28.2배 높음**       | **약 3.85배 높음**        |

*   **분석:**
    *   Throughput 역시 TensorRT가 압도적으로 높아, NPU 대비 약 28.2배, ONNX Runtime 대비 약 3.85배 더 많은 데이터를 초당 처리할 수 있습니다.
    *   ONNX Runtime은 NPU 대비 약 7.3배 높은 처리량을 보입니다.
    *   NPU JSON 파일의 `memory.avg_model_fps` (28.86 FPS)는 `GRU_Apple`과 마찬가지로 레이턴시 기반의 전체 `throughput_fps`와는 다른 지표이므로 직접 비교에는 주의가 필요합니다.

---

### 3. RMSE (Root Mean Squared Error, 예측 오차) - `LSTM_Apple` 모델 기준

| 구분   | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) | 비고 (낮을수록 좋음) |
| :----- | :------------- | :----------------- | :------------- | :------------------- |
| **rmse** | 64.9501505     | 64.9501505         | 64.9501507     | **거의 동일**        |

*   **분석:**
    *   `LSTM_Apple` 모델의 RMSE 값은 세 플랫폼 모두에서 소수점 여섯째 자리까지 거의 동일하게 나타났습니다. 이는 세 환경 모두 모델의 수학적 정확성을 매우 잘 유지하고 있음을 의미합니다.

---

### 4. Utilization (평균 사용률, %) - `LSTM_Apple` 모델 기준

| 구분                      | NPU (Hailo-8L) | ONNX Runtime (GPU) | TensorRT (GPU) | 비고                                                              |
| :------------------------ | :------------- | :----------------- | :------------- | :---------------------------------------------------------------- |
| **avg_device_utilization**| 14.13%         | 0.0%               | 3.0%           | NPU와 TensorRT는 사용률 기록, ONNX Runtime은 0%                     |
| **avg_model_utilization** | 14.13%         | 0.0%               | 3.0%           | `avg_device_utilization`과 동일 (NPU는 device=model, GPU는 전체 GPU) |

*   **분석:**
    *   NPU는 `LSTM_Apple` 모델 실행 시 평균 약 14.13%의 사용률을 보였습니다.
    *   TensorRT 환경에서는 평균 3.0%의 GPU 사용률이 기록되었습니다. 이는 `GRU_Apple` 모델보다는 `LSTM_Apple` 모델이 GPU 자원을 조금 더 활용했음을 나타냅니다.
    *   ONNX Runtime 환경에서는 여전히 GPU 사용률이 0.0%로 기록되었습니다. 이는 ONNX Runtime의 측정 방식 또는 실제 부하가 TensorRT 실행 시보다 낮았을 가능성을 시사합니다.
    *   세 플랫폼 모두 `LSTM_Apple` 모델에 대해서도 GPU/NPU의 전체 용량 대비 낮은 사용률을 보이고 있어, 해당 하드웨어들이 이 모델에 대해 충분한 처리 능력을 가지고 있음을 알 수 있습니다.

---

### 종합 결론 (`LSTM_Apple` 모델)

*   **성능 (Latency & Throughput):**
    *   `LSTM_Apple` 모델에서도 **TensorRT(GPU)**가 세 플랫폼 중 가장 뛰어난 추론 성능을 제공합니다. NPU 대비 약 28배, ONNX Runtime 대비 약 3.9배의 성능 향상을 보입니다.
    *   **ONNX Runtime(GPU)** 역시 NPU에 비해 약 7.3배 우수한 성능을 나타냅니다.
    *   NPU는 절대적인 속도에서는 GPU 플랫폼에 비해 느리지만, 일관된 지연 시간(낮은 std)을 보여주었습니다.

*   **정확도 (RMSE):**
    *   세 플랫폼 모두 `LSTM_Apple` 모델에 대해 거의 동일한 RMSE 값을 보여, 모델 정확성을 잘 유지하고 있습니다.

*   **자원 사용률:**
    *   NPU와 TensorRT는 `LSTM_Apple` 모델 실행 중 각각 14.13%, 3.0%의 사용률을 기록했습니다. ONNX Runtime은 0%로 측정되었습니다. 이는 `LSTM_Apple` 모델이 `GRU_Apple` 모델보다는 GPU/NPU 자원을 조금 더 사용하지만, 여전히 해당 하드웨어의 전체 용량에는 크게 미치지 못함을 의미합니다.

*   **선택 가이드 (LSTM_Apple 모델 기준):**
    *   `GRU_Apple` 모델과 마찬가지로, **최고의 추론 속도**가 필요하다면 **TensorRT(GPU)**가 가장 적합합니다.
    *   **범용성과 준수한 GPU 가속 성능**을 원한다면 **ONNX Runtime(GPU)**을 고려할 수 있습니다.
    *   **엣지 환경에서의 저전력 운영과 일관된 성능**이 중요하다면 **NPU(Hailo-8L)**가 적합한 선택입니다.

`LSTM_Apple` 모델의 경우에도 `GRU_Apple` 모델과 유사한 성능 경향성을 보이며, TensorRT가 가장 높은 성능을, NPU가 가장 높은 전력 효율성(추정)과 일관성을 제공하는 것으로 분석됩니다.