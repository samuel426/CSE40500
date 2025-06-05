## ONNX Runtime (GPU) vs. TensorRT (GPU) 성능 비교 분석

제공해주신 두 JSON 파일, onnx_performance_20250529_134235.json (ONNX Runtime 실행 결과로 추정)과 tensorrt_performance_20250529_135451.json (TensorRT 실행 결과로 추정)을 비교 분석했습니다. 비교는 `GRU_Apple` 모델을 기준으로 하며, 다른 모델에서도 유사한 경향성을 보일 것으로 예상됩니다.

---

### 1. Latency (지연 시간, 단위: ms) - `GRU_Apple` 모델 기준

| 구분     | ONNX Runtime (GRU_Apple) | TensorRT (GRU_Apple) | TensorRT 우위 (ONNX 대비) |
| :------- | :----------------------- | :------------------- | :------------------------ |
| **mean** | 0.209                    | 0.085                | **약 2.46배 빠름**        |
| **median**| 0.203                    | 0.072                | **약 2.82배 빠름**        |
| **min**  | 0.176                    | 0.067                | **약 2.63배 빠름**        |
| **max**  | 0.355                    | 0.703                | ONNX Runtime이 더 안정적  |
| **std**  | 0.028                    | 0.075                | ONNX Runtime이 더 안정적  |
| **p95**  | 0.262                    | 0.082                | **약 3.20배 빠름**        |
| **p99**  | 0.323                    | 0.396                | ONNX Runtime이 더 안정적  |

*   **분석:**
    *   **평균(mean), 중앙값(median), 최소(min) 및 p95 지연 시간**에서는 TensorRT가 ONNX Runtime보다 약 **2.4배에서 3.2배 더 빠른 속도**를 보여줍니다. 이는 TensorRT가 모델 최적화를 통해 더 효율적인 실행 계획을 생성했기 때문으로 보입니다.
    *   그러나 **최대(max) 지연 시간과 표준편차(std), p99 지연 시간**에서는 ONNX Runtime이 더 낮은 값(더 안정적인 성능)을 보였습니다. TensorRT의 경우, 몇몇 추론에서 상대적으로 큰 지연이 발생했을 가능성이 있습니다. 그럼에도 불구하고 전반적인 평균 및 일반적인 경우(p95)의 성능은 TensorRT가 우수합니다.

---

### 2. Throughput (초당 처리량, 단위: FPS) - `GRU_Apple` 모델 기준

| 구분             | ONNX Runtime (GRU_Apple) | TensorRT (GRU_Apple) | TensorRT 우위 (ONNX 대비) |
| :--------------- | :----------------------- | :------------------- | :------------------------ |
| **throughput_fps** | 4792.56                  | 11725.43             | **약 2.45배 높음**        |

*   **분석:**
    *   Throughput 역시 TensorRT가 ONNX Runtime보다 약 **2.45배 높은 처리량**을 보여, 지연 시간 감소와 일관된 성능 향상을 나타냅니다.

---

### 3. RMSE (Root Mean Squared Error, 예측 오차) - `GRU_Apple` 모델 기준

| 구분   | ONNX Runtime (GRU_Apple) | TensorRT (GRU_Apple) | 비고 (낮을수록 좋음) |
| :----- | :----------------------- | :------------------- | :------------------- |
| **rmse** | 52.14097                 | 52.14097             | **거의 동일**        |

*   **분석:**
    *   `GRU_Apple` 모델의 RMSE 값은 두 실행 환경에서 소수점 다섯째 자리까지 동일하게 나타났습니다. 이는 두 방식 모두 모델의 수학적 정확성을 잘 유지하고 있음을 의미합니다. 다른 모델들 (`GRU_KOSPI`, `GRU_NASDAQ` 등)에서는 RMSE 값에 약간의 차이가 있으나, 전반적으로 큰 차이는 보이지 않습니다. 예를 들어 `GRU_Samsung`의 경우 ONNX는 4689.58, TensorRT는 3595.43으로 TensorRT가 더 낮은 RMSE를 보였지만, `GRU_KOSPI`에서는 ONNX가 61.22, TensorRT가 177.27로 ONNX가 더 우수했습니다. 이는 모델별 최적화 및 부동소수점 연산 차이에서 기인할 수 있습니다.

---

### 4. Memory & Utilization (메모리 및 사용률) - `GRU_Apple` 모델 기준

| 구분                      | ONNX Runtime (GRU_Apple) | TensorRT (GRU_Apple) | 비고                                                                                                |
| :------------------------ | :----------------------- | :------------------- | :-------------------------------------------------------------------------------------------------- |
| **monitor_success**       | true                     | true                 |                                                                                                     |
| **total_lines**           | 1101                     | 1101                 | 동일 데이터셋 사용                                                                                      |
| **avg_device_utilization**| 0.0%                     | 0.0%                 | 두 경우 모두 매우 낮음. `GRU_SnP500` (ONNX: 7.44%), `LSTM_Apple` (TRT: 3.0%) 등 일부 모델에서만 사용률 기록. |
| **device_util_samples**   | 13                       | 1                    | ONNX Runtime 측정 시 더 많은 사용률 샘플 수집. TensorRT는 초기 스냅샷만 기록된 것으로 보임.                  |
| **device_util_values**    | `[0.0, 0.0, ...]`        | `[0.0]`              |                                                                                                     |
| **avg_model_fps**         | 4792.56                  | 11725.43             | Throughput과 동일                                                                                     |
| **avg_model_utilization** | 0.0%                     | 0.0%                 | `avg_device_utilization`과 동일한 경향                                                                |
| **utilization_samples**   | 13                       | 1                    |                                                                                                     |
| **utilization_values**    | `[0.0, 0.0, ...]`        | `[0.0]`              |                                                                                                     |

*   **분석:**
    *   대부분의 모델에서 `GRU_Apple`과 유사하게 GPU 사용률(`avg_device_utilization`, `avg_model_utilization`)이 0%로 기록되었습니다. 이는 테스트된 모델들이 RTX 3070 Ti의 전체 용량에 비해 매우 가벼워, 추론 중 GPU 부하가 거의 발생하지 않았음을 시사합니다.
    *   ONNX Runtime 결과에서는 `device_util_samples`가 TensorRT 결과보다 많게 나타나, 모니터링 스레드가 더 많은 데이터를 수집했음을 알 수 있습니다. TensorRT의 경우, `utilization_samples`가 1로 기록된 것은 초기 스냅샷 값만 사용되었을 가능성을 나타냅니다. (이전 스크립트 수정에서 TensorRT의 모니터링 로직이 NPU 스크립트와 유사하게 변경되었으므로, 이 부분은 스크립트 실행 시점의 설정이나 실제 부하에 따라 달라질 수 있습니다.)
    *   일부 모델(예: ONNX의 `GRU_SnP500`에서 7.44%, TensorRT의 `LSTM_Apple`에서 3.0%)에서는 약간의 GPU 사용률이 측정되었으나, 여전히 낮은 수준입니다.

---

### 종합 결론

*   **성능 (Latency & Throughput):** 전반적으로 **TensorRT가 ONNX Runtime보다 더 나은 추론 성능(더 낮은 평균 지연 시간 및 더 높은 처리량)을 제공**합니다. 이는 TensorRT의 모델 최적화 기능 덕분입니다. `GRU_Apple` 모델 기준 약 2.4배의 성능 향상이 관찰되었습니다.
*   **안정성:** 일부 지표(max latency, std, p99)에서는 ONNX Runtime이 더 안정적인 모습을 보였으나, 이는 특정 실행 환경이나 워크로드에 따른 특이값의 영향일 수 있습니다. 평균적인 성능은 TensorRT가 우세합니다.
*   **정확도 (RMSE):** 모델에 따라 약간의 차이는 있지만, 두 실행 환경 간에 전반적으로 큰 정확도 차이는 없습니다.
*   **GPU 사용률:** 두 환경 모두에서 대부분의 모델에 대해 GPU 사용률이 매우 낮게 측정되었습니다. 이는 사용된 GPU(RTX 3070 Ti로 추정)의 성능이 해당 모델들의 연산 요구량에 비해 매우 높다는 것을 의미합니다. TensorRT의 사용률 샘플 수가 적게 나온 것은 측정 방식의 차이 또는 실제 부하가 거의 없었기 때문일 수 있습니다.

결론적으로, NVIDIA GPU에서 최상의 추론 성능을 얻기 위해서는 **TensorRT를 사용하는 것이 유리**합니다. 다만, ONNX Runtime도 충분히 좋은 성능을 제공하며, 다양한 하드웨어 및 플랫폼 간의 이식성 측면에서는 더 넓은 호환성을 가질 수 있습니다. 사용률이 매우 낮은 점을 고려할 때, 해당 모델들은 더 저사양의 GPU나 특화된 AI 가속기에서도 충분히 효율적으로 실행될 수 있음을 시사합니다.