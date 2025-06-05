# all.sh 스크립트 상세 분석

이 문서는 `all.sh` 셸 스크립트의 각 단계별 실행 내용과 해당 스크립트/코드의 상세 설명을 제공합니다.

---

## 스크립트 개요

`all.sh` 스크립트는 머신러닝 모델의 전체 파이프라인을 자동화하여 실행합니다. 이 파이프라인은 모델 학습, 다양한 형식으로의 변환, Hailo AI 프로세서용 최적화 및 컴파일, 그리고 NPU, ONNX Runtime, TensorRT 환경에서의 성능 벤치마킹까지의 과정을 포함합니다.

---

## 단계별 상세 설명

### 1. 모델 학습

#### 1.1. `echo "[1-1] python train_gru.py"`
   - **실행 코드**: `python train_gru.py`
   - **설명**:
     - GRU(Gated Recurrent Unit) 모델을 학습합니다.
     - 일반적으로 시계열 데이터(예: 주가, 센서 데이터 등)를 입력으로 받아 특정 값을 예측하도록 학습됩니다.
     - 학습 과정에는 데이터 로딩, 전처리, 모델 정의, 손실 함수 및 옵티마이저 설정, 학습 루프 실행이 포함됩니다.
     - 학습된 모델의 가중치와 구조는 일반적으로 `.pth` 또는 `.pt` 파일 형식(PyTorch 모델 파일)으로 저장됩니다.
   - **주요 입력**: 학습용 데이터셋 (CSV 파일 등)
   - **주요 출력**: 학습된 GRU 모델 파일 (`.pth`)

#### 1.2. `echo "[1-2] python train_lstm.py"`
   - **실행 코드**: `python train_lstm.py`
   - **설명**:
     - LSTM(Long Short-Term Memory) 모델을 학습합니다.
     - GRU와 유사하게 시계열 데이터 처리에 강점을 가지며, 장기 의존성 학습에 효과적입니다.
     - `train_gru.py`와 유사한 학습 과정을 거칩니다.
   - **주요 입력**: 학습용 데이터셋 (CSV 파일 등)
   - **주요 출력**: 학습된 LSTM 모델 파일 (`.pth`)

#### 1.3. `echo "[1-3] python train_bilstm.py"`
   - **실행 코드**: `python train_bilstm.py`
   - **설명**:
     - BiLSTM(Bidirectional Long Short-Term Memory) 모델을 학습합니다.
     - 양방향 LSTM은 시퀀스 데이터의 과거와 미래 컨텍스트를 모두 활용하여 예측 정확도를 높일 수 있습니다.
     - `train_lstm.py`와 유사한 학습 과정을 거치되, 모델 구조가 양방향으로 구성됩니다.
   - **주요 입력**: 학습용 데이터셋 (CSV 파일 등)
   - **주요 출력**: 학습된 BiLSTM 모델 파일 (`.pth`)

### 2. ONNX 변환

#### `echo "[2] python convert_to_onnx.py"`
   - **실행 코드**: `python convert_to_onnx.py`
   - **설명**:
     - 이전 단계에서 학습된 PyTorch 모델 파일(`.pth`)들을 ONNX(Open Neural Network Exchange) 형식(`.onnx`)으로 변환합니다.
     - ONNX는 서로 다른 딥러닝 프레임워크 간의 모델 공유를 가능하게 하는 개방형 표준입니다.
     - 변환 과정에는 모델 로드, 더미 입력 생성, `torch.onnx.export` 함수 호출 등이 포함될 수 있습니다.
     - 변환된 ONNX 모델은 종종 `onnx-simplifier`와 같은 도구를 사용하여 최적화되거나, 모델 구조를 확인하기 위해 `onnx.checker` 또는 시각화 도구가 사용될 수 있습니다.
   - **주요 입력**: 학습된 PyTorch 모델 파일 (`.pth`)
   - **주요 출력**: ONNX 모델 파일 (`.onnx`)

### 3. ONNX 파일 복사

#### `echo "[3] ./cp_onnx.sh"`
   - **실행 코드**: `./cp_onnx.sh`
   - **설명**:
     - `convert_to_onnx.py`를 통해 생성된 `.onnx` 파일들을 특정 디렉토리로 복사하는 셸 스크립트입니다.
     - 이 디렉토리는 다음 단계(HAR 변환 등)에서 ONNX 파일들을 쉽게 참조할 수 있는 위치이거나, Docker 컨테이너와 호스트 시스템 간의 공유 볼륨일 수 있습니다.
     - 스크립트 내에는 `cp` 명령어가 포함되어 있으며, 대상 디렉토리가 존재하지 않을 경우 생성하거나, 파일이 없을 경우 복사를 건너뛰는 등의 로직이 포함될 수 있습니다.
   - **주요 입력**: `onnx_models` 디렉토리 내의 `.onnx` 파일들
   - **주요 출력**: 지정된 위치로 복사된 `.onnx` 파일들

### 4. HAR 파일 생성 (Hailo Archive)

#### `echo "[4] ./export_to_har.sh"`
   - **실행 코드**: `./export_to_har.sh`
   - **설명**:
     - ONNX 모델 파일(`.onnx`)을 Hailo AI 프로세서에서 사용하기 위한 중간 형식인 HAR(Hailo Archive) 파일로 변환하는 셸 스크립트입니다.
     - 이 과정은 Hailo Dataflow Compiler SDK의 일부 도구(예: `hailo parse`)를 사용합니다.
     - 스크립트는 `onnx_models` 디렉토리 내의 각 `.onnx` 파일을 순회하며 HAR 파일로 변환하는 명령을 실행합니다.
     - HAR 파일은 모델의 네트워크 구조, 가중치, 양자화 정보 등을 포함할 수 있는 아카이브입니다.
   - **주요 입력**: `.onnx` 모델 파일
   - **주요 출력**: HAR 파일 (`.har`)

### 5. HAR 파일 최적화

#### `echo "[5] ./optimize_har.sh"`
   - **실행 코드**: `./optimize_har.sh`
   - **설명**:
     - 이전 단계에서 생성된 HAR 파일들을 Hailo NPU에 맞게 최적화하는 셸 스크립트입니다.
     - 이 최적화 과정에는 모델 양자화(quantization), 연산자 융합(operator fusion), 레이어 최적화 등이 포함될 수 있습니다.
     - Hailo SDK의 `hailo optimize` 명령어가 주로 사용됩니다.
     - 스크립트는 `hailo_har` 디렉토리 내의 각 `.har` 파일을 순회하며 최적화 명령을 실행하고, 최적화된 HAR 파일을 `hailo_har_optimized` 디렉토리에 저장합니다.
   - **주요 입력**: 원본 HAR 파일 (`.har`)
   - **주요 출력**: 최적화된 HAR 파일 (`.har`)

### 6. HEF 파일 컴파일 (Hailo Executable Format)

#### `echo "[6] ./compile_to_hef.sh"`
   - **실행 코드**: `./compile_to_hef.sh`
   - **설명**:
     - 최적화된 HAR 파일을 Hailo AI 프로세서에서 직접 실행할 수 있는 HEF(Hailo Executable Format) 파일로 컴파일하는 셸 스크립트입니다.
     - Hailo SDK의 `hailo compile` 명령어가 사용됩니다.
     - 컴파일 시 다양한 최적화 옵션(예: `compiler_optimization_level=max`)이 적용될 수 있습니다.
     - 스크립트는 `hailo_har_optimized` 디렉토리 내의 각 최적화된 `.har` 파일을 순회하며 HEF 파일로 컴파일하고, 생성된 HEF 파일을 `hef_files` 디렉토리에 저장합니다.
   - **주요 입력**: 최적화된 HAR 파일 (`.har`)
   - **주요 출력**: HEF 파일 (`.hef`)

### 7. HEF 파일 파싱 및 검증

#### `echo "[7] ./parse_hef_files.sh"`
   - **실행 코드**: `./parse_hef_files.sh`
   - **설명**:
     - 생성된 HEF 파일들이 Hailo 하드웨어에서 올바르게 실행될 수 있도록 구성되었는지 파싱하고 검증하는 셸 스크립트입니다.
     - Hailo SDK의 도구를 사용하여 HEF 파일의 내부 구조, 레이어 정보, 예상 성능 등을 확인할 수 있습니다.
     - 이 단계는 컴파일 과정에서 발생할 수 있는 오류를 사전에 감지하고, 모델이 NPU에 적합하게 변환되었는지 확인하는 데 목적이 있습니다.
     - 스크립트는 `hef_files` 디렉토리 내의 각 `.hef` 파일을 순회하며 파싱 및 검증 명령을 실행하고, 그 결과를 로그 파일로 저장하거나 터미널에 출력합니다.
   - **주요 입력**: HEF 파일 (`.hef`)
   - **주요 출력**: 파싱 및 검증 결과 (로그 또는 터미널 출력)

### 8. NPU 성능 측정

#### `echo "[8] python npu_performance_suite9.py"`
   - **실행 코드**: `python npu_performance_suite9.py`
   - **설명**:
     - 컴파일된 HEF 파일을 사용하여 Hailo-8L NPU에서의 실제 모델 추론 성능을 측정하는 Python 스크립트입니다.
     - 측정 항목에는 평균 지연 시간(latency), 초당 처리량(throughput), NPU 사용률 등이 포함됩니다.
     - 스크립트는 지정된 데이터셋을 사용하여 여러 번 추론을 실행하고, 통계적인 성능 지표를 계산하여 JSON 형식 등으로 결과를 저장합니다.
   - **주요 입력**: HEF 파일 (`.hef`), 성능 측정용 데이터셋
   - **주요 출력**: NPU 성능 측정 결과 (JSON 파일)

### 9. ONNX Runtime (GPU) 성능 측정

#### `echo "[9] python onnx_model_2_performance.py"`
   - **실행 코드**: `python onnx_model_2_performance.py`
   - **설명**:
     - 변환된 ONNX 모델(`.onnx`)을 사용하여 ONNX Runtime 환경(주로 GPU 가속을 활용)에서의 추론 성능을 측정하는 Python 스크립트입니다.
     - 측정 항목에는 지연 시간, 처리량, GPU 사용률, GPU 메모리 사용량, GPU 전력 소비량 등이 포함될 수 있습니다.
     - `npu_performance_suite9.py`와 유사하게 데이터셋을 사용하여 추론을 반복 실행하고 결과를 JSON 형식으로 저장합니다.
     - 이 스크립트는 NPU 성능과의 비교를 위해 GPU 환경에서의 베이스라인 성능을 제공합니다.
   - **주요 입력**: ONNX 모델 파일 (`.onnx`), 성능 측정용 데이터셋
   - **주요 출력**: ONNX Runtime (GPU) 성능 측정 결과 (JSON 파일)

### 10. TensorRT (GPU) 성능 측정

#### `echo "[10] python tensorrt_model_2_performance.py"`
    - **실행 코드**: `python tensorrt_model_2_performance.py`
    - **설명**:
      - ONNX 모델을 NVIDIA TensorRT로 최적화하고 컴파일하여 생성된 TensorRT 엔진을 사용하여 NVIDIA GPU에서의 추론 성능을 측정하는 Python 스크립트입니다.
      - TensorRT는 NVIDIA GPU에서 딥러닝 추론 성능을 극대화하기 위한 라이브러리입니다.
      - 측정 항목은 ONNX Runtime 성능 측정과 유사하게 지연 시간, 처리량, GPU 관련 메트릭 등을 포함합니다.
      - 스크립트는 ONNX 파일을 로드하여 TensorRT 엔진을 빌드하거나, 이미 빌드된 엔진 파일(`.engine` 또는 `.plan`)을 로드하여 사용합니다.
      - 결과는 JSON 형식으로 저장되어 다른 플랫폼(NPU, ONNX Runtime)의 성능과 비교 분석하는 데 사용됩니다.
    - **주요 입력**: ONNX 모델 파일 (`.onnx`) 또는 TensorRT 엔진 파일, 성능 측정용 데이터셋
    - **주요 출력**: TensorRT (GPU) 성능 측정 결과 (JSON 파일)

---

이러한 단계들을 통해 `all.sh` 스크립트는 모델 개발부터 다양한 하드웨어 플랫폼에서의 배포 및 성능 평가까지의 복잡한 과정을 자동화합니다.