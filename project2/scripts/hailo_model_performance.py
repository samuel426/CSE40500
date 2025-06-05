#!/usr/bin/env python3

import numpy as np
import csv
import time
import json
import os
import subprocess
import threading
import queue
import select
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import re

from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, 
                           HailoStreamInterface, ConfigureParams,
                           InputVStreamParams, OutputVStreamParams,
                           FormatType, InferVStreams)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NPUPerformanceMeasurer:
    def __init__(self, models_dir: str, data_dir: str, results_dir: str = "results"):
        """
        NPU 성능 측정 클래스
        
        Args:
            models_dir: .hef 파일들이 있는 디렉토리
            data_dir: CSV 데이터 파일들이 있는 디렉토리  
            results_dir: 결과를 저장할 디렉토리
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 종목 및 모델 정의
        self.stocks = ["Apple", "KOSPI", "NASDAQ", "Samsung", "Tesla", "SnP500"]
        self.model_types = ["GRU", "LSTM"]
        
        # 측정 설정
        self.warmup_iterations = 10
        self.measurement_iterations = 50
        self.batch_size = 1
        
        # 결과 저장용
        self.results = {}
        
    def preprocess_csv_data(self, csv_path: str, sequence_length: int = 8) -> np.ndarray:
        """
        CSV 데이터를 NPU 입력 형태로 전처리
        
        Args:
            csv_path: CSV 파일 경로
            sequence_length: 시퀀스 길이 (기본값 8)
            
        Returns:
            전처리된 데이터 배열
        """
        try:
            logger.info(f"Reading CSV file: {csv_path}")
            
            data_rows = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # 처음 3줄 헤더 스킵
                headers1 = next(reader)  # Price,Open,High,Low,Close,Volume
                headers2 = next(reader)  # Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
                headers3 = next(reader)  # Date,,,,,
                
                logger.info(f"CSV headers: {headers1}")
                logger.info(f"Skipped lines: {headers2}, {headers3}")
                
                row_count = 0
                for row in reader:
                    if row_count < 3:  # 처음 3행 로그로 확인
                        logger.info(f"Sample row {row_count}: {row}")
                    
                    try:
                        # 첫 번째 컬럼은 날짜이므로 스킵하고, 나머지 5개 컬럼(OHLCV) 사용
                        if len(row) >= 6:  # 날짜 + OHLCV 5개
                            # 날짜 컬럼(index 0) 스킵하고 OHLCV 데이터(index 1-5) 추출
                            ohlcv_values = []
                            for i in range(1, 6):  # Open, High, Low, Close, Volume
                                if i < len(row):
                                    try:
                                        value = float(row[i].strip())
                                        ohlcv_values.append(value)
                                    except (ValueError, AttributeError):
                                        logger.warning(f"Cannot convert '{row[i]}' to float in row {row_count}")
                                        break
                                else:
                                    break
                            
                            # 5개 값이 모두 정상적으로 추출되었으면 추가
                            if len(ohlcv_values) == 5:
                                data_rows.append(ohlcv_values)
                            
                        row_count += 1
                        if row_count >= 2000:  # 충분한 데이터만 사용
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error parsing row {row_count}: {e}")
                        continue
            
            if not data_rows:
                logger.error(f"No valid rows found in {csv_path}")
                # 파일 내용 일부 출력
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    for i, line in enumerate(lines):
                        logger.error(f"Line {i}: {line.strip()}")
                raise ValueError(f"No valid numeric data found in {csv_path}")
            
            logger.info(f"Successfully loaded {len(data_rows)} rows from {csv_path}")
            logger.info(f"Sample data (first row): {data_rows[0] if data_rows else 'None'}")
            logger.info(f"Sample data (last row): {data_rows[-1] if data_rows else 'None'}")
            
            # numpy 배열로 변환
            ohlcv_data = np.array(data_rows, dtype=np.float32)
            
            # 정규화 (TODO: GPU 학습시 사용한 scaler 파라미터 적용 필요)
            # 임시로 MinMax 정규화 사용 (0-1 범위)
            data_min = np.min(ohlcv_data, axis=0, keepdims=True)
            data_max = np.max(ohlcv_data, axis=0, keepdims=True)
            
            logger.info(f"Data range - Min: {data_min.flatten()}, Max: {data_max.flatten()}")
            
            # Zero division 방지
            data_range = data_max - data_min
            data_range[data_range == 0] = 1.0
            
            normalized_data = (ohlcv_data - data_min) / data_range
            
            # UINT8 변환 (0-255 범위)
            uint8_data = (normalized_data * 255).astype(np.uint8)
            
            # 시퀀스 생성 (sliding window)
            sequences = []
            for i in range(sequence_length, len(uint8_data)):
                seq = uint8_data[i-sequence_length:i]  # (8, 5)
                sequences.append(seq)
            
            logger.info(f"Generated {len(sequences)} sequences of shape (8, 5)")
            return np.array(sequences), data_min, data_max
            
        except Exception as e:
            logger.error(f"Error preprocessing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def create_gru_inputs(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GRU 모델용 입력 데이터 생성
        
        Args:
            sequences: 전처리된 시퀀스 데이터
            
        Returns:
            (input_x, input_h0) 튜플
        """
        # input_x: (1, 8, 5) -> (1, 1, 8, 5)
        input_x = sequences[0:1].reshape(1, 1, 8, 5)
        
        # input_h0: 영벡터 초기화 (1, 1, 5)
        input_h0 = np.zeros((1, 1, 1, 5), dtype=np.uint8)
        
        return input_x, input_h0
    
    def create_lstm_inputs(self, sequences: np.ndarray) -> np.ndarray:
        """
        LSTM 모델용 입력 데이터 생성
        
        Args:
            sequences: 전처리된 시퀀스 데이터
            
        Returns:
            input_x 배열
        """
        # input_x: (1, 8, 5) -> (1, 1, 8, 5)
        input_x = sequences[0:1].reshape(1, 1, 8, 5)
        return input_x
    
    def measure_latency(self, hef_path: str, input_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        레이턴시 측정
        
        Args:
            hef_path: HEF 파일 경로
            input_data: 입력 데이터 딕셔너리
            
        Returns:
            레이턴시 통계
        """
        try:
            hef = HEF(hef_path)
            
            # VDevice 생성
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            
            with VDevice(params) as vdevice:
                # 모델 구성
                infer_model = vdevice.create_infer_model(hef_path)
                infer_model.set_batch_size(self.batch_size)
                
                with infer_model.configure() as configured_model:
                    # Bindings 생성
                    bindings = configured_model.create_bindings()
                    
                    # 입력 버퍼 설정
                    for input_name in infer_model.input_names:
                        if input_name in input_data:
                            bindings.input(input_name).set_buffer(input_data[input_name])
                        else:
                            logger.warning(f"Input {input_name} not found in input_data")
                    
                    # 출력 버퍼 설정
                    for output_name in infer_model.output_names:
                        output_stream = infer_model.output(output_name)
                        shape = output_stream.shape
                        buffer = np.empty(shape, dtype=np.uint8)
                        bindings.output(output_name).set_buffer(buffer)
                    
                    # 레이턴시 측정
                    latencies = []
                    timeout_ms = 1000
                    
                    # Warmup
                    logger.info(f"Warming up with {self.warmup_iterations} iterations...")
                    for _ in range(self.warmup_iterations):
                        configured_model.run([bindings], timeout_ms)
                    
                    # 실제 측정
                    logger.info(f"Measuring latency over {self.measurement_iterations} iterations...")
                    for i in range(self.measurement_iterations):
                        start = time.perf_counter()
                        configured_model.run([bindings], timeout_ms)
                        end = time.perf_counter()
                        
                        latencies.append((end - start) * 1000)  # ms 변환
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Progress: {i + 1}/{self.measurement_iterations}")
                    
                    # 통계 계산
                    latencies = np.array(latencies)
                    return {
                        "mean": float(np.mean(latencies)),
                        "median": float(np.median(latencies)),
                        "min": float(np.min(latencies)),
                        "max": float(np.max(latencies)),
                        "std": float(np.std(latencies)),
                        "p95": float(np.percentile(latencies, 95)),
                        "p99": float(np.percentile(latencies, 99))
                    }
        
        except Exception as e:
            logger.error(f"Error measuring latency for {hef_path}: {e}")
            return None
    
    def measure_throughput_infer_model(self, hef_path: str, input_data: Dict[str, np.ndarray], 
                                      duration: int = 10) -> Optional[float]:
        """
        InferModel API를 사용한 스루풋(FPS) 측정
        
        Args:
            hef_path: HEF 파일 경로
            input_data: 입력 데이터 딕셔너리
            duration: 측정 지속 시간 (초)
            
        Returns:
            FPS 값
        """
        try:
            hef = HEF(hef_path)
            
            # VDevice 생성
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            
            with VDevice(params) as vdevice:
                # 모델 구성
                infer_model = vdevice.create_infer_model(hef_path)
                infer_model.set_batch_size(self.batch_size)
                
                with infer_model.configure() as configured_model:
                    # Bindings 생성
                    bindings = configured_model.create_bindings()
                    
                    # 입력 버퍼 설정
                    for input_name in infer_model.input_names:
                        if input_name in input_data:
                            bindings.input(input_name).set_buffer(input_data[input_name])
                    
                    # 출력 버퍼 설정
                    for output_name in infer_model.output_names:
                        output_stream = infer_model.output(output_name)
                        shape = output_stream.shape
                        buffer = np.empty(shape, dtype=np.uint8)
                        bindings.output(output_name).set_buffer(buffer)
                    
                    # Throughput 측정
                    timeout_ms = 1000
                    
                    # Warmup
                    logger.info("Warming up for throughput measurement...")
                    for _ in range(20):
                        configured_model.run([bindings], timeout_ms)
                    
                    # 실제 측정
                    logger.info(f"Measuring throughput for {duration} seconds...")
                    start_time = time.time()
                    inference_count = 0
                    
                    while time.time() - start_time < duration:
                        configured_model.run([bindings], timeout_ms)
                        inference_count += 1
                        
                        # 1초마다 진행상황 출력
                        elapsed = time.time() - start_time
                        if inference_count % 100 == 0:
                            current_fps = inference_count / elapsed
                            logger.info(f"  {elapsed:.1f}s: {current_fps:.1f} FPS")
                    
                    total_time = time.time() - start_time
                    fps = inference_count / total_time
                    
                    logger.info(f"Completed {inference_count} inferences in {total_time:.2f}s")
                    return fps
        
        except Exception as e:
            logger.error(f"Error measuring throughput for {hef_path}: {e}")
            return None
    
    def start_memory_monitor(self, duration: int = 60) -> queue.Queue:
        """
        메모리 모니터링 시작 (개선된 버전)
        
        Args:
            duration: 모니터링 지속 시간 (초)
            
        Returns:
            모니터링 결과를 담을 큐
        """
        monitor_queue = queue.Queue()
        
        def monitor_thread():
            try:
                # 환경변수 설정 (기존 환경변수 복사 + HAILO_MONITOR 추가)
                env = os.environ.copy()
                env['HAILO_MONITOR'] = '1'
                env['HAILO_MONITOR_TIME_INTERVAL'] = '500'  # 500ms 간격
                
                logger.info("Starting hailortcli monitor...")
                
                # hailortcli monitor 실행
                cmd = ["hailortcli", "monitor"]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    universal_newlines=True,
                    bufsize=1  # 라인 버퍼링
                )
                
                start_time = time.time()
                output_lines = []
                error_lines = []
                
                logger.info(f"Monitor started, PID: {process.pid}")
                
                # Non-blocking 읽기로 변경
                while time.time() - start_time < duration:
                    try:
                        # stdout에서 읽기 (timeout 1초)
                        ready_stdout, _, _ = select.select([process.stdout], [], [], 1.0)
                        
                        if ready_stdout:
                            line = process.stdout.readline()
                            if line:
                                line = line.strip()
                                if line:  # 빈 줄 제외
                                    output_lines.append(line)
                                    logger.debug(f"Monitor output: {line}")
                        
                        # stderr 확인
                        ready_stderr, _, _ = select.select([process.stderr], [], [], 0.1)
                        if ready_stderr:
                            error_line = process.stderr.readline()
                            if error_line:
                                error_lines.append(error_line.strip())
                                logger.warning(f"Monitor error: {error_line.strip()}")
                        
                        # 프로세스 종료 확인
                        if process.poll() is not None:
                            logger.warning(f"Monitor process terminated early with code: {process.poll()}")
                            break
                            
                    except Exception as e:
                        logger.error(f"Error reading monitor output: {e}")
                        break
                
                # 프로세스 종료
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Monitor process did not terminate, killing...")
                    process.kill()
                    process.wait()
                
                logger.info(f"Monitor finished. Captured {len(output_lines)} output lines, {len(error_lines)} error lines")
                
                if error_lines:
                    logger.error(f"Monitor errors: {error_lines[:5]}")  # 처음 5개 에러만 로그
                
                monitor_queue.put({
                    'output': output_lines,
                    'errors': error_lines,
                    'success': len(output_lines) > 0
                })
                
            except Exception as e:
                logger.error(f"Memory monitoring thread error: {e}")
                import traceback
                traceback.print_exc()
                monitor_queue.put({
                    'output': [],
                    'errors': [str(e)],
                    'success': False
                })
        
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
        
        return monitor_queue
    
    def parse_monitor_output(self, monitor_data: Dict) -> Dict[str, float]:
        """
        모니터 출력 파싱 (테이블 구조 기반)
        
        Args:
            monitor_data: 모니터링 결과 딕셔너리
            
        Returns:
            파싱된 메모리/성능 정보
        """
        if not monitor_data.get('success', False):
            logger.warning(f"Monitor failed: {monitor_data.get('errors', [])}")
            return {
                "monitor_success": False,
                "error_message": str(monitor_data.get('errors', ['Unknown error'])[0]) if monitor_data.get('errors') else "No data"
            }
        
        monitor_lines = monitor_data.get('output', [])
        device_utilization = []
        model_fps = []
        model_utilization = []
        
        logger.info(f"Parsing {len(monitor_lines)} monitor lines...")
        
        # 샘플 로그 출력 (처음 10줄)
        for i, line in enumerate(monitor_lines[:10]):
            logger.info(f"Monitor line {i}: {line}")
        
        parsing_device_table = False
        parsing_model_table = False
        
        for line in monitor_lines:
            line = line.strip()
            
            # 테이블 섹션 감지
            if "Device ID" in line and "Utilization" in line:
                parsing_device_table = True
                parsing_model_table = False
                continue
            elif "Model" in line and "Utilization" in line and "FPS" in line:
                parsing_device_table = False
                parsing_model_table = True
                continue
            elif "Stream" in line and "Direction" in line:
                parsing_device_table = False
                parsing_model_table = False
                continue
            elif line.startswith("---"):
                continue
            
            # Device 테이블 파싱
            if parsing_device_table and line and not line.startswith("Device"):
                # 형태: "0000:01:00.0                                                75.1                     HAILO8L"
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # 두 번째 컬럼이 utilization
                        util = float(parts[1])
                        if 0 <= util <= 100:
                            device_utilization.append(util)
                            logger.debug(f"Device utilization: {util}%")
                    except (ValueError, IndexError):
                        continue
            
            # Model 테이블 파싱  
            elif parsing_model_table and line and not line.startswith("Model"):
                # 형태: "GRU_Apple                                                   75.1                     765.4          4823"
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # 두 번째 컬럼: Model utilization, 세 번째 컬럼: FPS
                        model_util = float(parts[1])
                        fps = float(parts[2])
                        
                        if 0 <= model_util <= 100:
                            model_utilization.append(model_util)
                            logger.debug(f"Model utilization: {model_util}%")
                        
                        if 0 < fps < 10000:
                            model_fps.append(fps)
                            logger.debug(f"Model FPS: {fps}")
                            
                    except (ValueError, IndexError):
                        continue
            
            # 추가: 정규식으로 숫자만 찾기 (백업 방법)
            else:
                # 백분율 찾기
                percentages = re.findall(r'(\d+\.?\d*)\s*%', line)
                for p in percentages:
                    try:
                        util = float(p)
                        if 0 <= util <= 100:
                            if "device" in line.lower() or "0000:" in line:
                                device_utilization.append(util)
                            else:
                                model_utilization.append(util)
                    except ValueError:
                        continue
                
                # FPS 패턴 찾기 (숫자 뒤에 공백이나 탭이 있는 경우)
                fps_matches = re.findall(r'(\d+\.?\d+)\s+\d+', line)  # 숫자 뒤에 PID가 오는 패턴
                for fps_str in fps_matches:
                    try:
                        fps = float(fps_str)
                        if 0 < fps < 10000:
                            model_fps.append(fps)
                    except ValueError:
                        continue
        
        # 통계 계산
        result = {
            "monitor_success": True,
            "total_lines": len(monitor_lines),
            
            # Device 통계
            "avg_device_utilization": float(np.mean(device_utilization)) if device_utilization else 0.0,
            "max_device_utilization": float(np.max(device_utilization)) if device_utilization else 0.0,
            "min_device_utilization": float(np.min(device_utilization)) if device_utilization else 0.0,
            "device_util_samples": len(device_utilization),
            "device_util_values": device_utilization[:10],  # 처음 10개 값 저장
            
            # Model FPS 통계
            "avg_model_fps": float(np.mean(model_fps)) if model_fps else 0.0,
            "max_model_fps": float(np.max(model_fps)) if model_fps else 0.0,
            "min_model_fps": float(np.min(model_fps)) if model_fps else 0.0,
            "fps_samples": len(model_fps),
            "fps_values": model_fps[:10],  # 처음 10개 값 저장
            
            # Model Utilization 통계
            "avg_model_utilization": float(np.mean(model_utilization)) if model_utilization else 0.0,
            "max_model_utilization": float(np.max(model_utilization)) if model_utilization else 0.0,
            "min_model_utilization": float(np.min(model_utilization)) if model_utilization else 0.0,
            "utilization_samples": len(model_utilization),
            "utilization_values": model_utilization[:10]  # 처음 10개 값 저장
        }
        
        logger.info(f"Parsed monitor data successfully:")
        logger.info(f"  Device utilization samples: {len(device_utilization)} (avg: {result['avg_device_utilization']:.1f}%)")
        logger.info(f"  Model FPS samples: {len(model_fps)} (avg: {result['avg_model_fps']:.1f})")
        logger.info(f"  Model utilization samples: {len(model_utilization)} (avg: {result['avg_model_utilization']:.1f}%)")
        
        return result
    
    def calculate_rmse(self, predictions: np.ndarray, actual_data: np.ndarray, 
                      data_min: np.ndarray, data_max: np.ndarray) -> float:
        """
        RMSE 계산
        
        Args:
            predictions: NPU 예측 결과
            actual_data: 실제 데이터
            data_min, data_max: 정규화 파라미터
            
        Returns:
            RMSE 값
        """
        try:
            # UINT8 -> 정규화된 값으로 변환
            pred_normalized = predictions.astype(np.float32) / 255.0
            
            # 원본 스케일로 복원
            pred_original = pred_normalized * (data_max - data_min) + data_min
            
            # 실제값과 비교 (Close 가격만 비교)
            actual_close = actual_data[:len(pred_original), 3]  # Close는 4번째 컬럼
            pred_close = pred_original[:, 3] if pred_original.ndim > 1 else pred_original
            
            # RMSE 계산
            mse = np.mean((actual_close - pred_close) ** 2)
            rmse = np.sqrt(mse)
            
            return float(rmse)
            
        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            return None
    
    def measure_single_model(self, stock: str, model_type: str) -> Dict:
        """
        단일 모델 성능 측정
        
        Args:
            stock: 종목명
            model_type: 모델 타입 (GRU/LSTM)
            
        Returns:
            측정 결과 딕셔너리
        """
        logger.info(f"Measuring {model_type}_{stock}...")
        
        # 파일 경로 설정
        hef_path = self.models_dir / f"{model_type}_{stock}.hef"
        csv_path = self.data_dir / f"{stock}_actual_ohlcv.csv"
        
        if not hef_path.exists():
            logger.error(f"HEF file not found: {hef_path}")
            return None
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return None
        
        # 데이터 전처리
        sequences, data_min, data_max = self.preprocess_csv_data(str(csv_path))
        if sequences is None:
            return None
        
        # 입력 데이터 준비
        if model_type == "GRU":
            input_x, input_h0 = self.create_gru_inputs(sequences)
            input_data = {
                f"{model_type}_{stock}/input_layer1": input_x,
                f"{model_type}_{stock}/input_layer2": input_h0
            }
        else:  # LSTM
            input_x = self.create_lstm_inputs(sequences)
            input_data = {f"{model_type}_{stock}/input_layer1": input_x}
        
        # 1. 레이턴시 측정
        logger.info("=== Measuring Latency ===")
        latency_stats = self.measure_latency(str(hef_path), input_data)
        
        # 2. 스루풋 측정 (새로운 방식)
        logger.info("=== Measuring Throughput ===")
        throughput = self.measure_throughput_infer_model(str(hef_path), input_data, duration=10)
        
        # 3. 메모리 모니터링 (추론과 동시에)
        logger.info("=== Measuring Memory Usage ===")
        
        # 메모리 모니터링 시작
        monitor_queue = self.start_memory_monitor(duration=15)
        time.sleep(1)  # 모니터 시작 대기
        
        # 추론을 약간 실행하여 메모리 사용량 측정
        try:
            hef = HEF(str(hef_path))
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            
            with VDevice(params) as vdevice:
                infer_model = vdevice.create_infer_model(str(hef_path))
                infer_model.set_batch_size(self.batch_size)
                
                with infer_model.configure() as configured_model:
                    bindings = configured_model.create_bindings()
                    
                    # 입력 버퍼 설정
                    for input_name in infer_model.input_names:
                        if input_name in input_data:
                            bindings.input(input_name).set_buffer(input_data[input_name])
                    
                    # 출력 버퍼 설정
                    for output_name in infer_model.output_names:
                        output_stream = infer_model.output(output_name)
                        shape = output_stream.shape
                        buffer = np.empty(shape, dtype=np.uint8)
                        bindings.output(output_name).set_buffer(buffer)
                    
                    # 메모리 사용량 측정을 위해 추론 실행
                    logger.info("Running inference for memory monitoring...")
                    for i in range(100):
                        configured_model.run([bindings], 1000)
                        if i % 20 == 0:
                            logger.info(f"  Memory test progress: {i}/100")
                        time.sleep(0.05)  # 모니터링 시간 확보
                    
        except Exception as e:
            logger.error(f"Error during memory monitoring: {e}")
        
        # 메모리 모니터링 결과 수집
        time.sleep(3)  # 모니터링 완료 대기
        try:
            monitor_data = monitor_queue.get(timeout=10)
            memory_stats = self.parse_monitor_output(monitor_data)
            logger.info(f"Memory monitoring success: {monitor_data.get('success', False)}")
        except queue.Empty:
            logger.warning("Memory monitoring queue timeout")
            memory_stats = {"monitor_success": False, "error_message": "Queue timeout"}
        
        # RMSE 계산 (단순화된 버전)
        # TODO: 실제 NPU 추론 결과로 RMSE 계산
        rmse = None  # 추후 구현
        
        # 결과 정리
        result = {
            "model": f"{model_type}_{stock}",
            "timestamp": datetime.now().isoformat(),
            "latency": latency_stats,
            "throughput_fps": throughput,
            "memory": memory_stats,
            "rmse": rmse,
            "data_samples": len(sequences)
        }
        
        logger.info(f"=== Results for {model_type}_{stock} ===")
        if latency_stats:
            logger.info(f"  Latency: {latency_stats['mean']:.2f}ms (±{latency_stats['std']:.2f})")
        if throughput:
            logger.info(f"  Throughput: {throughput:.1f} FPS")
        if memory_stats:
            logger.info(f"  Memory: {memory_stats}")
        
        return result
    
    def run_all_measurements(self):
        """
        모든 모델에 대해 성능 측정 실행
        """
        logger.info("Starting NPU performance measurements...")
        
        total_models = len(self.stocks) * len(self.model_types)
        current = 0
        
        for stock in self.stocks:
            for model_type in self.model_types:
                current += 1
                logger.info(f"Progress: {current}/{total_models}")
                
                result = self.measure_single_model(stock, model_type)
                if result:
                    self.results[f"{model_type}_{stock}"] = result
                
                # 모델 간 쿨다운
                time.sleep(3)
        
        # 결과 저장
        self.save_results()
        logger.info("All measurements completed!")
    
    def save_results(self):
        """
        측정 결과를 JSON 파일로 저장
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"npu_performance_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_file}")
        
        # 요약 출력
        self.print_summary()
    
    def print_summary(self):
        """
        측정 결과 요약 출력
        """
        print("\n" + "="*70)
        print("NPU PERFORMANCE MEASUREMENT SUMMARY")
        print("="*70)
        
        gru_results = {}
        lstm_results = {}
        
        # 결과 분류
        for model_name, result in self.results.items():
            if result and 'GRU' in model_name:
                gru_results[model_name] = result
            elif result and 'LSTM' in model_name:
                lstm_results[model_name] = result
        
        # GRU 결과
        print("\n🔸 GRU Models:")
        for model_name, result in gru_results.items():
            print(f"\n  {model_name}:")
            if result.get('latency'):
                lat = result['latency']
                print(f"    Latency: {lat['mean']:.2f}ms (±{lat['std']:.2f}ms)")
                print(f"    Range: {lat['min']:.2f} - {lat['max']:.2f}ms")
            if result.get('throughput_fps'):
                print(f"    Throughput: {result['throughput_fps']:.1f} FPS")
            if result.get('memory', {}).get('monitor_success'):
                mem = result['memory']
                if mem.get('avg_device_utilization', 0) > 0:
                    print(f"    Device Util: {mem['avg_device_utilization']:.1f}% (max: {mem['max_device_utilization']:.1f}%)")
                if mem.get('avg_model_fps', 0) > 0:
                    print(f"    Monitor FPS: {mem['avg_model_fps']:.1f}")
            elif result.get('memory', {}).get('error_message'):
                print(f"    Memory: {result['memory']['error_message']}")
        
        # LSTM 결과
        print("\n🔹 LSTM Models:")
        for model_name, result in lstm_results.items():
            print(f"\n  {model_name}:")
            if result.get('latency'):
                lat = result['latency']
                print(f"    Latency: {lat['mean']:.2f}ms (±{lat['std']:.2f}ms)")
                print(f"    Range: {lat['min']:.2f} - {lat['max']:.2f}ms")
            if result.get('throughput_fps'):
                print(f"    Throughput: {result['throughput_fps']:.1f} FPS")
            if result.get('memory', {}).get('monitor_success'):
                mem = result['memory']
                if mem.get('avg_device_utilization', 0) > 0:
                    print(f"    Device Util: {mem['avg_device_utilization']:.1f}% (max: {mem['max_device_utilization']:.1f}%)")
                if mem.get('avg_model_fps', 0) > 0:
                    print(f"    Monitor FPS: {mem['avg_model_fps']:.1f}")
            elif result.get('memory', {}).get('error_message'):
                print(f"    Memory: {result['memory']['error_message']}")
        
        # 전체 통계
        print("\n📊 Overall Statistics:")
        all_latencies = []
        all_throughputs = []
        
        for result in self.results.values():
            if result and result.get('latency'):
                all_latencies.append(result['latency']['mean'])
            if result and result.get('throughput_fps'):
                all_throughputs.append(result['throughput_fps'])
        
        if all_latencies:
            print(f"  Average Latency: {np.mean(all_latencies):.2f}ms")
            print(f"  Latency Range: {np.min(all_latencies):.2f} - {np.max(all_latencies):.2f}ms")
        
        if all_throughputs:
            print(f"  Average Throughput: {np.mean(all_throughputs):.1f} FPS")
            print(f"  Throughput Range: {np.min(all_throughputs):.1f} - {np.max(all_throughputs):.1f} FPS")
        
        print(f"\n  Total Models Tested: {len(self.results)}")
        print("="*70)


def main():
    """
    메인 실행 함수
    """
    # 메모리 모니터링을 위한 환경변수 설정
    os.environ['HAILO_MONITOR'] = '1'
    os.environ['HAILO_MONITOR_TIME_INTERVAL'] = '500'  # 500ms 간격
    
    # 경로 설정 (정확한 경로로 수정)
    models_dir = "/root/2025-05-25/compiled_model/hef"  # .hef 파일들이 있는 디렉토리
    data_dir = "/root/2025-05-25/data"  # CSV 파일들이 있는 디렉토리
    
    # 경로 존재 여부 확인
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print(f"Models directory: {models_dir}")
    print(f"Data directory: {data_dir}")
    print(f"HAILO_MONITOR environment variable set: {os.environ.get('HAILO_MONITOR')}")
    
    # 성능 측정 실행
    measurer = NPUPerformanceMeasurer(models_dir, data_dir)
    measurer.run_all_measurements()


if __name__ == "__main__":
    main()