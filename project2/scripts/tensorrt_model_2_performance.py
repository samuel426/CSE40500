import time
import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler # 데이터 정규화를 위해 추가
import psutil # CPU 메모리 사용량 측정을 위해 추가
import pandas as pd # 데이터 로드를 위해 추가
import glob
import json # Ensure json is imported
from datetime import datetime # Ensure datetime is imported
import threading # Ensure threading is imported


try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# from hailo_sdk import HailoNPU, HailoModel  # Hailo 관련 부분은 Hailo 장착 컴퓨터에서만 사용

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_model_path, max_batch_size=1):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
        
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        profile = builder.create_optimization_profile()
        # Assuming the first input tensor. Adjust if your model has multiple inputs.
        input_name = network.get_input(0).name
        input_tensor_shape = network.get_input(0).shape 

        # Define min, opt, max shapes for the profile.
        # For this script, batch_size is effectively fixed by `max_batch_size` parameter.
        # Assuming other dimensions are fixed as per typical model structure (e.g., seq_len, features)
        # If your ONNX model has fixed dimensions (e.g. [1, 8, 5]), those will be used.
        # If it has dynamic ones (e.g. [-1, 8, 5]), we specify them here.
        
        # Example: If network.get_input(0).shape is like (-1, 8, 5)
        # And your script intends to run with batch_size=1, seq_len=8, feature_dim=5
        # seq_len and feature_dim would typically come from model inspection or constants
        # For this example, let's assume they are known (e.g., 8 and 5)
        # If input_tensor_shape already has concrete values (e.g. [1,8,5]), this will just confirm them.
        
        # Determine dimensions for profiling
        # Fallback to known dimensions if model uses dynamic axes extensively
        # These values are examples; adjust based on your model's actual expected input dimensions
        # Typically, seq_len and feature_dim are fixed for a given model.
        # The `max_batch_size` parameter to this function dictates the batch size for the profile.
        
        # Get non-batch dimensions from the input tensor shape
        # If input_tensor_shape is like `[-1, seq_dim, feat_dim]`
        # We need to provide concrete values for seq_dim and feat_dim if they are also dynamic,
        # or use the values from the model if they are static.
        # For this script, seq_len=8, feature_dim=5 are used later.
        # Let's assume these are the intended dimensions for the profile.
        
        # Construct shapes for the profile
        # If input_tensor_shape[0] is -1 (dynamic batch)
        min_shape_dims = list(input_tensor_shape)
        opt_shape_dims = list(input_tensor_shape)
        max_shape_dims = list(input_tensor_shape)

        if min_shape_dims[0] == -1: # Dynamic batch axis
            min_shape_dims[0] = 1
            opt_shape_dims[0] = max_batch_size # max_batch_size is 1 in your current usage
            max_shape_dims[0] = max_batch_size
        
        # Ensure other dynamic dimensions are also set if any.
        # For this example, assuming only batch is dynamic or all are fixed.
        # If seq_len (dim 1) or feature_dim (dim 2) were -1, they'd need to be set too.
        # e.g., if input_tensor_shape = [-1, -1, -1] and you know it's [batch, 8, 5]
        # opt_shape_dims[1] = 8 
        # opt_shape_dims[2] = 5
        # (and similarly for min/max)

        profile.set_shape(input_name, tuple(min_shape_dims), tuple(opt_shape_dims), tuple(max_shape_dims))
        config.add_optimization_profile(profile)

        # engine = builder.build_engine(network, config) # Deprecated
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            raise RuntimeError("Failed to build serialized TensorRT engine.")
        
        # Deserialize the serialized engine to return an ICudaEngine object
        # This is necessary because the calling code expects an engine object
        # that can be further serialized (e.g., engine.serialize()) or used directly.
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine from serialized plan.")
        return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding_name in engine: # Iterates over binding names
        # shape = engine.get_binding_shape(binding_name) # Deprecated
        shape = engine.get_tensor_shape(binding_name)
        
        # For explicit batch, the shape from get_tensor_shape already includes the batch dimension.
        # engine.max_batch_size is 1 and its use here is deprecated.
        # size = trt.volume(shape) * engine.max_batch_size # Old way
        size = trt.volume(shape)
        if size < 0: # Happens if shape still has dynamic dimensions (-1)
            # This indicates an issue with profile or model not having shapes resolved at this stage.
            # For allocation, we need a concrete size.
            # This might require context.set_binding_shape() then context.get_binding_shape()
            # or ensuring the profile's OPT shapes are used by default.
            # For now, we'll assume `get_tensor_shape` returns a usable shape due to the profile.
            # If this error occurs, you might need to pass batch_size to allocate_buffers
            # and manually construct the shape if dynamic dims are present.
            print(f"Warning: Binding '{binding_name}' has dynamic shape {shape}. Buffer allocation might be incorrect.")
            # As a fallback, if batch is the only dynamic dim, and we know it's 1 for this script:
            # temp_shape = list(shape)
            # if temp_shape[0] == -1: temp_shape[0] = 1 # Assuming batch_size = 1
            # size = trt.volume(temp_shape)
            # if size < 0: raise RuntimeError(f"Cannot determine size for binding {binding_name} with shape {shape}")


        # dtype = trt.nptype(engine.get_binding_dtype(binding_name)) # Deprecated
        dtype_trt = engine.get_tensor_dtype(binding_name)
        dtype_np = trt.nptype(dtype_trt)

        host_mem = cuda.pagelocked_empty(size, dtype_np)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        # if engine.binding_is_input(binding_name): # Deprecated
        if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'name': binding_name})
    return inputs, outputs, bindings, stream


def print_gpu_usage():
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 첫 번째 GPU 기준
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Watts 단위
            print(f"GPU Memory Used: {meminfo.used / 1024**2:.2f} MB / {meminfo.total / 1024**2:.2f} MB ({meminfo.used / meminfo.total * 100:.1f}%)")
            print(f"GPU Utilization: {util.gpu}%")
            print(f"GPU Power Draw: {power_watts:.2f} W")
        except pynvml.NVMLError as e:
            print(f"Could not retrieve GPU info from pynvml: {e}")
    else:
        print("pynvml is not installed. GPU usage/power info not available.")

def do_inference_trt(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    # Return host outputs
    return [out['host'][:batch_size] for out in outputs]


def measure_trt_performance(engine, input_data, repeat=100):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # Assume single input
    np.copyto(inputs[0]['host'], input_data.ravel())
    latencies = []
    all_outputs_for_rmse = []
    # Warm-up
    do_inference_trt(context, bindings, inputs, outputs, stream, batch_size=input_data.shape[0])
    for i in range(repeat):
        start_time = time.time()
        out = do_inference_trt(context, bindings, inputs, outputs, stream, batch_size=input_data.shape[0])
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # 초 → 밀리초(ms)로 변환
        if i == 0:
            all_outputs_for_rmse = out
    latencies = np.array(latencies)
    stats = {
        "mean": np.mean(latencies),
        "median": np.median(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "std": np.std(latencies),
        "throughput": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else float('inf'),  # ms 기준
        "latencies": latencies  # p95, p99 계산용
    }
    return stats, all_outputs_for_rmse

def load_and_preprocess_data(csv_path, seq_len, feature_dim, target_col_index=4):
    try:
        df = pd.read_csv(csv_path, skiprows=3)
    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_path}")
        return None, None, None, None, None, None

    date_list = df.iloc[:, 0].values

    if feature_dim == 5:
        data_columns = df.iloc[:, 1:1+feature_dim].values
    else:
        data_columns = df.iloc[:, 0:feature_dim].values

    try:
        data = data_columns.astype(np.float32)
    except ValueError:
        print("Error: Could not convert data to float. Check for non-numeric values in the selected columns after skipping rows.")
        return None, None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < seq_len + 1:
        print("Error: Not enough data to create a sample.")
        return None, None, None, None, None, None

    sample_input_data = scaled_data[0:seq_len, :]
    sample_input_data = np.reshape(sample_input_data, (1, seq_len, feature_dim)).astype(np.float32)
    ground_truth_value = scaled_data[seq_len, target_col_index]
    ground_truth = np.array([[ground_truth_value]]).astype(np.float32)
    target_date = date_list[seq_len]

    # 반환값에 df, scaled_data 추가
    return sample_input_data, ground_truth, scaler, target_date, df, scaled_data

def extract_ticker_from_model(model_filename):
    # 예: GRU_Apple.onnx -> Apple
    base = os.path.basename(model_filename)
    parts = base.replace('.onnx', '').split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

def extract_ticker_from_csv(csv_filename):
    # 예: AAPL_actual_ohlcv.csv -> AAPL
    base = os.path.basename(csv_filename)
    parts = base.split('_')
    if len(parts) >= 1:
        return parts[0]
    return None

def monitor_gpu_utilization(stop_event, interval=0.1):
    values = []
    while not stop_event.is_set():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            values.append(util.gpu)
        except Exception:
            values.append(0.0)
        time.sleep(interval)
    return values

def run_single_model_trt(onnx_model_path, actual_data_csv_path, results_dict):
    batch_size = 1
    seq_len = 8
    feature_dim = 5
    target_column_for_rmse = 3

    model_basename = os.path.splitext(os.path.basename(onnx_model_path))[0]

    engine_path = onnx_model_path.replace('.onnx', '.engine')
    if not os.path.exists(engine_path):
        print(f"Building engine for {onnx_model_path}...")
        engine = build_engine_from_onnx(onnx_model_path, max_batch_size=batch_size)
        if engine is None:
            print(f"Failed to build engine for {onnx_model_path}. Skipping.")
            results_dict[model_basename] = {
                "model": model_basename,
                "timestamp": pd.Timestamp.now().isoformat(),
                "error": "Engine build failed"
            }
            return
        try:
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"Engine saved to {engine_path}")
        except Exception as e:
            print(f"Could not save engine: {e}")
            # Continue with the in-memory engine if saving fails but building succeeded
    else:
        print(f"Loading engine from {engine_path}...")
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print(f"Failed to load engine from {engine_path}. Skipping.")
            results_dict[model_basename] = {
                "model": model_basename,
                "timestamp": pd.Timestamp.now().isoformat(),
                "error": "Engine load failed"
            }
            return

    input_data, ground_truth_for_sample, scaler, target_date, df, scaled_data = load_and_preprocess_data(
        actual_data_csv_path, seq_len, feature_dim, target_col_index=target_column_for_rmse
    )
    if input_data is None or ground_truth_for_sample is None or scaled_data is None:
        print(f"Failed to load or preprocess data for {actual_data_csv_path}. Skipping model {onnx_model_path}.")
        results_dict[model_basename] = {
            "model": model_basename,
            "timestamp": pd.Timestamp.now().isoformat(),
            "error": "Data loading/preprocessing failed"
        }
        return

    process = psutil.Process(os.getpid())
    cpu_mem_before_gb = process.memory_info().rss / (1024 ** 3)
    
    initial_snapshot_gpu_metrics = {'util': 0.0, 'mem_mb': 0.0, 'power_w': 0.0, 'valid': False}
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            initial_snapshot_gpu_metrics['util'] = float(util_rates.gpu)
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            initial_snapshot_gpu_metrics['mem_mb'] = float(mem_info.used / (1024**2))
            
            power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            initial_snapshot_gpu_metrics['power_w'] = float(power_draw_mw / 1000.0)
            initial_snapshot_gpu_metrics['valid'] = True
        except pynvml.NVMLError:
            pass # initial_snapshot_gpu_metrics will remain with valid=False

    collected_gpu_metrics = [] # Stores dicts: {'util': gpu_util, 'mem_mb': mem_used_mb, 'power_w': power_watts}
    stop_event = threading.Event()
    def monitor():
        while not stop_event.is_set():
            gpu_util_val = 0.0
            mem_used_mb_val = 0.0
            power_watts_val = 0.0
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util_rates_thread = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_val = float(util_rates_thread.gpu)
                    
                    mem_info_thread = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_mb_val = float(mem_info_thread.used / (1024**2))
                    
                    power_draw_mw_thread = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_watts_val = float(power_draw_mw_thread / 1000.0)
                except pynvml.NVMLError as e:
                    print(f"NVML Error in monitor thread: {e}") # Add this line
                    gpu_util_val, mem_used_mb_val, power_watts_val = 0.0, 0.0, 0.0
                except Exception as e:
                    print(f"Generic Error in monitor thread: {e}") # Add this line
                    gpu_util_val, mem_used_mb_val, power_watts_val = 0.0, 0.0, 0.0
            collected_gpu_metrics.append({'util': gpu_util_val, 'mem_mb': mem_used_mb_val, 'power_w': power_watts_val})
            time.sleep(0.025)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    repeat_count = 100 
    stats, model_outputs_sample_main = measure_trt_performance(engine, input_data, repeat=repeat_count)

    stop_event.set()
    monitor_thread.join()

    avg_fps_from_stats = stats['throughput']
    
    _avg_util, _max_util, _min_util = 0.0, 0.0, 0.0
    _util_samples_count = 0
    _util_values_list_for_json = []
    
    _avg_mem_mb, _max_mem_mb, _min_mem_mb = 0.0, 0.0, 0.0
    _mem_mb_values_list_for_json = []
    
    _avg_power_w, _max_power_w, _min_power_w = 0.0, 0.0, 0.0
    _power_w_values_list_for_json = []
    
    _monitor_successful = False

    if collected_gpu_metrics:
        gpu_utils_collected = [m['util'] for m in collected_gpu_metrics]
        gpu_mems_mb_collected = [m['mem_mb'] for m in collected_gpu_metrics]
        gpu_powers_w_collected = [m['power_w'] for m in collected_gpu_metrics]

        if gpu_utils_collected: 
            _avg_util = float(np.mean(gpu_utils_collected))
            _max_util = float(np.max(gpu_utils_collected))
            _min_util = float(np.min(gpu_utils_collected))
            _util_values_list_for_json = [float(v) for v in gpu_utils_collected[:10]]
        
        if gpu_mems_mb_collected:
            _avg_mem_mb = float(np.mean(gpu_mems_mb_collected))
            _max_mem_mb = float(np.max(gpu_mems_mb_collected))
            _min_mem_mb = float(np.min(gpu_mems_mb_collected))
            _mem_mb_values_list_for_json = [float(v) for v in gpu_mems_mb_collected[:10]]

        if gpu_powers_w_collected:
            _avg_power_w = float(np.mean(gpu_powers_w_collected))
            _max_power_w = float(np.max(gpu_powers_w_collected))
            _min_power_w = float(np.min(gpu_powers_w_collected))
            _power_w_values_list_for_json = [float(v) for v in gpu_powers_w_collected[:10]]
            
        _util_samples_count = len(collected_gpu_metrics)
        _monitor_successful = True if _util_samples_count > 0 else False


    elif NVML_AVAILABLE and initial_snapshot_gpu_metrics['valid']:
        _avg_util = initial_snapshot_gpu_metrics['util']
        _max_util = initial_snapshot_gpu_metrics['util']
        _min_util = initial_snapshot_gpu_metrics['util']
        _util_values_list_for_json = [initial_snapshot_gpu_metrics['util']]
        
        _avg_mem_mb = initial_snapshot_gpu_metrics['mem_mb']
        _max_mem_mb = initial_snapshot_gpu_metrics['mem_mb']
        _min_mem_mb = initial_snapshot_gpu_metrics['mem_mb']
        _mem_mb_values_list_for_json = [initial_snapshot_gpu_metrics['mem_mb']]
        
        _avg_power_w = initial_snapshot_gpu_metrics['power_w']
        _max_power_w = initial_snapshot_gpu_metrics['power_w']
        _min_power_w = initial_snapshot_gpu_metrics['power_w']
        _power_w_values_list_for_json = [initial_snapshot_gpu_metrics['power_w']]
        
        _util_samples_count = 1
        _monitor_successful = True
        
    memory = {
        "monitor_success": _monitor_successful,
        "total_lines": len(scaled_data) if scaled_data is not None else 0,

        "avg_device_utilization": _avg_util,
        "max_device_utilization": _max_util,
        "min_device_utilization": _min_util,
        "device_util_samples": _util_samples_count,
        "device_util_values": _util_values_list_for_json,

        "avg_gpu_memory_mb": _avg_mem_mb,
        "max_gpu_memory_mb": _max_mem_mb,
        "min_gpu_memory_mb": _min_mem_mb,
        "gpu_memory_mb_samples": _util_samples_count, 
        "gpu_memory_mb_values": _mem_mb_values_list_for_json,

        "avg_gpu_power_watts": _avg_power_w,
        "max_gpu_power_watts": _max_power_w,
        "min_gpu_power_watts": _min_power_w,
        "gpu_power_watts_samples": _util_samples_count,
        "gpu_power_watts_values": _power_w_values_list_for_json,

        "avg_model_fps": avg_fps_from_stats,
        "max_model_fps": avg_fps_from_stats,
        "min_model_fps": avg_fps_from_stats,
        "fps_samples": 1, 
        "fps_values": [avg_fps_from_stats],

        "avg_model_utilization": _avg_util, 
        "max_model_utilization": _max_util,
        "min_model_utilization": _min_util,
        "utilization_samples": _util_samples_count, 
        "utilization_values": _util_values_list_for_json 
    }
    
    latencies_list = stats.get('latencies', np.array([])) 
    latency_dict = {
        "mean": stats['mean'],
        "median": stats['median'],
        "min": stats['min'],
        "max": stats['max'],
        "std": stats['std'],
        "p95": float(np.percentile(latencies_list, 95)) if latencies_list.size > 0 else stats['mean'],
        "p99": float(np.percentile(latencies_list, 99)) if latencies_list.size > 0 else stats['mean'],
    }
    throughput_fps = stats['throughput']


    # RMSE calculation for the entire dataset
    all_rmse = []
    if scaled_data is not None and scaler is not None:
        for i in range(len(scaled_data) - seq_len):
            input_seq = scaled_data[i:i+seq_len, :]
            gt_val = scaled_data[i+seq_len, target_column_for_rmse] 
            
            input_seq_reshaped = input_seq.reshape(1, seq_len, feature_dim).astype(np.float32)
            
            rmse_stats, model_outputs_sample_rmse = measure_trt_performance(engine, input_seq_reshaped, repeat=1)
            
            if not model_outputs_sample_rmse or not model_outputs_sample_rmse[0].size > 0 : 
                print(f"Warning: No output or empty output from model for RMSE calculation at index {i}")
                continue

            pred_scaled_val = model_outputs_sample_rmse[0].item() 
            
            # Inverse transform
            # Create a dummy array with the feature_dim, put prediction/gt at target_column_for_rmse
            pred_for_inv = np.zeros((1, feature_dim))
            gt_for_inv = np.zeros((1, feature_dim))
            
            pred_for_inv[0, target_column_for_rmse] = pred_scaled_val
            gt_for_inv[0, target_column_for_rmse] = gt_val # gt_val is already a scalar
            
            pred_unscaled = scaler.inverse_transform(pred_for_inv)[0, target_column_for_rmse]
            gt_unscaled = scaler.inverse_transform(gt_for_inv)[0, target_column_for_rmse]
            
            rmse_unscaled = abs(pred_unscaled - gt_unscaled) # MAE-like for unscaled, or use squared error
            all_rmse.append(rmse_unscaled**2) # Append squared error for RMSE calculation
    
    avg_rmse = math.sqrt(np.mean(all_rmse)) if all_rmse else None # Calculate RMSE from mean of squared errors

    cpu_mem_after_gb = process.memory_info().rss / (1024 ** 3)

    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    result = {
        "model": model_name,
        "timestamp": pd.Timestamp.now().isoformat(), # Using pandas Timestamp for consistency
        "latency": latency_dict,
        "throughput_fps": throughput_fps,
        "memory": memory,
        "rmse": avg_rmse,
        "data_samples": len(scaled_data)
    }

    results_dict[model_name] = result
    print(f"Finished processing {model_name}. Avg Latency: {latency_dict['mean']:.2f}ms, RMSE: {avg_rmse:.4f}" if avg_rmse is not None else f"Finished processing {model_name}. RMSE not calculated.")


def run_single_model_trt_org(onnx_model_path, actual_data_csv_path, results_dict):
    batch_size = 1
    seq_len = 8
    feature_dim = 5
    target_column_for_rmse = 3

    engine_path = onnx_model_path.replace('.onnx', '.engine')
    if not os.path.exists(engine_path):
        print(f"Building engine for {onnx_model_path}...")
        engine = build_engine_from_onnx(onnx_model_path, max_batch_size=batch_size)
        if engine is None:
            print(f"Failed to build engine for {onnx_model_path}. Skipping.")
            return
        try:
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"Engine saved to {engine_path}")
        except Exception as e:
            print(f"Could not save engine: {e}")
            # Continue with the in-memory engine if saving fails but building succeeded
    else:
        print(f"Loading engine from {engine_path}...")
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print(f"Failed to load engine from {engine_path}. Skipping.")
            return

    input_data, ground_truth_for_sample, scaler, target_date, df, scaled_data = load_and_preprocess_data(
        actual_data_csv_path, seq_len, feature_dim, target_col_index=target_column_for_rmse
    )
    if input_data is None or ground_truth_for_sample is None or scaled_data is None:
        print(f"Failed to load or preprocess data for {actual_data_csv_path}. Skipping model {onnx_model_path}.")
        return

    process = psutil.Process(os.getpid())
    cpu_mem_before_gb = process.memory_info().rss / (1024 ** 3)
    
    initial_snapshot_gpu_util = None
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            initial_snapshot_gpu_util = util.gpu
        except pynvml.NVMLError:
            pass # initial_snapshot_gpu_util will remain None

    util_values = []
    stop_event = threading.Event()
    def monitor():
        while not stop_event.is_set():
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_values.append(util.gpu)
                except Exception: # Catch broad exceptions during monitoring
                    util_values.append(0.0) # Log 0.0 if a read fails
            else:
                # If NVML is not available, we can't collect util values.
                # The loop will still run but util_values will remain empty or fill with 0.0 if NVML_AVAILABLE was true initially then failed.
                # To prevent busy-waiting if NVML becomes unavailable mid-run (unlikely but possible):
                pass # Or append a marker like -1, or simply let util_values be empty.
                     # Appending 0.0 is consistent with error handling above.
                util_values.append(0.0)
            time.sleep(0.025)  # Adjusted to 25ms to be comparable if NPU is 20x slower and uses 500ms

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    repeat_count = 100 # Main inference loop for latency/throughput
    stats, model_outputs_sample_main = measure_trt_performance(engine, input_data, repeat=repeat_count)

    stop_event.set()
    monitor_thread.join()

    avg_fps_from_stats = stats['throughput']
    _avg_util = 0.0
    _max_util = 0.0
    _min_util = 0.0
    _util_samples_count = 0
    _util_values_list_for_json = []
    _monitor_successful = False

    if util_values:
        _avg_util = float(np.mean(util_values))
        _max_util = float(np.max(util_values))
        _min_util = float(np.min(util_values))
        _util_samples_count = len(util_values)
        _util_values_list_for_json = [float(v) for v in util_values[:10]] # Store first 10, ensure float
        _monitor_successful = True
    elif NVML_AVAILABLE and initial_snapshot_gpu_util is not None:
        # Fallback to the initial snapshot if monitor didn't yield values but NVML was initially OK
        _avg_util = float(initial_snapshot_gpu_util)
        _max_util = float(initial_snapshot_gpu_util)
        _min_util = float(initial_snapshot_gpu_util)
        _util_samples_count = 1
        _util_values_list_for_json = [float(initial_snapshot_gpu_util)]
        _monitor_successful = True # Monitor might have run, or this is a pre-monitor snapshot
    # If NVML_AVAILABLE is false, _monitor_successful remains False, and utils are 0/empty

    memory = {
        "monitor_success": _monitor_successful,
        "total_lines": len(scaled_data), # Number of rows in the preprocessed data

        "avg_device_utilization": _avg_util,
        "max_device_utilization": _max_util,
        "min_device_utilization": _min_util,
        "device_util_samples": _util_samples_count,
        "device_util_values": _util_values_list_for_json,

        "avg_model_fps": avg_fps_from_stats,
        "max_model_fps": avg_fps_from_stats,
        "min_model_fps": avg_fps_from_stats,
        "fps_samples": 1, # Reflects that FPS is from one main measurement block
        "fps_values": [avg_fps_from_stats],

        "avg_model_utilization": _avg_util, # For GPU, model utilization is overall GPU utilization
        "max_model_utilization": _max_util,
        "min_model_utilization": _min_util,
        "utilization_samples": _util_samples_count,
        "utilization_values": _util_values_list_for_json
    }
    
    latencies_list = stats.get('latencies', []) # Get the raw latencies list
    latency_dict = {
        "mean": stats['mean'],
        "median": stats['median'],
        "min": stats['min'],
        "max": stats['max'],
        "std": stats['std'],
        "p95": float(np.percentile(latencies_list, 95)) if latencies_list.size > 0 else stats['mean'],
        "p99": float(np.percentile(latencies_list, 99)) if latencies_list.size > 0 else stats['mean'],
    }
    throughput_fps = stats['throughput']

    # RMSE calculation for the entire dataset
    all_rmse = []
    # Ensure 'scaled_data' and 'scaler' are available from load_and_preprocess_data
    if scaled_data is not None and scaler is not None:
        for i in range(len(scaled_data) - seq_len):
            input_seq = scaled_data[i:i+seq_len, :]
            gt_val = scaled_data[i+seq_len, target_column_for_rmse] # Ground truth is a scalar
            
            input_seq_reshaped = input_seq.reshape(1, seq_len, feature_dim).astype(np.float32)
            
            # For RMSE, run inference once per sample.
            # The 'measure_trt_performance' function can be used with repeat=1
            # Or, a more direct inference call if 'measure_trt_performance' adds too much overhead for single inferences.
            # Using measure_trt_performance with repeat=1 is simpler here.
            rmse_stats, model_outputs_sample_rmse = measure_trt_performance(engine, input_seq_reshaped, repeat=1)
            
            if not model_outputs_sample_rmse: # Check if output is valid
                print(f"Warning: No output from model for RMSE calculation at index {i}")
                continue

            pred_scaled_val = model_outputs_sample_rmse[0].item() # Assuming single output, single value

            # Inverse transform
            # Create a dummy array with the feature_dim, put prediction/gt at target_column_for_rmse
            pred_for_inv = np.zeros((1, feature_dim))
            gt_for_inv = np.zeros((1, feature_dim))
            
            pred_for_inv[0, target_column_for_rmse] = pred_scaled_val
            gt_for_inv[0, target_column_for_rmse] = gt_val # gt_val is already a scalar
            
            pred_unscaled = scaler.inverse_transform(pred_for_inv)[0, target_column_for_rmse]
            gt_unscaled = scaler.inverse_transform(gt_for_inv)[0, target_column_for_rmse]
            
            rmse_unscaled = abs(pred_unscaled - gt_unscaled) # MAE-like for unscaled, or use squared error
            all_rmse.append(rmse_unscaled**2) # Append squared error for RMSE calculation
    
    avg_rmse = math.sqrt(np.mean(all_rmse)) if all_rmse else None # Calculate RMSE from mean of squared errors

    cpu_mem_after_gb = process.memory_info().rss / (1024 ** 3)

    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    result = {
        "model": model_name,
        "timestamp": pd.Timestamp.now().isoformat(), # Using pandas Timestamp for consistency
        "latency": latency_dict,
        "throughput_fps": throughput_fps,
        "memory": memory,
        "rmse": avg_rmse,
        "data_samples": len(scaled_data)
    }

    results_dict[model_name] = result
    print(f"Finished processing {model_name}. Avg Latency: {latency_dict['mean']:.2f}ms, RMSE: {avg_rmse:.4f}" if avg_rmse is not None else f"Finished processing {model_name}. RMSE not calculated.")


def main():
    onnx_dir = './onnx_models'
    data_dir = './data'
    onnx_files = sorted(glob.glob(os.path.join(onnx_dir, '*.onnx')))
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*_actual_ohlcv.csv')))
    csv_ticker_map = {extract_ticker_from_csv(f): f for f in csv_files}
    results_dict = {}
    for onnx_model_path in onnx_files:
        model_ticker = extract_ticker_from_model(onnx_model_path)
        if not model_ticker:
            print(f"Skip {onnx_model_path}: ticker not found in filename.")
            continue
        # ticker_map = {'Apple': 'AAPL'}  # 이 줄을 제거
        # data_ticker = ticker_map.get(model_ticker, model_ticker)  # 이 줄을 제거
        data_ticker = model_ticker  # 모델에서 추출한 ticker 그대로 사용
        csv_path = csv_ticker_map.get(data_ticker)
        if not csv_path:
            print(f"Skip {onnx_model_path}: No matching CSV for ticker {data_ticker}")
            continue
        print(f"\n=== Processing Model: {onnx_model_path} | Data: {csv_path} ===")
        run_single_model_trt(onnx_model_path, csv_path, results_dict)
        time.sleep(1)
    # Save results to tensorrt_performance_yyyymmdd_hhmmss.json
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'tensorrt_performance_{now}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
