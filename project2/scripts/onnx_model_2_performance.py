import time
import numpy as np
import onnx
import onnxruntime as ort
import os
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import psutil
import pandas as pd
import glob
import json
from datetime import datetime
import threading

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
except pynvml.NVMLError: # Handle cases where NVML might be present but fails to init
    NVML_AVAILABLE = False


def print_gpu_usage(): # This function can remain for console logging if needed
    if NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            print(f"GPU Memory Used: {meminfo.used / 1024**2:.2f} MB / {meminfo.total / 1024**2:.2f} MB ({meminfo.used / meminfo.total * 100:.1f}%)")
            print(f"GPU Utilization: {util.gpu}%")
            print(f"GPU Power Draw: {power_watts:.2f} W")
        except pynvml.NVMLError as e:
            print(f"Could not retrieve GPU info from pynvml: {e}")
    else:
        print("pynvml is not installed or failed to initialize. GPU usage/power info not available.")

def measure_gpu_performance(ort_session, input_data, repeat=100):
    current_providers = ort_session.get_providers()
    is_cuda_used = 'CUDAExecutionProvider' in current_providers
    input_feed = {}
    all_inputs = ort_session.get_inputs()
    if not all_inputs:
        raise ValueError("ONNX model has no inputs!")

    main_input_meta = all_inputs[0]
    input_feed[main_input_meta.name] = input_data
    batch_size = input_data.shape[0]

    for i in range(1, len(all_inputs)):
        meta = all_inputs[i]
        shape = []
        for dim in meta.shape:
            if isinstance(dim, str) or dim is None: # Dynamic dimension
                # Attempt to use batch_size for dynamic dimensions, assuming it's the batch dim
                # This might need more sophisticated handling if other dims are dynamic
                shape.append(batch_size)
            else:
                shape.append(dim)
        # Ensure all dimensions are integers
        try:
            shape = [int(s) for s in shape]
        except ValueError:
            raise ValueError(f"Could not determine concrete shape for input {meta.name}. Original shape: {meta.shape}, derived: {shape}")

        dtype_str = meta.type
        if dtype_str == 'tensor(float)':
            dtype = np.float32
        elif dtype_str == 'tensor(double)':
            dtype = np.float64
        elif dtype_str == 'tensor(int64)':
            dtype = np.int64
        elif dtype_str == 'tensor(int32)':
            dtype = np.int32
        # Add more type mappings if needed
        else:
            # Fallback or raise error for unhandled types
            print(f"Warning: Unhandled ONNX input type '{dtype_str}' for input '{meta.name}'. Defaulting to float32.")
            dtype = np.float32
        input_feed[meta.name] = np.zeros(shape, dtype=dtype)


    latencies_ms = []
    all_outputs_for_rmse = []
    power_readings_watts = []

    if repeat > 1: # Warm-up only if multiple iterations
        ort_session.run(None, input_feed)

    for i in range(repeat):
        start_time = time.time()
        outputs = ort_session.run(None, input_feed)
        end_time = time.time()
        latencies_ms.append((end_time - start_time) * 1000) # Convert to milliseconds
        if i == 0:
            all_outputs_for_rmse = outputs
        if NVML_AVAILABLE and is_cuda_used:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_readings_watts.append(power_draw_mw / 1000.0)
            except pynvml.NVMLError:
                pass # Ignore if power reading fails for a single iteration

    latencies_arr = np.array(latencies_ms)
    avg_power_watts = np.mean(power_readings_watts) if power_readings_watts else None
    
    mean_latency_ms = np.mean(latencies_arr) if latencies_arr.size > 0 else 0
    
    stats = {
        "mean": mean_latency_ms,
        "median": np.median(latencies_arr) if latencies_arr.size > 0 else 0,
        "min": np.min(latencies_arr) if latencies_arr.size > 0 else 0,
        "max": np.max(latencies_arr) if latencies_arr.size > 0 else 0,
        "std": np.std(latencies_arr) if latencies_arr.size > 0 else 0,
        "throughput": 1000 / mean_latency_ms if mean_latency_ms > 0 else float('inf'),
        "avg_power_watts": avg_power_watts,
        "latencies": latencies_arr.tolist() # For p95/p99
    }
    return stats, all_outputs_for_rmse

def load_and_preprocess_data(csv_path, seq_len, feature_dim, target_col_index=3): # target_col_index default to 3 (Close)
    try:
        df = pd.read_csv(csv_path, skiprows=3)
    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_path}")
        return None, None, None, None, None, None

    date_list = df.iloc[:, 0].values

    # Assuming 'Open', 'High', 'Low', 'Close', 'Volume' are contiguous after the Date column
    # And that the first of these is at index 1 of the CSV (after Date at index 0)
    if feature_dim == 5: # Standard OCHLV
        data_columns = df.iloc[:, 1:1+feature_dim].values
    else: # Flexible, but ensure it matches your model's expected features
        print(f"Warning: feature_dim is {feature_dim}, not 5. Ensure data selection is correct.")
        data_columns = df.iloc[:, 1:1+feature_dim].values # Adjust if necessary

    try:
        data = data_columns.astype(np.float32)
    except ValueError:
        print("Error: Could not convert data to float. Check for non-numeric values in the selected columns after skipping rows.")
        return None, None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < seq_len + 1:
        print(f"Error: Not enough data (got {len(scaled_data)}, need {seq_len + 1}) to create a sample from {csv_path}.")
        return None, None, None, None, None, None

    sample_input_data = scaled_data[0:seq_len, :]
    sample_input_data = np.reshape(sample_input_data, (1, seq_len, feature_dim)).astype(np.float32)
    
    # Ensure target_col_index is within the bounds of the scaled_data columns
    if target_col_index >= scaled_data.shape[1]:
        print(f"Error: target_col_index {target_col_index} is out of bounds for scaled_data with shape {scaled_data.shape}")
        return None, None, None, None, None, None
        
    ground_truth_value = scaled_data[seq_len, target_col_index]
    ground_truth = np.array([[ground_truth_value]]).astype(np.float32)
    target_date = date_list[seq_len]

    return sample_input_data, ground_truth, scaler, target_date, df, scaled_data

def extract_ticker_from_model(model_filename):
    base = os.path.basename(model_filename)
    parts = base.replace('.onnx', '').split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

def extract_ticker_from_csv(csv_filename):
    base = os.path.basename(csv_filename)
    parts = base.split('_')
    if len(parts) >= 1:
        return parts[0]
    return None

def run_single_model_onnx(onnx_model_path, actual_data_csv_path, results_dict):
    batch_size = 1
    seq_len = 8
    feature_dim = 5
    target_column_for_rmse = 3 # 0:Open, 1:High, 2:Low, 3:Close(target), 4:Volume (relative to feature columns)

    model_basename = os.path.splitext(os.path.basename(onnx_model_path))[0]

    ort_session = ort.InferenceSession(
        onnx_model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print("ONNX Runtime Providers Used:", ort_session.get_providers())
    is_cuda_provider_used = 'CUDAExecutionProvider' in ort_session.get_providers()

    input_data, ground_truth_for_sample, scaler, target_date, df, scaled_data = load_and_preprocess_data(
        actual_data_csv_path,
        seq_len,
        feature_dim,
        target_col_index=target_column_for_rmse
    )

    if input_data is None or ground_truth_for_sample is None or scaled_data is None:
        print(f"Skipping {onnx_model_path} due to data loading/preprocessing error.")
        results_dict[model_basename] = {
            "model": model_basename,
            "timestamp": pd.Timestamp.now().isoformat(),
            "error": "Data loading/preprocessing failed."
        }
        return

    process = psutil.Process(os.getpid())
    cpu_mem_before_gb = process.memory_info().rss / (1024 ** 3)
    
    initial_snapshot_gpu_metrics = {'util': 0.0, 'mem_mb': 0.0, 'power_w': 0.0, 'valid': False}

    if NVML_AVAILABLE and is_cuda_provider_used:
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
            if NVML_AVAILABLE and is_cuda_provider_used:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util_rates_thread = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_val = float(util_rates_thread.gpu)
                    
                    mem_info_thread = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_mb_val = float(mem_info_thread.used / (1024**2))
                    
                    power_draw_mw_thread = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_watts_val = float(power_draw_mw_thread / 1000.0)
                except pynvml.NVMLError: 
                    gpu_util_val, mem_used_mb_val, power_watts_val = 0.0, 0.0, 0.0
                except Exception:
                    gpu_util_val, mem_used_mb_val, power_watts_val = 0.0, 0.0, 0.0
            collected_gpu_metrics.append({'util': gpu_util_val, 'mem_mb': mem_used_mb_val, 'power_w': power_watts_val})
            time.sleep(0.025)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    repeat_count_main = 100 
    main_stats, main_model_outputs_sample = measure_gpu_performance(ort_session, input_data, repeat=repeat_count_main)

    stop_event.set()
    monitor_thread.join()

    _monitor_successful = False
    _avg_util, _max_util, _min_util = 0.0, 0.0, 0.0
    _util_samples_count = 0
    _util_values_list_for_json = []
    
    _avg_mem_mb, _max_mem_mb, _min_mem_mb = 0.0, 0.0, 0.0
    _mem_mb_values_list_for_json = []
    
    _avg_power_w, _max_power_w, _min_power_w = 0.0, 0.0, 0.0
    _power_w_values_list_for_json = []
    
    _error_message_monitor = None

    if NVML_AVAILABLE and is_cuda_provider_used:
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

        elif initial_snapshot_gpu_metrics['valid']: 
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
            _error_message_monitor = "Monitor thread collected no GPU metrics, using initial snapshot."
        else:
            _monitor_successful = False
            _error_message_monitor = "NVML available but failed to get initial snapshot or monitor values."
    else:
        _monitor_successful = False
        if not NVML_AVAILABLE:
            _error_message_monitor = "pynvml not available or failed to initialize."
        elif not is_cuda_provider_used:
            _error_message_monitor = "CUDAExecutionProvider not used, GPU monitoring skipped."

    memory_metrics = {
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

        "avg_model_fps": main_stats['throughput'],
        "max_model_fps": main_stats['throughput'], 
        "min_model_fps": main_stats['throughput'], 
        "fps_samples": 1, 
        "fps_values": [main_stats['throughput']],

        "avg_model_utilization": _avg_util, 
        "max_model_utilization": _max_util,
        "min_model_utilization": _min_util,
        "utilization_samples": _util_samples_count,
        "utilization_values": _util_values_list_for_json
    }
    if not _monitor_successful and _error_message_monitor:
        memory_metrics["error_message"] = _error_message_monitor
    
    latencies_list = main_stats.get('latencies', [])
    latency_dict = {
        "mean": main_stats['mean'],
        "median": main_stats['median'],
        "min": main_stats['min'],
        "max": main_stats['max'],
        "std": main_stats['std'],
        "p95": float(np.percentile(latencies_list, 95)) if latencies_list else main_stats['mean'],
        "p99": float(np.percentile(latencies_list, 99)) if latencies_list else main_stats['mean'],
    }
    throughput_fps = main_stats['throughput']

    all_rmse_squared_errors = []
    if scaled_data is not None and scaler is not None:
        for i in range(len(scaled_data) - seq_len):
            input_seq = scaled_data[i:i+seq_len, :]
            gt_val = scaled_data[i+seq_len, target_column_for_rmse]
            input_seq_reshaped = input_seq.reshape(1, seq_len, feature_dim).astype(np.float32)
            
            rmse_input_feed = {ort_session.get_inputs()[0].name: input_seq_reshaped}
            for k_idx in range(1, len(ort_session.get_inputs())):
                meta = ort_session.get_inputs()[k_idx]
                shape = [dim if isinstance(dim, int) else 1 for dim in meta.shape] 
                dtype_str = meta.type
                if dtype_str == 'tensor(float)': dtype_np = np.float32
                elif dtype_str == 'tensor(double)': dtype_np = np.float64
                elif dtype_str == 'tensor(int64)': dtype_np = np.int64
                elif dtype_str == 'tensor(int32)': dtype_np = np.int32
                else: dtype_np = np.float32 
                rmse_input_feed[meta.name] = np.zeros(shape, dtype=dtype_np)

            model_outputs_rmse = ort_session.run(None, rmse_input_feed)
            
            if not model_outputs_rmse or not model_outputs_rmse[0].size > 0:
                print(f"Warning: No output or empty output from model for RMSE calculation at index {i} for {model_basename}")
                continue
            pred_scaled_val = model_outputs_rmse[0].item() 


            pred_for_inv = np.zeros((1, feature_dim))
            gt_for_inv = np.zeros((1, feature_dim))
            pred_for_inv[0, target_column_for_rmse] = pred_scaled_val
            gt_for_inv[0, target_column_for_rmse] = gt_val
            
            pred_unscaled = scaler.inverse_transform(pred_for_inv)[0, target_column_for_rmse]
            gt_unscaled = scaler.inverse_transform(gt_for_inv)[0, target_column_for_rmse]
            
            all_rmse_squared_errors.append((pred_unscaled - gt_unscaled)**2) 
    
    avg_rmse = math.sqrt(np.mean(all_rmse_squared_errors)) if all_rmse_squared_errors else None
    cpu_mem_after_gb = process.memory_info().rss / (1024 ** 3)

    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    result = {
        "model": model_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "latency": latency_dict,
        "throughput_fps": throughput_fps,
        "memory": memory_metrics, # Use the new memory_metrics structure
        "rmse": avg_rmse,
        "data_samples": len(scaled_data) if scaled_data is not None else 0,
        # "avg_power_watts": main_stats.get("avg_power_watts") # This is extra, not in NPU format's top level
    }
    # Optionally, if avg_power_watts is desired, it can be added to the top level or inside memory.
    # For strict NPU format matching at the top level, it would be omitted here.
    # If keeping it, maybe add to memory: memory_metrics["avg_power_watts"] = main_stats.get("avg_power_watts")

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

        data_ticker = model_ticker # Use model_ticker directly
        csv_path = csv_ticker_map.get(data_ticker)
        if not csv_path:
            print(f"Skip {onnx_model_path}: No matching CSV for ticker {data_ticker}")
            continue

        print(f"\n=== Processing Model: {onnx_model_path} | Data: {csv_path} ===")
        run_single_model_onnx(onnx_model_path, csv_path, results_dict)
        time.sleep(1) # Consistent with tensorrt script

    output_filename_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_json_path = f'onnx_performance_{output_filename_ts}.json'
    
    with open(output_json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    main()