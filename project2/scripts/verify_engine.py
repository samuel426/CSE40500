import tensorrt as trt
import onnxruntime
import numpy as np
import argparse
import os
import pycuda.driver as cuda
import pycuda.autoinit # Ensures CUDA context is initialized

# Helper class for managing host and device memory for TensorRT
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def do_inference_trt(context, bindings, inputs, outputs, stream):
    """
    Performs inference using TensorRT.
    inputs: list of HostDeviceMem for input buffers.
    outputs: list of HostDeviceMem for output buffers.
    bindings: list of device buffer pointers.
    stream: CUDA stream.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return host outputs
    return [out.host for out in outputs]

def main(args):
    status_message = "FAILURE: Verification not yet run."
    print(f"TensorRT version: {trt.__version__}")
    print(f"ONNX Runtime version: {onnxruntime.__version__}")

    try:
        # 1. Load TensorRT Engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(args.engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {args.engine_path}")
        
        with open(args.engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if not engine:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        print(f"Successfully loaded TensorRT engine: {args.engine_path}")

        # 2. Load ONNX Model
        if not os.path.exists(args.onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {args.onnx_path}")
        
        # Use CUDAExecutionProvider if available and desired for ONNX, otherwise CPU
        providers = ['CPUExecutionProvider']
        if args.onnx_gpu and onnxruntime.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        ort_session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
        print(f"Successfully loaded ONNX model: {args.onnx_path} (using {ort_session.get_providers()})")

        # 3. Prepare Input Data (based on ONNX model's input)
        onnx_inputs_meta = ort_session.get_inputs()
        dummy_inputs_onnx_feed_dict = {}
        # This list will store numpy arrays in the order TRT expects its inputs
        # We'll try to match ONNX inputs to TRT input bindings by order or name if possible
        
        print("\nGenerating input data based on ONNX model specs:")
        # Store generated numpy arrays for TRT input preparation
        # This list should be ordered according to TRT's input binding order
        # For now, we assume the order of onnx_inputs_meta matches TRT input binding order
        # A more robust approach would map by name.
        ordered_input_data_np = [None] * sum(1 for i in range(engine.num_bindings) if engine.binding_is_input(i))


        for i_meta in onnx_inputs_meta:
            name = i_meta.name
            shape = []
            for dim_idx, dim in enumerate(i_meta.shape):
                if isinstance(dim, str) or dim is None or dim == 0 or (isinstance(dim, int) and dim < 0) : # dynamic or unknown
                    print(f"  Input '{name}' has dynamic dimension at index {dim_idx} ('{dim}'). Using default size 1.")
                    shape.append(1) # Default to 1 for dynamic/batch dimension
                else:
                    shape.append(dim)
            
            onnx_dtype_str = i_meta.type
            if onnx_dtype_str == 'tensor(float)': np_dtype = np.float32
            elif onnx_dtype_str == 'tensor(float16)': np_dtype = np.float16
            elif onnx_dtype_str == 'tensor(int64)': np_dtype = np.int64
            elif onnx_dtype_str == 'tensor(int32)': np_dtype = np.int32
            elif onnx_dtype_str == 'tensor(uint8)': np_dtype = np.uint8
            elif onnx_dtype_str == 'tensor(bool)': np_dtype = np.bool_
            else: raise NotImplementedError(f"Unsupported ONNX input type: {onnx_dtype_str} for input '{name}'")

            print(f"  Input '{name}': Shape for generation: {shape}, ONNX Type: {onnx_dtype_str}, NumPy Dtype: {np_dtype}")
            dummy_input_data = np.random.randn(*shape).astype(np_dtype)
            dummy_inputs_onnx_feed_dict[name] = dummy_input_data
            
            # Attempt to place this input data into the correct slot for TRT
            # This assumes ONNX input names match TRT binding names for inputs
            found_binding_for_onnx_input = False
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i) and engine.get_binding_name(i) == name:
                    # This is a naive way to get the "index" of the input binding
                    input_binding_index_in_engine = sum(1 for k in range(i) if engine.binding_is_input(k))
                    if input_binding_index_in_engine < len(ordered_input_data_np):
                         ordered_input_data_np[input_binding_index_in_engine] = dummy_input_data
                         found_binding_for_onnx_input = True
                         break
            if not found_binding_for_onnx_input:
                print(f"Warning: Could not find matching input binding in TRT engine for ONNX input '{name}'. Order will be assumed if names don't match.")


        # If name matching failed for some, fill by order (less reliable)
        current_onnx_input_idx = 0
        for i in range(len(ordered_input_data_np)):
            if ordered_input_data_np[i] is None:
                if current_onnx_input_idx < len(onnx_inputs_meta):
                    onnx_input_name_by_order = onnx_inputs_meta[current_onnx_input_idx].name
                    ordered_input_data_np[i] = dummy_inputs_onnx_feed_dict[onnx_input_name_by_order]
                    print(f"  Assigned ONNX input '{onnx_input_name_by_order}' to TRT input binding slot {i} by order.")
                    current_onnx_input_idx += 1
                else:
                    raise RuntimeError(f"Mismatch in number of inputs between ONNX and TRT (could not fill TRT input slot {i}).")
        if any(inp is None for inp in ordered_input_data_np):
             raise RuntimeError("Failed to prepare all TRT input data buffers.")


        # 4. Run Inference
        # ONNX Inference
        onnx_output_names = [output.name for output in ort_session.get_outputs()]
        print(f"\nRunning ONNX inference. Output names: {onnx_output_names}")
        onnx_results_list = ort_session.run(onnx_output_names, dummy_inputs_onnx_feed_dict)
        onnx_results_dict = {name: res for name, res in zip(onnx_output_names, onnx_results_list)}

        # TensorRT Inference
        print("\nPreparing for TensorRT inference...")
        context = engine.create_execution_context()
        if not context: raise RuntimeError("Failed to create TensorRT execution context.")

        trt_input_allocs = []
        trt_output_allocs = []
        bindings = [None] * engine.num_bindings
        
        # Prepare TRT input buffers and set binding shapes
        current_trt_input_idx = 0
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            binding_dtype_trt = engine.get_binding_dtype(i)
            np_binding_dtype = trt.nptype(binding_dtype_trt)
            
            if engine.binding_is_input(i):
                if current_trt_input_idx >= len(ordered_input_data_np):
                    raise RuntimeError(f"TRT engine expects more inputs than prepared from ONNX model.")
                
                input_data_np = ordered_input_data_np[current_trt_input_idx]
                # Ensure data type matches what TRT expects for the binding
                input_data_np = input_data_np.astype(np_binding_dtype, copy=False) 
                
                print(f"  TRT Input Binding {i} ('{binding_name}'): Shape from ONNX data: {input_data_np.shape}, Dtype: {np_binding_dtype}")
                if not context.set_binding_shape(i, input_data_np.shape): # Set shape for dynamic inputs
                    print(f"    Warning: Failed to set binding shape for TRT input '{binding_name}'. This might be an issue if shapes are truly dynamic and not covered by a default profile.")

                # Allocate memory after setting shape
                # Use context.get_binding_shape(i) AFTER set_binding_shape if it was dynamic
                resolved_shape = context.get_binding_shape(i)
                if any(d < 0 for d in resolved_shape): # Should be resolved now
                    raise RuntimeError(f"Dynamic dimension for TRT input '{binding_name}' not resolved after set_binding_shape. Shape: {resolved_shape}")

                volume = trt.volume(resolved_shape)
                host_mem = cuda.pagelocked_empty(volume, np_binding_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings[i] = int(device_mem)
                
                np.copyto(host_mem, np.ascontiguousarray(input_data_np).ravel()) # Ensure contiguous and copy
                trt_input_allocs.append(HostDeviceMem(host_mem, device_mem))
                current_trt_input_idx += 1
        
        # Prepare TRT output buffers
        # Output shapes are determined by TRT after inputs are set
        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                binding_name = engine.get_binding_name(i)
                np_binding_dtype = trt.nptype(engine.get_binding_dtype(i))
                output_shape_trt = context.get_binding_shape(i) # Get shape after inputs are set
                if any(d < 0 for d in output_shape_trt):
                     raise RuntimeError(f"Dynamic dimension for TRT output '{binding_name}' not resolved. Shape: {output_shape_trt}")
                
                print(f"  TRT Output Binding {i} ('{binding_name}'): Shape from context: {output_shape_trt}, Dtype: {np_binding_dtype}")
                volume = trt.volume(output_shape_trt)
                host_mem = cuda.pagelocked_empty(volume, np_binding_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings[i] = int(device_mem)
                trt_output_allocs.append(HostDeviceMem(host_mem, device_mem))

        stream = cuda.Stream()
        print("Running TensorRT inference...")
        trt_raw_results_flat = do_inference_trt(context, bindings, trt_input_allocs, trt_output_allocs, stream)

        # Reshape TRT outputs and store in a dictionary by binding name
        trt_results_dict = {}
        current_trt_output_idx = 0
        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                binding_name = engine.get_binding_name(i)
                # Shape was determined when allocating output buffer
                output_shape_trt = context.get_binding_shape(i) 
                raw_flat_output = trt_raw_results_flat[current_trt_output_idx]
                trt_results_dict[binding_name] = raw_flat_output.reshape(output_shape_trt)
                current_trt_output_idx += 1
        
        # 5. Compare Outputs
        print("\nComparing ONNX and TensorRT outputs...")
        all_match = True
        
        if len(onnx_output_names) != len(trt_results_dict):
            status_message = (f"FAILURE: Output count mismatch. "
                              f"ONNX: {len(onnx_output_names)} outputs ({onnx_output_names}), "
                              f"TRT: {len(trt_results_dict)} outputs ({list(trt_results_dict.keys())})")
            all_match = False
        else:
            comparison_details = []
            for onnx_out_name in onnx_output_names:
                if onnx_out_name not in trt_results_dict:
                    comparison_details.append(f"Output '{onnx_out_name}': MISMATCH (Not found in TRT outputs).")
                    all_match = False
                    continue

                onnx_val = onnx_results_dict[onnx_out_name]
                trt_val = trt_results_dict[onnx_out_name]

                if onnx_val.shape != trt_val.shape:
                    comparison_details.append(f"Output '{onnx_out_name}': MISMATCH (Shape diff: ONNX {onnx_val.shape}, TRT {trt_val.shape}).")
                    all_match = False
                    continue
                
                # Cast to float32 for comparison to handle potential dtype differences (e.g. TRT FP16 vs ONNX FP32)
                onnx_comp = onnx_val.astype(np.float32)
                trt_comp = trt_val.astype(np.float32)

                match = np.allclose(trt_comp, onnx_comp, rtol=args.rtol, atol=args.atol)
                if not match:
                    all_match = False
                    diff = np.abs(trt_comp - onnx_comp)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    num_mismatched_elements = np.sum(~np.isclose(trt_comp, onnx_comp, rtol=args.rtol, atol=args.atol))
                    comparison_details.append(
                        f"Output '{onnx_out_name}': MISMATCH. Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Mismatched elements: {num_mismatched_elements}/{diff.size}"
                    )
                    # print(f"  ONNX sample ({onnx_out_name}): {onnx_val.flatten()[:5]}")
                    # print(f"  TRT sample ({onnx_out_name}): {trt_val.flatten()[:5]}")
                else:
                    comparison_details.append(f"Output '{onnx_out_name}': MATCH")
            
            if all_match:
                status_message = "SUCCESS: Outputs match.\n" + "\n".join(comparison_details)
            else:
                status_message = "FAILURE: Output mismatch.\n" + "\n".join(comparison_details)

    except FileNotFoundError as e:
        status_message = f"FAILURE: File not found - {e}"
        print(f"\nERROR: {status_message}")
    except RuntimeError as e:
        status_message = f"FAILURE: Runtime error - {e}"
        print(f"\nERROR: {status_message}")
    except Exception as e:
        status_message = f"FAILURE: An unexpected error occurred - {type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {status_message}")
    finally:
        # 6. Write Status
        with open(args.output_status_file, "w") as f:
            f.write(status_message)
        print(f"\nVerification status written to: {args.output_status_file}")
        print("--- Final Status ---")
        print(status_message)
        print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TensorRT engine against ONNX model.")
    parser.add_argument("--engine_path", required=True, help="Path to the TensorRT engine (.plan) file.")
    parser.add_argument("--onnx_path", required=True, help="Path to the ONNX (.onnx) model file.")
    parser.add_argument("--output_status_file", required=True, help="Path to write the verification status.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for np.allclose comparison.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for np.allclose comparison.")
    parser.add_argument("--onnx_gpu", action='store_true', help="Use GPU (CUDAExecutionProvider) for ONNX Runtime if available.")
    
    parsed_args = parser.parse_args()
    main(parsed_args)
