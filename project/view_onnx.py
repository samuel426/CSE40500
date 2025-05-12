import onnx

# ONNX 모델 로드
model = onnx.load("onnx_models/GRU/KOSPI.onnx")

# 입력 텐서 정보 출력
print("Inputs:")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    dims = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")

# 출력 텐서 정보 출력
print("\nOutputs:")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}")
    dims = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")

