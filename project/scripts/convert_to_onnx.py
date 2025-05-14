# scripts/convert_to_onnx.py
import os, sys, importlib.util, torch, pathlib
from hyp import DEVICE, INPUT_SIZE, SEQ_LEN

ROOT = pathlib.Path(__file__).resolve().parent          # scripts/
PROJECT = ROOT.parent                                  # 프로젝트 최상위
MODEL_ROOT = PROJECT / "models"
ONNX_ROOT  = PROJECT / "onnx_models"
ONNX_ROOT.mkdir(exist_ok=True)

def load_class(file_path: pathlib.Path, class_name: str):
    """파일 경로로부터 클래스 로드 (패키지 여부 무관)."""
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore
    return getattr(mod, class_name)

def export(model, dst):
    model.eval()
    dummy = torch.randn(1, 1, SEQ_LEN, INPUT_SIZE, device=DEVICE)
    torch.onnx.export(model, dummy, dst,
                      input_names=["input"],
                      output_names=["pred"],
                      opset_version=13,
                      export_params=True)
    print("✅", dst)

def process(group):
    src_dir = MODEL_ROOT / group
    dst_dir = ONNX_ROOT  / group
    dst_dir.mkdir(exist_ok=True)

    # 학습 스크립트 위치
    script_file = ROOT / f"train_{group.lower()}.py"
    ModelCls = load_class(script_file, f"Price{group}")

    for pth in src_dir.glob("*.pth"):
        model = ModelCls().to(DEVICE)
        model.load_state_dict(torch.load(pth, map_location=DEVICE))
        export(model, dst_dir / f"{pth.stem}.onnx")

if __name__ == "__main__":
    for g in ["GRU", "LSTM", "BiLSTM"]:
        process(g)
