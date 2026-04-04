from pathlib import Path

import onnx
import onnxruntime as ort
import torch as tch

import models as mdl


def export_model(model: tch.nn.Module, model_path: Path) -> None:
    model.eval()

    dummy_input = tch.rand(1, 4, 512, 512, dtype=tch.float32)

    tch.onnx.export(
        model,
        dummy_input,
        model_path,
        dynamo=True,
        external_data=False,    
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy_input.numpy()})
    print("ORT output shapes:", [x.shape for x in out])


if __name__ == "__main__":
    DEVICE = tch.device("cuda" if tch.cuda.is_available() else "cpu")

    checkpoint_path = Path("checkpoints/33_01-04-2026_13_17_41.pth")

    checkpoint = tch.load(checkpoint_path, map_location=DEVICE)

    model_path = Path(__file__).parent / f"{checkpoint_path.stem}.onnx"

    model = mdl.AlphaGenerator().to(DEVICE).eval()

    model.load_state_dict(checkpoint["model_state"])

    export_model(model, model_path)
