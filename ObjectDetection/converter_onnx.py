#!/usr/bin/env python3
"""
convert_to_onnx.py

Usage:
  python convert_to_onnx.py --framework <FRAMEWORK> --input <INPUT_PATH> --output <OUTPUT_PATH> [--opset <OPSET>]

Supported frameworks:
  pytorch     : .pt/.pth files or TorchScript (.pt)
  tensorflow  : TensorFlow SavedModel directory
  keras       : Keras .h5 files or tf.keras models
  mxnet       : MXNet .params + JSON symbol files
"""
import sys, os, argparse

def convert_pytorch_0(input_path, output_path, opset):
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: PyTorch not installed. `pip install torch`")
    model = torch.jit.load(input_path) if input_path.endswith(".pt") and not input_path.endswith(".pth") \
            else torch.load(input_path, map_location="cpu")
    model.eval()
    # you may need to adjust dummy input shape
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, output_path,
                      opset_version=opset,
                      input_names=["input"], output_names=["output"])
    print(f"✅ PyTorch → ONNX saved to {output_path}")
    
    
def convert_pytorch(input_path, output_path, opset):
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: PyTorch not installed. `pip install torch`")

    # 1) Try to load as TorchScript archive
    model = None
    if input_path.endswith(".pt") and not input_path.endswith(".pth"):
        try:
            model = torch.jit.load(input_path).eval()
            print("✔️ Loaded TorchScript module")
        except RuntimeError:
            # 2) Fallback: treat as a checkpoint dict
            ckpt = torch.load(input_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt:
                # Ultralytics YOLOv5 style checkpoint
                model = ckpt["model"].float().eval()
                print("✔️ Loaded YOLOv5 checkpoint model")
            else:
                sys.exit(f"ERROR: Failed to load {input_path} as TorchScript or checkpoint")

    # 3) PyTorch state-dict .pth or other formats
    if model is None:
        model = torch.load(input_path, map_location="cpu")
        if hasattr(model, "eval"):
            model = model.eval()
        print("✔️ Loaded generic PyTorch object")

    # 4) Dummy input (adjust spatial dims to match your model)
    dummy = torch.randn(1, 3, 640, 640)

    # 5) Export to ONNX
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"✅ PyTorch → ONNX saved to {output_path} (opset {opset})")

def convert_tensorflow(input_path, output_path, opset):
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        sys.exit("ERROR: tf2onnx or TensorFlow not installed. `pip install tf2onnx tensorflow`")
    # assume input_path is a SavedModel directory
    spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_saved_model(
        input_path, input_signature=spec, opset=opset)
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ TensorFlow SavedModel → ONNX saved to {output_path}")

def convert_keras(input_path, output_path, opset):
    try:
        import tf2onnx
        from tensorflow import keras
        import tensorflow as tf
    except ImportError:
        sys.exit("ERROR: tf2onnx or TensorFlow/Keras not installed. `pip install tf2onnx tensorflow`")
    model = keras.models.load_model(input_path)
    # build a dummy input spec from model.input_shape
    shape = model.input_shape
    # ensure batch dim = 1
    shape = tuple(1 if d is None else d for d in shape)
    spec = (tf.TensorSpec(shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ Keras .h5 → ONNX saved to {output_path}")

def convert_mxnet(input_path, output_path, opset):
    try:
        import mxnet as mx
        from mxnet.contrib import onnx as onnx_mx
    except ImportError:
        sys.exit("ERROR: MXNet or onnx-mxnet not installed. `pip install mxnet onnx-mxnet`")
    # expect input_path without extension, e.g. model-symbol.json + model-0000.params
    prefix, epoch = os.path.splitext(input_path)[0], 0
    onnx_mx.export_model(f"{prefix}-symbol.json", f"{prefix}-0000.params",
                         [("data", (1, 3, 224, 224))], np.float32,
                         output_path, verbose=True, opset_version=opset)
    print(f"✅ MXNet → ONNX saved to {output_path}")

def main():
    p = argparse.ArgumentParser(description="Convert various models to ONNX")
    p.add_argument("--framework", required=True,
                   choices=["pytorch","tensorflow","keras","mxnet"])
    p.add_argument("--input",      required=True, help="Path to model")
    p.add_argument("--output",     required=True, help="Path for .onnx")
    p.add_argument("--opset",      type=int, default=18, help="ONNX opset version")
    args = p.parse_args()

    conv = {
        "pytorch":    convert_pytorch,
        "tensorflow": convert_tensorflow,
        "keras":      convert_keras,
        "mxnet":      convert_mxnet,
    }[args.framework]

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input path {args.input} not found")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    conv(args.input, args.output, args.opset)

if __name__ == "__main__":
    main()

