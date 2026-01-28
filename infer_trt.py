#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
import tensorrt as trt
import matplotlib.pyplot as plt

import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context


# DepthAnythingV2 preprocessing:
# - BGR -> RGB
# - scale to [0, 1]
# - normalize with ImageNet mean/std
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_to_nchw_fp16(bgr: np.ndarray, input_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    nchw = np.expand_dims(chw, axis=0)  # 1x3xHxW
    return np.ascontiguousarray(nchw.astype(np.float16))


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalizes depth map to 0.0 - 1.0 for visualization.
    """
    d_min = depth_map.min()
    d_max = depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)


def load_engine(engine_path: str, logger: trt.ILogger) -> trt.ICudaEngine:
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(
            "Failed to deserialize engine. Common causes: TensorRT/CUDA version mismatch, "
            "GPU mismatch, or corrupted engine file."
        )
    return engine


def get_io_tensors(engine: trt.ICudaEngine):
    """
    TensorRT 10+ uses the I/O tensor API (name-based) instead of legacy bindings.
    """
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)  # trt.TensorIOMode.INPUT / OUTPUT
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        (inputs if mode == trt.TensorIOMode.INPUT else outputs).append((name, dtype))

    if len(inputs) != 1:
        raise ValueError(f"Expected exactly 1 input, found {len(inputs)}: {inputs}")
    if len(outputs) < 1:
        raise ValueError("Expected at least 1 output")

    return inputs, outputs


def _is_dynamic_shape(shape):
    # shape is a trt.Dims / tuple-like; dynamic dims are -1
    return any(int(d) == -1 for d in shape)


def allocate_io(engine: trt.ICudaEngine, context: trt.IExecutionContext, input_shape):
    """
    Allocates pinned host buffers + device buffers for input and outputs.
    Binds buffers to tensor names via context.set_tensor_address (TRT10).
    """
    # (in_name, in_dtype), outputs = get_io_tensors(engine)
    inputs, outputs = get_io_tensors(engine)
    in_name, in_dtype = inputs[0]

    # Set input shape for dynamic engines
    cur_in_shape = tuple(context.get_tensor_shape(in_name))
    if cur_in_shape != tuple(input_shape):
        # If engine has dynamic shape, we must set it; if static, this will either be no-op or error.
        context.set_input_shape(in_name, tuple(input_shape))

    # Validate the input shape took effect
    in_shape = tuple(context.get_tensor_shape(in_name))
    if _is_dynamic_shape(in_shape):
        raise RuntimeError(
            f"Input shape for '{in_name}' is still dynamic after set_input_shape: {in_shape}. "
            "This usually means the engine expects shape tensors or profiles weren't set correctly."
        )

    stream = cuda.Stream()

    # Allocate input buffers
    in_vol = int(np.prod(in_shape))
    host_in = cuda.pagelocked_empty(in_vol, dtype=in_dtype)
    dev_in = cuda.mem_alloc(host_in.nbytes)

    # Allocate outputs
    host_outs, dev_outs, out_meta = [], [], []
    for out_name, out_dtype in outputs:
        out_shape = tuple(context.get_tensor_shape(out_name))
        if _is_dynamic_shape(out_shape):
            raise RuntimeError(
                f"Output shape for '{out_name}' is dynamic: {out_shape}. "
                "Ensure input shape is set and profiles are valid."
            )
        out_vol = int(np.prod(out_shape))
        host_out = cuda.pagelocked_empty(out_vol, dtype=out_dtype)
        dev_out = cuda.mem_alloc(host_out.nbytes)

        host_outs.append(host_out)
        dev_outs.append(dev_out)
        out_meta.append((out_name, out_dtype, out_shape))

    # Bind device pointers to tensor names
    context.set_tensor_address(in_name, int(dev_in))
    for dev_out, (out_name, _, _) in zip(dev_outs, out_meta):
        context.set_tensor_address(out_name, int(dev_out))

    return (in_name, host_in, dev_in, in_shape), (host_outs, dev_outs, out_meta), stream


def infer_one(engine: trt.ICudaEngine, context: trt.IExecutionContext, inp_nchw_fp16: np.ndarray):
    """
    One inference using TRT10 execute_async_v3.
    Returns list of (output_name, output_array_reshaped).
    """
    input_shape = tuple(inp_nchw_fp16.shape)  # (1,3,H,W)

    (in_name, host_in, dev_in, in_shape), (host_outs, dev_outs, out_meta), stream = allocate_io(
        engine, context, input_shape
    )

    # Basic safety check
    if int(np.prod(in_shape)) != inp_nchw_fp16.size:
        raise ValueError(f"Input size mismatch. Engine expects {in_shape} (vol={np.prod(in_shape)}), "
                         f"but got {inp_nchw_fp16.shape} (vol={inp_nchw_fp16.size}).")

    # H2D
    np.copyto(host_in, inp_nchw_fp16.ravel())
    cuda.memcpy_htod_async(dev_in, host_in, stream)

    # Execute (TRT10)
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed (execute_async_v3 returned False).")

    # D2H
    for host_out, dev_out in zip(host_outs, dev_outs):
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    # Reshape outputs
    outputs = []
    for host_out, (name, _, shape) in zip(host_outs, out_meta):
        outputs.append((name, host_out.reshape(shape)))
    return outputs


def squeeze_depth_to_hw(out: np.ndarray, name: str) -> np.ndarray:
    """
    Try common depth output layouts and return HxW float32.
    """
    depth = out
    if depth.ndim == 4 and depth.shape[1] == 1:     # [N,1,H,W]
        depth = depth[:, 0, :, :]
    if depth.ndim == 3:                              # [N,H,W]
        depth2d = depth[0]
    elif depth.ndim == 2:                            # [H,W]
        depth2d = depth
    else:
        raise ValueError(f"Unexpected output shape for {name}: {out.shape}")
    return depth2d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to TensorRT engine (.engine/.plan)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--out", default="depth_color.png", help="Output visualization path")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--verbose-io", action="store_true", help="Print engine I/O tensor info")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    engine = load_engine(args.engine, logger)
    context = engine.create_execution_context()

    if args.verbose_io:
        print("=== Engine I/O tensors ===")
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            print(
                f"{i:02d}  {n:30s}  mode={engine.get_tensor_mode(n)}  "
                f"dtype={engine.get_tensor_dtype(n)}  shape={engine.get_tensor_shape(n)}"
            )

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    h0, w0 = bgr.shape[:2]

    inp = preprocess_bgr_to_nchw_fp16(bgr, args.input_size)

    # Warmup
    for _ in range(args.warmup):
        _ = infer_one(engine, context, inp)

    # Timed runs
    t0 = time.perf_counter()
    last_out = None
    for _ in range(args.runs):
        last_out = infer_one(engine, context, inp)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / args.runs
    print(f"Avg latency: {ms:.3f} ms  ({1000.0/ms:.1f} FPS)")

    # Use first output as depth by default
    name, out = last_out[0]
    depth2d = squeeze_depth_to_hw(out, name)

    # Resize and normalize depth for visualization
    depth_resized = cv2.resize(depth2d, (w0, h0), interpolation=cv2.INTER_CUBIC)
    depth_norm = normalize_depth(depth_resized)

    # Plot grid with original image and depth map (same format as infer_mlpackage.py)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Depth Prediction")
    plt.imshow(depth_norm, cmap='inferno')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved depth visualization to: {args.out}")


if __name__ == "__main__":
    main()
