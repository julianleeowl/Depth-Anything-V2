#!/usr/bin/env python3
"""
TensorRT PTQ (INT8) build from ONNX with calibration images.

Example:
  python build_trt_int8.py \
    --onnx depth_anything_v2.onnx \
    --calib-dir ./calib_images \
    --engine depth_anything_v2_int8.engine \
    --input-name input \
    --input-shape 1x3x518x518 \
    --batch 8 \
    --calib-batches 50 \
    --fp16
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import tensorrt as trt

# PyCUDA is commonly used for TRT Python calibration
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def parse_shape(s: str) -> Tuple[int, int, int, int]:
    # "1x3x518x518" -> (1,3,518,518)
    parts = s.lower().replace(" ", "").split("x")
    if len(parts) != 4:
        raise ValueError(f"--input-shape must be like 1x3xHxW, got: {s}")
    return tuple(int(p) for p in parts)  # type: ignore


def list_images(calib_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = []
    for p in calib_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def preprocess_image(
    img_path: Path,
    hw: Tuple[int, int],
    to_rgb: bool = True,
) -> np.ndarray:
    """
    IMPORTANT: You MUST match your model's training/inference preprocessing.

    This default does:
      - load image
      - resize to (W,H)
      - RGB
      - scale to [0,1]
      - CHW float32
      - ImageNet mean/std normalize (common, but may differ for your model)
    """
    H, W = hw
    img = Image.open(img_path)
    if to_rgb:
        img = img.convert("RGB")
    img = img.resize((W, H), Image.BILINEAR)

    x = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))

    # ImageNet normalization (adjust if your model expects something else)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = (x - mean) / std

    return np.ascontiguousarray(x, dtype=np.float32)


class ImageFolderCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        calib_dir: Path,
        input_name: str,
        input_shape: Tuple[int, int, int, int],  # NCHW (N is calibration batch)
        batch_size: int,
        max_batches: int,
        cache_file: Optional[Path] = None,
    ):
        super().__init__()

        self.calib_dir = calib_dir
        self.input_name = input_name
        self.n, self.c, self.h, self.w = input_shape
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.cache_file = cache_file

        self.image_paths = list_images(calib_dir)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {calib_dir}")

        # limit to needed count
        needed = batch_size * max_batches
        if len(self.image_paths) < needed:
            print(f"[WARN] Only {len(self.image_paths)} images found, "
                  f"but requested {needed}. Will reuse images (wrap-around).")

        # Allocate one device buffer for a batch
        self.device_input = cuda.mem_alloc(batch_size * self.c * self.h * self.w * np.float32().nbytes)

        self.batch_idx = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.batch_idx >= self.max_batches:
            return None

        # Create a batch
        batch = np.empty((self.batch_size, self.c, self.h, self.w), dtype=np.float32)

        start = self.batch_idx * self.batch_size
        for i in range(self.batch_size):
            p = self.image_paths[(start + i) % len(self.image_paths)]
            batch[i] = preprocess_image(p, (self.h, self.w))

        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)

        self.batch_idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_file and self.cache_file.exists():
            print(f"[INFO] Using calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_bytes(cache)
            print(f"[INFO] Wrote calibration cache: {self.cache_file}")


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_name: str,
    input_shape: Tuple[int, int, int, int],  # NCHW
    batch_size: int,
    calib_batches: int,
    fp16: bool,
    workspace_gb: float,
    calib_cache: Optional[Path],
):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        print("[ERROR] ONNX parse failed:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if not builder.platform_has_fast_int8:
        print("[WARN] Platform does not report fast INT8. Will still try INT8 build.")
    config.set_flag(trt.BuilderFlag.INT8)

    # Set optimization profile (fixed shape by default)
    n, c, h, w = input_shape
    if n != 1:
        # For TRT profiles, N is dynamic in explicit batch; we set min/opt/max for N too.
        # But many models assume N fixed; safest is to keep N == batch_size for build.
        pass

    profile = builder.create_optimization_profile()

    # Treat N as dynamic batch dim in explicit-batch mode:
    min_shape = (1, c, h, w)
    opt_shape = (batch_size, c, h, w)
    max_shape = (batch_size, c, h, w)

    # Make sure input exists
    inp = network.get_input(0)
    if input_name and inp.name != input_name:
        # try to find by name
        found = False
        for i in range(network.num_inputs):
            if network.get_input(i).name == input_name:
                inp = network.get_input(i)
                found = True
                break
        if not found:
            print("[WARN] --input-name not found; using first network input:", network.get_input(0).name)
            inp = network.get_input(0)

    input_name_resolved = inp.name
    print(f"[INFO] Using input tensor: {input_name_resolved}")

    profile.set_shape(input_name_resolved, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # Calibrator uses *batch_size* as calibration batch
    calibrator = ImageFolderCalibrator(
        calib_dir=Path(args.calib_dir),
        input_name=input_name_resolved,
        input_shape=(batch_size, c, h, w),
        batch_size=batch_size,
        max_batches=calib_batches,
        cache_file=calib_cache,
    )
    config.int8_calibrator = calibrator

    print("[INFO] Building engine... (this can take a while)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized engine (got None).")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)
    print(f"[OK] Wrote TensorRT engine: {engine_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--calib-dir", type=str, required=True)
    ap.add_argument("--engine", type=str, required=True)

    ap.add_argument("--input-name", type=str, default="")  # optional
    ap.add_argument("--input-shape", type=str, required=True, help="e.g. 1x3x518x518")
    ap.add_argument("--batch", type=int, default=8, help="Calibration batch size & TRT opt/max batch")
    ap.add_argument("--calib-batches", type=int, default=50, help="How many calibration batches to run")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16 fallback")
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    ap.add_argument("--calib-cache", type=str, default="", help="Path to calibration cache file")

    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    input_shape = parse_shape(args.input_shape)
    cache_path = Path(args.calib_cache) if args.calib_cache else None

    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        input_name=args.input_name,
        input_shape=input_shape,
        batch_size=args.batch,
        calib_batches=args.calib_batches,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
        calib_cache=cache_path,
    )
