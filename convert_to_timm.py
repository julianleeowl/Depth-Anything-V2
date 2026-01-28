import argparse
import os
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file, save_file

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.timm_dpt import TimmDepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description='Convert original model to timm-based model')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--safetensor-path', type=str,
                        default='/mnt/model-weights/depth-model/distill-any-depth-multi-teacher-small.safetensors')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save converted model (optional)')
    parser.add_argument('--image', type=str, default='assets/examples/demo01.jpg',
                        help='Path to input image for inference comparison')
    parser.add_argument('--output-image', type=str, default='comparison_result.png',
                        help='Path to save comparison image')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs for profiling')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for profiling')
    args = parser.parse_args()

    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    config = model_configs[args.encoder]
    print(f"Using encoder: {args.encoder}")
    print(f"Config: {config}")

    # =========================================================================
    # Step 1: Load original model with safetensors
    # =========================================================================
    print("\n[Step 1] Loading original DepthAnythingV2 model...")
    original_model = DepthAnythingV2(**config)

    state_dict = load_file(args.safetensor_path, device='cpu')
    original_model.load_state_dict(state_dict)
    original_model.eval()
    print(f"Loaded weights from: {args.safetensor_path}")

    # =========================================================================
    # Step 2: Initialize timm model and copy weights
    # =========================================================================
    print("\n[Step 2] Initializing TimmDepthAnythingV2 model...")
    timm_model = TimmDepthAnythingV2(
        **config,
        input_size=args.input_size,
        pretrained_backbone=True  # Load pretrained DINOv2 from timm
    )
    timm_model.eval()

    # Copy depth_head weights from original to timm model
    print("Copying depth_head weights from original model...")

    # Get the depth_head state dict from original model
    original_head_state = {k: v for k, v in original_model.state_dict().items()
                          if k.startswith('depth_head.')}

    # Get the timm model's current state dict
    timm_state = timm_model.state_dict()

    # Update timm model's depth_head with original weights
    matched_keys = 0
    for key, value in original_head_state.items():
        if key in timm_state:
            if timm_state[key].shape == value.shape:
                timm_state[key] = value
                matched_keys += 1
            else:
                print(f"  Shape mismatch for {key}: {timm_state[key].shape} vs {value.shape}")
        else:
            print(f"  Key not found in timm model: {key}")

    print(f"Copied {matched_keys} depth_head parameters")

    # Also copy pretrained (backbone) weights from original model
    print("Copying backbone weights from original model...")
    original_backbone_state = {k: v for k, v in original_model.state_dict().items()
                               if k.startswith('pretrained.')}

    # Map original backbone keys to timm backbone keys
    # Original: pretrained.xxx -> timm: pretrained.model.xxx
    backbone_matched = 0
    backbone_mismatched = []

    for orig_key, value in original_backbone_state.items():
        # Convert key: pretrained.xxx -> pretrained.model.xxx
        timm_key = orig_key.replace('pretrained.', 'pretrained.model.')

        if timm_key in timm_state:
            if timm_state[timm_key].shape == value.shape:
                timm_state[timm_key] = value
                backbone_matched += 1
            else:
                backbone_mismatched.append((timm_key, timm_state[timm_key].shape, value.shape))
        else:
            # Try direct mapping for some keys
            pass

    print(f"Copied {backbone_matched} backbone parameters")
    if backbone_mismatched:
        print(f"Shape mismatches: {len(backbone_mismatched)}")
        for key, timm_shape, orig_shape in backbone_mismatched[:5]:
            print(f"  {key}: timm={timm_shape}, orig={orig_shape}")

    # Load the updated state dict
    timm_model.load_state_dict(timm_state)
    print("Weights loaded into timm model")

    # =========================================================================
    # Step 3: Run inference and compare results
    # =========================================================================
    print("\n[Step 3] Running inference comparison...")

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(1, 3, args.input_size, args.input_size)

    with torch.no_grad():
        original_output = original_model(test_input)
        timm_output = timm_model(test_input)

    print(f"Original output shape: {original_output.shape}")
    print(f"Timm output shape: {timm_output.shape}")

    # Compare outputs
    diff = (original_output - timm_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Check correlation
    orig_flat = original_output.flatten().numpy()
    timm_flat = timm_output.flatten().numpy()
    correlation = np.corrcoef(orig_flat, timm_flat)[0, 1]
    print(f"  Correlation coefficient: {correlation:.6f}")

    # Determine if results are acceptable
    if max_diff < 1e-5:
        print("\n✓ Results are nearly identical (max diff < 1e-5)")
    elif max_diff < 1e-3:
        print("\n✓ Results are very close (max diff < 1e-3)")
    elif correlation > 0.99:
        print("\n✓ Results are highly correlated (correlation > 0.99)")
    else:
        print("\n⚠ Results differ significantly - backbone weights may not match")
        print("  This is expected if using timm's pretrained weights instead of original backbone")

    # =========================================================================
    # Step 4: Test with real image and profile runtime
    # =========================================================================
    if not os.path.exists(args.image):
        print(f"\n[Step 4] Skipped - image not found: {args.image}")
        return original_model, timm_model

    print(f"\n[Step 4] Testing with image: {args.image}")

    raw_image = cv2.imread(args.image)
    h, w = raw_image.shape[:2]
    print(f"Image shape: {raw_image.shape}")

    # Move models to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_model = original_model.to(device)
    timm_model = timm_model.to(device)
    print(f"Device: {device}")

    # Warmup runs
    print(f"\nWarming up ({args.warmup} runs)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = original_model.infer_image(raw_image, input_size=args.input_size)
            _ = timm_model.infer_image(raw_image, input_size=args.input_size)

    # Profile original model
    print(f"Profiling original model ({args.runs} runs)...")
    original_times = []
    with torch.no_grad():
        for _ in range(args.runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            original_depth = original_model.infer_image(raw_image, input_size=args.input_size)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            original_times.append(end - start)

    # Profile timm model
    print(f"Profiling timm model ({args.runs} runs)...")
    timm_times = []
    with torch.no_grad():
        for _ in range(args.runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            timm_depth = timm_model.infer_image(raw_image, input_size=args.input_size)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            timm_times.append(end - start)

    # Print profiling results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'Mean (ms)':<15} {'Std (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}")
    print("-" * 60)
    print(f"{'DepthAnythingV2':<25} {np.mean(original_times)*1000:<15.2f} {np.std(original_times)*1000:<15.2f} {np.min(original_times)*1000:<15.2f} {np.max(original_times)*1000:<15.2f}")
    print(f"{'TimmDepthAnythingV2':<25} {np.mean(timm_times)*1000:<15.2f} {np.std(timm_times)*1000:<15.2f} {np.min(timm_times)*1000:<15.2f} {np.max(timm_times)*1000:<15.2f}")
    print("=" * 60)

    speedup = np.mean(original_times) / np.mean(timm_times)
    if speedup > 1:
        print(f"Timm model is {speedup:.2f}x faster")
    else:
        print(f"Original model is {1/speedup:.2f}x faster")

    # Compare outputs
    diff = np.abs(original_depth - timm_depth)
    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(original_depth.flatten(), timm_depth.flatten())[0, 1]:.6f}")

    # =========================================================================
    # Step 5: Save comparison image
    # =========================================================================
    print(f"\n[Step 5] Saving comparison image to: {args.output_image}")

    # Normalize depths for visualization
    def normalize_depth(depth):
        depth_min, depth_max = depth.min(), depth.max()
        return (depth - depth_min) / (depth_max - depth_min + 1e-8)

    original_depth_norm = normalize_depth(original_depth)
    timm_depth_norm = normalize_depth(timm_depth)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')

    # Original model depth
    im1 = axes[1].imshow(original_depth_norm, cmap='inferno')
    axes[1].set_title(f'DepthAnythingV2\n({np.mean(original_times)*1000:.1f} ms)', fontsize=14)
    axes[1].axis('off')

    # Timm model depth
    im2 = axes[2].imshow(timm_depth_norm, cmap='inferno')
    axes[2].set_title(f'TimmDepthAnythingV2\n({np.mean(timm_times)*1000:.1f} ms)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.output_image, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison image saved to: {args.output_image}")

    # =========================================================================
    # Optional: Save converted model
    # =========================================================================
    if args.output_path:
        print(f"\nSaving converted model to: {args.output_path}")
        save_file(timm_model.state_dict(), args.output_path)
        print("Model saved successfully")

    # =========================================================================
    # Step 6: Export timm model to ONNX
    # =========================================================================
    onnx_path = f'timm_depth_anything_v2_{args.encoder}_{args.input_size}.onnx'
    print(f"\n[Step 6] Exporting timm model to ONNX: {onnx_path}")

    # Move model to CPU for ONNX export
    device = 'cpu'
    dtype = torch.float32
    timm_model = timm_model.to(device).eval()

    # Create dummy input (same as export_v2_new.py)
    dummy_input = torch.ones((1, 3, args.input_size, args.input_size), dtype=dtype).to(device)

    # Run example forward pass
    example_output = timm_model.forward(dummy_input)

    # Export to ONNX (same settings as export_v2_new.py)
    torch.onnx.export(
        timm_model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
        export_params=True,
        do_constant_folding=True,
    )

    print(f"ONNX model exported to: {onnx_path}")

    return original_model, timm_model


if __name__ == "__main__":
    main()
