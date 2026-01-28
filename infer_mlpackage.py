import coremltools as ct
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import cv2
import math

def run_inference(model_path, image_path):
    """
    Runs inference on a single CoreML model.
    """
    model_name = os.path.basename(model_path)
    print(f"[INFO] Loading model: {model_name}...")
    
    try:
        # Load Model
        # model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.ALL)
        model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        
        # Get Input Specs
        input_spec = model.input_description._fd_spec[0]
        input_name = input_spec.name
        
        if input_spec.type.WhichOneof('Type') == 'imageType':
            h = input_spec.type.imageType.height
            w = input_spec.type.imageType.width
            input_size = (w, h)
        else:
            input_size = (518, 392)

        # Preprocess
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        original_image = Image.open(image_path).convert('RGB')
        input_image = original_image.resize(input_size, Image.Resampling.BILINEAR)

        # Predict
        print(f"[INFO] Running inference on {model_name}...")
        prediction = model.predict({input_name: input_image})

        # Extract Output
        output_key = list(prediction.keys())[0]
        output_data = prediction[output_key]
        
        if isinstance(output_data, Image.Image):
            depth_map = np.array(output_data)
        elif isinstance(output_data, np.ndarray):
            depth_map = output_data.squeeze()
        else:
            depth_map = np.array(output_data)
            
        return original_image, depth_map

    except Exception as e:
        print(f"[ERR] Failed to run {model_name}: {e}")
        return None, None

def normalize_depth(depth_map):
    """
    Normalizes depth map to 0.0 - 1.0 for visualization.
    """
    d_min = depth_map.min()
    d_max = depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    model_list = [
        "/Users/julian/model-weights/depth-model/distill_any_depth_fp16_224_224.mlpackage",
        "/Users/julian/model-weights/depth-model/DepthAnythingV2SmallF16.mlpackage",
        "/Users/julian/model-weights/depth-model/distill_any_depth_w8a16_224_224.mlpackage",
        "/Users/julian/model-weights/depth-model/DepthAnythingV2Small_w8a16_fixed.mlpackage",
        "/Users/julian/model-weights/depth-model/distill_any_depth_w8a8_224_224.mlpackage",
        "/Users/julian/model-weights/depth-model/DepthAnythingV2Small_w8a8.mlpackage",
    ]
    # ---------------------

    parser = argparse.ArgumentParser(description="Compare Multiple CoreML Depth Models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="comparison_grid.png", help="Path to save output png")
    args = parser.parse_args()
    
    results = []
    original_img_ref = None 
    
    # 1. Run Inference on Models
    print(f"Starting comparison on {len(model_list)} models...")
    
    for model_path in model_list:
        if not os.path.exists(model_path):
            print(f"[SKIP] Model file not found: {model_path}")
            continue
            
        orig_img, raw_depth = run_inference(model_path, args.image)
        
        if raw_depth is not None:
            if original_img_ref is None:
                original_img_ref = orig_img
            
            # Resize & Normalize
            orig_w, orig_h = orig_img.size
            depth_resized = cv2.resize(raw_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            depth_norm = normalize_depth(depth_resized)
            
            results.append({
                "name": os.path.basename(model_path),
                "data": depth_norm
            })

    # 2. Check for Ground Truth (.npy)
    gt_data_norm = None
    
    # Assumes gt is named "image.npy" in the same folder as "image.jpg"
    base_name = os.path.splitext(args.image)[0]
    gt_path = base_name + ".npy"
    
    if os.path.exists(gt_path):
        print(f"[INFO] Found Ground Truth: {gt_path}")
        try:
            gt_arr = np.load(gt_path).astype(np.float32)
            if gt_arr.ndim == 3: gt_arr = gt_arr.squeeze()
            
            # Resize GT to match image (just in case)
            if original_img_ref:
                orig_w, orig_h = original_img_ref.size
                gt_arr = cv2.resize(gt_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            gt_arr = 1.0/gt_arr
            
            gt_data_norm = normalize_depth(gt_arr)
        except Exception as e:
            print(f"[WARN] Failed to load GT: {e}")
    else:
        print("[INFO] No Ground Truth (.npy) found. Leaving slot blank.")

    # 3. Grid Plotting
    if results and original_img_ref:
        # Layout Logic:
        # Slot 1: Original
        # Slot 2: GT (or Blank)
        # Slot 3+: Models
        
        total_slots = 2 + len(results)
        cols = 2
        rows = math.ceil(total_slots / cols)
        
        print(f"[INFO] Plotting grid: {rows} rows x {cols} cols")
        plt.figure(figsize=(10, 5 * rows))
        
        # --- Plot 1: Original Image ---
        plt.subplot(rows, cols, 1)
        plt.title("Original Input")
        plt.imshow(original_img_ref)
        plt.axis("off")
        
        # --- Plot 2: Ground Truth OR Blank ---
        plt.subplot(rows, cols, 2)
        if gt_data_norm is not None:
            plt.title("Ground Truth", fontweight='bold')
            plt.imshow(gt_data_norm, cmap='inferno')
            plt.axis("off")
        else:
            # Leave blank
            plt.axis("off")
        
        # --- Plot 3..N: Model Results ---
        # Models start at index 3 (which is 2+1 in python list, but subplot uses 1-based index so it is 3)
        current_plot_idx = 3
        
        for res in results:
            plt.subplot(rows, cols, current_plot_idx)
            # Make title smaller so long filenames fit
            plt.title(res["name"], fontsize=8) 
            plt.imshow(res["data"], cmap='inferno')
            plt.axis("off")
            current_plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"[SUCCESS] Comparison grid saved to: {args.output}")
        plt.close()
    else:
        print("[ERR] No valid results to plot.")