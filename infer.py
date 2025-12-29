import cv2
import torch
import numpy as np
import pathlib

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def infer_save_result(model, img_path):
    raw_img = cv2.imread(img_path)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy

    # depth: HxW float numpy (from model.infer_image)
    d = depth.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # use percentile for robust normalize, prevent outlier influence
    lo, hi = np.percentile(d, [1, 99])
    d_norm = (d - lo) / (hi - lo + 1e-6)
    d_norm = np.clip(d_norm, 0, 1)

    depth_u8 = (d_norm * 255).astype(np.uint8)
    res_path = img_path.parent / f"{img_path.stem}_depth_u8.png"
    cv2.imwrite(res_path, depth_u8)
    print(f"saved: {res_path}")

    res_path = img_path.parent / f"{img_path.stem}_depth_u16.png"
    depth_u16 = (d_norm * 65535).astype(np.uint16)
    cv2.imwrite(res_path, depth_u16)
    print(f"saved: {res_path}")


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

img_path_list = pathlib.Path('assets/examples').glob('*.jpg')
for img_path in img_path_list:
    infer_save_result(model, img_path)

import ipdb; ipdb.set_trace()
