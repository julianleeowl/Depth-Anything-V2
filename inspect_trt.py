import json
import tensorrt as trt
from pathlib import Path

engine_path = Path("distill_any_depth_multi_teacher_small_vits_518_int8.engine")
out_path = engine_path.with_suffix(".engine_inspector.json")

logger = trt.Logger(trt.Logger.INFO)
with engine_path.open("rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())

inspector = engine.create_engine_inspector()

# This returns a JSON string
info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)

# Save as pretty JSON (falls back to raw string if it's not valid JSON)
try:
    info = json.loads(info_str)
    out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
except json.JSONDecodeError:
    out_path.write_text(info_str, encoding="utf-8")

print(f"Saved: {out_path}")
