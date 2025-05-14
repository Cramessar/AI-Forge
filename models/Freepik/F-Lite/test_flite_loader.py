import sys
import os
from diffusers import DiffusionPipeline
import torch

# Add current directory to Python path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# ✅ Import class directly from file
from f_lite_pipeline import FLitePipeline
print("✅ Successfully imported FLitePipeline")

try:
    # ✅ Pass the class object, not a string
    pipeline = DiffusionPipeline.from_pretrained(
        current_dir,
        pipeline_class=FLitePipeline,
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print("✅ Successfully loaded custom pipeline")
except Exception as e:
    print("💥 Error while loading custom pipeline:")
    print(e)
