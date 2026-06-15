"""EasyOCR GPU 可用性诊断"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 1. 检查 torch CUDA
print("=== 1. PyTorch CUDA 状态 ===")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 2. EasyOCR 导入
print("\n=== 2. EasyOCR 信息 ===")
try:
    import easyocr
    print(f"  EasyOCR: {easyocr.__version__}")
except ImportError:
    print("  未安装，请执行: pip install easyocr")
    sys.exit(1)

# 3. 测试图片
print("\n=== 3. GPU 推理速度测试 ===")
from PIL import Image

img_path = os.path.join(PROJECT_ROOT, "data/demo/images/document_sample.png")
if not os.path.exists(img_path):
    img = Image.new("RGB", (1280, 900), "white")
    print(f"  测试图不存在，使用空白图")
    arr = None
else:
    img = Image.open(img_path).convert("RGB")
    print(f"  测试图: {img_path} ({img.size})")
    import numpy as np
    arr = np.array(img)

print("\n  --- EasyOCR GPU ---")
reader = easyocr.Reader(["ch_sim", "en"], gpu=True)

for i in range(3):
    t0 = time.time()
    result = reader.readtext(arr if arr is not None else img)
    elapsed = time.time() - t0
    texts = [text for _, text, _ in result]
    print(f"  第{i+1}次: {elapsed:.2f}s, {len(result)} 行")
    if texts:
        print(f"    前3行: {texts[:3]}")

# 4. GPU 利用率
print("\n=== 4. GPU 显存使用 ===")
if torch.cuda.is_available():
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
