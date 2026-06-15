"""RapidOCR GPU 速度诊断"""
import time, numpy as np, os
from PIL import Image

import onnxruntime as ort
print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")
print(f"Device: {ort.get_device()}")

# ---------- RapidOCR 加载 ----------
from rapidocr_onnxruntime import RapidOCR

t0 = time.time()
ocr = RapidOCR()
print(f"\nRapidOCR 加载: {time.time()-t0:.1f}s")

# ---------- 检查内部 session provider ----------
model_attrs = ["det_model", "cls_model", "rec_model", "_det_model", "_cls_model", "_rec_model"]
for attr in model_attrs:
    obj = getattr(ocr, attr, None)
    if obj is not None:
        sess = getattr(obj, "session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            print(f"  {attr} providers: {sess.get_providers()}")
            break
else:
    print("  无法获取内部 session provider（可能未暴露）")

# ---------- 速度测试 ----------
print("\n=== 速度测试 (5 次) ===")
img = Image.new("RGB", (1600, 1200), "white")
arr = np.array(img)

times = []
for i in range(5):
    t0 = time.time()
    result, _ = ocr(arr)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  第{i+1}次: {elapsed:.2f}s")
print(f"  均值: {sum(times)/len(times):.2f}s")

# ---------- GPU 信息 ----------
print("\n=== GPU 状态 ===")
os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
