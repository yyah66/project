"""OCR 耗时诊断 —— 定位真正的瓶颈"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.ocr import OCRService
from PIL import Image
import json

# 完全模拟原始行为，不设任何特殊配置
ocr = OCRService()

# 找一张真实图片
with open(os.path.join(PROJECT_ROOT, "data/training/chartqa/chartqa_train.jsonl")) as f:
    sample = json.loads(f.readline())
img_path = sample["image_path"]
if not os.path.isabs(img_path):
    img_path = os.path.join(PROJECT_ROOT, img_path)
print(f"测试图片: {img_path}")
print(f"文件大小: {os.path.getsize(img_path) / 1024:.0f} KB")

# 1. 加载
t0 = time.time()
img = Image.open(img_path).convert("RGB")
print(f"1. 加载图片: {time.time()-t0:.2f}s, 尺寸={img.size}")

# 2. resize (模拟 precompute_all_ocr 的逻辑)
t0 = time.time()
w, h = img.size
MAX_SIDE = 1600
if max(w, h) > MAX_SIDE:
    scale = MAX_SIDE / max(w, h)
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
print(f"2. Resize: {time.time()-t0:.2f}s, 新尺寸={img.size}")

# 3. RapidOCR 首次（含模型加载 + ONNX 编译）
t0 = time.time()
lines = ocr._extract_with_rapidocr(img)
print(f"3. RapidOCR 首次(含模型加载): {time.time()-t0:.2f}s, {len(lines)} 行")

# 4. RapidOCR 第二次（纯推理）
t0 = time.time()
lines2 = ocr._extract_with_rapidocr(img)
print(f"4. RapidOCR 第二次(纯推理): {time.time()-t0:.2f}s, {len(lines2)} 行")

# 5. normalize
t0 = time.time()
lines_norm = ocr._normalize_lines(lines2)
print(f"5. Normalize: {time.time()-t0:.2f}s")

# 6. analyze
t0 = time.time()
analysis = ocr.analyze_image(img, lines_norm)
print(f"6. Analyze: {time.time()-t0:.2f}s")

# 7. 试试 CPU provider 对比
print()
print("--- 对比: CPUExecutionProvider ---")
ocr2 = OCRService(providers=["CPUExecutionProvider"])
t0 = time.time()
lines3 = ocr2._extract_with_rapidocr(img)
print(f"CPU provider 首次: {time.time()-t0:.2f}s, {len(lines3)} 行")
t0 = time.time()
lines4 = ocr2._extract_with_rapidocr(img)
print(f"CPU provider 第二次: {time.time()-t0:.2f}s, {len(lines4)} 行")

# 8. ONNX Runtime 信息
print()
print("--- ONNX Runtime 信息 ---")
import onnxruntime as ort
print(f"版本: {ort.__version__}")
print(f"可用 providers: {ort.get_available_providers()}")
