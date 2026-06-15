"""PaddleOCR GPU 可用性诊断"""
import os, sys, time

# 抑制 PaddlePaddle OMP 警告
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---- 1. 检查 paddle 安装 ----
print("=== 1. PaddlePaddle 信息 ===")
try:
    import paddle
    print(f"  版本: {paddle.__version__}")
    print(f"  编译 CUDA: {paddle.is_compiled_with_cuda()}")
    gpu_count = paddle.device.cuda.device_count()
    print(f"  GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"    GPU {i}: {paddle.device.cuda.get_device_name(i)}")
        cap = paddle.device.cuda.get_device_capability(i)
        print(f"      算力: {cap[0]}.{cap[1]}")
except ImportError:
    print("  未安装 paddlepaddle")
    print("  请执行: pip install paddlepaddle-gpu")
    sys.exit(1)
except Exception as e:
    print(f"  错误: {e}")
    sys.exit(1)

if not paddle.is_compiled_with_cuda():
    print("\n  *** paddle 未编译 CUDA 支持，GPU 不可用 ***")
    print("  请先卸载 CPU 版: pip uninstall paddlepaddle")
    print("  再安装 GPU 版: pip install paddlepaddle-gpu")
    sys.exit(1)

# ---- 2. 检查 PaddleOCR ----
print("\n=== 2. PaddleOCR 信息 ===")
try:
    from paddleocr import PaddleOCR
    print("  PaddleOCR 已安装")
except ImportError:
    print("  未安装 PaddleOCR")
    print("  请执行: pip install paddleocr")
    sys.exit(1)

# ---- 3. GPU 推理速度测试 ----
print("\n=== 3. GPU 推理速度测试 ===")
from PIL import Image
import numpy as np

img_path = os.path.join(PROJECT_ROOT, "data/demo/images/document_sample.png")
if not os.path.exists(img_path):
    img = Image.new("RGB", (1280, 900), "white")
    print(f"  测试图不存在，使用空白图")
else:
    img = Image.open(img_path).convert("RGB")
    print(f"  测试图: {img_path} ({img.size})")

# GPU OCR
print("\n  --- PaddleOCR GPU ---")
# PaddleOCR 2.x 自动使用 paddle 的默认设备（已检测到 CUDA → GPU）
ocr_gpu = PaddleOCR(lang="ch")

arr = np.array(img)
for i in range(3):
    t0 = time.time()
    result = ocr_gpu.ocr(arr, cls=True)
    elapsed = time.time() - t0
    n_lines = len(result[0]) if result and result[0] else 0
    texts = [line[1][0] for line in result[0]] if result and result[0] else []
    print(f"  第{i+1}次: {elapsed:.2f}s, {n_lines} 行")
    if texts:
        print(f"    前3行: {texts[:3]}")

# ---- 4. 对比 CPU 版 RapidOCR ----
print("\n  --- 对比 RapidOCR CPU ---")
os.environ["OMP_NUM_THREADS"] = "1"
from src.ocr import OCRService
ocr_cpu = OCRService(providers=["CPUExecutionProvider"])
for i in range(2):
    t0 = time.time()
    lines = ocr_cpu._extract_with_rapidocr(img)
    elapsed = time.time() - t0
    print(f"  第{i+1}次: {elapsed:.2f}s, {len(lines)} 行")
