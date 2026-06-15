"""OCR 深度诊断 —— 定位 19s/张 的根本原因"""
import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
import numpy as np

# ---- 准备测试图 ----
img_path = os.path.join(PROJECT_ROOT, "data/demo/images/document_sample.png")
img = Image.open(img_path).convert("RGB")
arr = np.array(img)
print(f"图片: {img.size}, dtype={arr.dtype}, shape={arr.shape}")

# ============================================================
# 测试 1：不同 OMP_NUM_THREADS 的影响
# ============================================================
print("\n" + "=" * 60)
print("测试 1: OMP_NUM_THREADS 对推理速度的影响")
print("=" * 60)

for omp in ["1", "2", "4"]:
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = omp
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    # 预热
    ocr(arr)

    t0 = time.time()
    result, _ = ocr(arr)
    elapsed = time.time() - t0
    n_lines = len(result) if result else 0
    print(f"  OMP_NUM_THREADS={omp}: {elapsed:.2f}s, {n_lines} 行")

    del ocr
    # 强制重新加载模块以重置 ONNX
    for mod in list(sys.modules.keys()):
        if 'rapidocr' in mod or 'onnxruntime' in mod:
            del sys.modules[mod]

# ============================================================
# 测试 2：直接测 ONNX Runtime 推理耗时
# ============================================================
print("\n" + "=" * 60)
print("测试 2: ONNX Runtime 底层的单次 session.run 耗时")
print("=" * 60)

import onnxruntime as ort
print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")

# 找 RapidOCR 的模型文件
import rapidocr_onnxruntime
rapidocr_path = os.path.dirname(rapidocr_onnxruntime.__file__)
print(f"RapidOCR 安装路径: {rapidocr_path}")

# 列出模型文件
model_dir = os.path.join(rapidocr_path, "models")
if os.path.exists(model_dir):
    print(f"模型目录: {model_dir}")
    for f in sorted(os.listdir(model_dir)):
        fpath = os.path.join(model_dir, f)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")

# ============================================================
# 测试 3：Direct ONNX session（绕过 RapidOCR 的开销）
# ============================================================
print("\n" + "=" * 60)
print("测试 3: 直接加载 ONNX session 并推理")
print("=" * 60)

# 检测模型
det_model = os.path.join(model_dir, "ch_PP-OCRv4_det_infer.onnx")
rec_model = os.path.join(model_dir, "ch_PP-OCRv4_rec_infer.onnx")
cls_model = os.path.join(model_dir, "ch_ppocr_mobile_v2.0_cls_infer.onnx")

for name, model_path in [("det", det_model), ("rec", rec_model), ("cls", cls_model)]:
    if not os.path.exists(model_path):
        print(f"  {name}: 不存在 -> {model_path}")
        continue

    for provider in [["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]]:
        try:
            sess = ort.InferenceSession(model_path, providers=provider)
            actual_provider = sess.get_providers()

            # 获取输入信息
            inputs = sess.get_inputs()
            input_name = inputs[0].name
            input_shape = inputs[0].shape

            t0 = time.time()
            sess.run(None, {input_name: arr.astype(np.float32)[:100, :100, :]})
            elapsed = time.time() - t0

            print(f"  {name} [{actual_provider[0]}]: {elapsed:.4f}s (dummy input 100x100)")
        except Exception as e:
            print(f"  {name} [{provider[0]}]: ERROR - {e}")

# ============================================================
# 测试 4：检查 ONNX Runtime session options
# ============================================================
print("\n" + "=" * 60)
print("测试 4: CPU 核心数 / 可用资源")
print("=" * 60)
print(f"  os.cpu_count(): {os.cpu_count()}")
try:
    cpu_affinity = os.sched_getaffinity(0)
    print(f"  可用 CPU 核心: {sorted(cpu_affinity)}")
except Exception:
    print(f"  无法获取 CPU affinity")

print(f"  OMP_NUM_THREADS (当前): {os.environ.get('OMP_NUM_THREADS', '未设置')}")
