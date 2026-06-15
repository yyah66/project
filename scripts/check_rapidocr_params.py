"""测试 RapidOCR 能否指定 CUDA provider"""
import onnxruntime as ort
print("ONNX Providers:", ort.get_available_providers())

from rapidocr_onnxruntime import RapidOCR
try:
    ocr = RapidOCR(
        det_providers=["CUDAExecutionProvider"],
        rec_providers=["CUDAExecutionProvider"],
        cls_providers=["CUDAExecutionProvider"],
    )
    print("CUDA providers 设置成功")

    # 测速
    import time, numpy as np
    from PIL import Image
    img = Image.new("RGB", (1600, 1200), "white")
    arr = np.array(img)
    times = []
    for i in range(5):
        t0 = time.time()
        result, _ = ocr(arr)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  第{i+1}次: {elapsed:.2f}s")
    print(f"  均值: {sum(times[1:])/len(times[1:]):.2f}s (去掉第1次)")

    import os
    print("\nGPU 状态:")
    os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
except Exception as e:
    print(f"Error: {e}")
