"""
GPU OCR 兼容性诊断与速度基准测试

在运行 OCR 预计算前执行此脚本，确认 GPU OCR 后端可用且稳定。
用法:
  python scripts/check_gpu_ocr.py
  python scripts/check_gpu_ocr.py --image-path data/demo/images/document_sample.png
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def _get_model_dir(subdir: str) -> str:
    """获取模型存储目录。如果 setup_autodl.sh 运行过则优先数据盘，否则使用默认路径。"""
    data_path = f"/root/autodl-tmp/cache/{subdir}"
    if os.path.isdir("/root/autodl-tmp"):
        os.makedirs(data_path, exist_ok=True)
        return data_path
    return os.path.expanduser(f"~/.{subdir}")


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_cuda() -> dict:
    """检查 CUDA / PyTorch GPU 基础环境。"""
    sep("1. CUDA / PyTorch 基础环境")

    info: dict = {"cuda_ok": False, "gpu_name": "", "vram_gb": 0, "sm": ""}

    try:
        import torch
    except ImportError:
        print(f"  {FAIL} PyTorch 未安装")
        return info

    print(f"  PyTorch {torch.__version__}  |  CUDA {torch.version.cuda or 'N/A'}")

    if not torch.cuda.is_available():
        print(f"  {FAIL} torch.cuda.is_available() = False")
        print(f"      请确认 AutoDL 镜像已包含 CUDA 版 PyTorch")
        return info

    info["cuda_ok"] = True
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    cap_major = torch.cuda.get_device_capability(0)
    cap_minor = torch.cuda.get_device_capability(0)
    # get_device_capability returns (major, minor) tuple
    cap = torch.cuda.get_device_capability(0)
    info["sm"] = f"{cap[0]}.{cap[1]}"

    print(f"  {PASS} GPU: {info['gpu_name']}")
    print(f"  {PASS} 显存: {info['vram_gb']:.1f} GB")
    print(f"  {PASS} 计算能力: sm_{info['sm']}")

    # 检查 cuDNN
    try:
        _ = torch.backends.cudnn.version()
        print(f"  {PASS} cuDNN {torch.backends.cudnn.version()}")
    except Exception:
        print(f"  {FAIL} cuDNN 不可用")

    # 检查 bf16 支持（RTX 5090 应支持）
    if torch.cuda.is_bf16_supported():
        print(f"  {PASS} bf16 支持")
    else:
        print(f"  {WARN} bf16 不支持（训练可能使用 fp16 代替）")

    return info


def check_paddle_gpu() -> dict:
    """检查 PaddlePaddle GPU 是否可用。"""
    sep("2. PaddlePaddle GPU")

    info: dict = {"paddle_ok": False, "paddle_version": "", "gpu_count": 0}

    try:
        import paddle
        info["paddle_version"] = paddle.__version__
        print(f"  版本: {paddle.__version__}")
    except ImportError:
        print(f"  {FAIL} PaddlePaddle 未安装")
        print(f"      安装: pip install paddlepaddle-gpu")
        return info

    if not paddle.is_compiled_with_cuda():
        print(f"  {FAIL} PaddlePaddle 未编译 CUDA 支持")
        print(f"      当前安装的是 CPU 版本，请卸载后安装 GPU 版:")
        print(f"        pip uninstall paddlepaddle")
        print(f"        pip install paddlepaddle-gpu")
        return info

    try:
        gpu_count = paddle.device.cuda.device_count()
        info["gpu_count"] = gpu_count
        print(f"  {PASS} GPU 数量: {gpu_count}")
        for i in range(min(gpu_count, 2)):
            name = paddle.device.cuda.get_device_name(i)
            print(f"      GPU {i}: {name}")
    except Exception as e:
        print(f"  {FAIL} 获取 GPU 信息失败: {e}")
        return info

    # 快速推理测试
    try:
        paddle.set_device("gpu:0")
        x = paddle.randn([1, 3, 224, 224])
        y = x * 2 + 1
        _ = y.numpy()
        print(f"  {PASS} GPU 张量运算正常")
        info["paddle_ok"] = True
    except Exception as e:
        print(f"  {FAIL} GPU 张量运算失败: {e}")
        print(f"      这通常表示 CUDA/cuDNN 版本不匹配")
        print(f"      对于 RTX 5090 (Blackwell)，需要 CUDA >= 12.8")
        print(f"      请在 AutoDL 选择包含 CUDA 12.8+ 的镜像")

    return info


def check_paddleocr_gpu() -> dict:
    """检查 PaddleOCR 能否在 GPU 上运行。"""
    sep("3. PaddleOCR GPU 推理测试")

    info: dict = {"paddleocr_ok": False, "speed_s": 0, "n_lines": 0}

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print(f"  {FAIL} PaddleOCR 未安装")
        print(f"      安装: pip install paddleocr")
        return info

    # 初始化（PaddlePaddle GPU 版会自动使用 GPU）
    print("  初始化 PaddleOCR (GPU)...")

    ocr = None
    init_errors = []

    # 尝试不同参数组合（兼容新老版本 PaddleOCR）
    init_attempts = [
        ("最新 API", lambda: PaddleOCR(lang="ch")),
    ]

    for label, factory in init_attempts:
        try:
            t0 = time.time()
            ocr = factory()
            init_time = time.time() - t0
            print(f"  {PASS} 初始化成功 ({init_time:.1f}s)")
            break
        except TypeError as e:
            init_errors.append(f"{label}: TypeError - {e}")
            continue
        except Exception as e:
            init_errors.append(f"{label}: {e}")
            continue

    if ocr is None:
        print(f"  {FAIL} 所有初始化方式均失败:")
        for err in init_errors:
            print(f"      {err}")
        return info
        print(f"      可能是 PaddlePaddle 版本与 PaddleOCR 不兼容")
        return info

    # 获取一张真实图片测试
    from PIL import Image
    import numpy as np

    test_img = _find_test_image()
    if test_img is None:
        img = Image.new("RGB", (1280, 900), "white")
        print(f"  {WARN} 无测试图片，使用空白图")
    else:
        img = Image.open(str(test_img)).convert("RGB")
        w, h = img.size
        print(f"  测试图: {test_img.name} ({w}x{h})")

    arr = np.array(img)

    print("  运行 3 次推理...")
    times = []
    for i in range(3):
        t0 = time.time()
        result = ocr.ocr(arr, cls=True)
        elapsed = time.time() - t0
        times.append(elapsed)
        n_lines = len(result[0]) if result and result[0] else 0
        texts = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)):
                        texts.append(str(text_info[0]))
                    else:
                        texts.append(str(text_info))
        print(f"    第{i+1}次: {elapsed:.2f}s, {n_lines} 行")
        if texts and i == 0:
            preview = texts[:3]
            print(f"      前3行: {preview}")

    avg_time = sum(times) / len(times)
    info["speed_s"] = avg_time
    info["n_lines"] = n_lines if result and result[0] else 0
    info["paddleocr_ok"] = True

    # 估算 5000 张图片耗时
    est_total = avg_time * 5000
    print(f"\n  {PASS} PaddleOCR GPU 可用!")
    print(f"  平均单张: {avg_time:.2f}s")
    print(f"  预估 5000 张: {est_total/60:.1f} 分钟")

    return info


def check_easyocr_gpu() -> dict:
    """检查 EasyOCR GPU 是否可用（作为 PaddleOCR 的备选）。"""
    sep("4. EasyOCR GPU (备选后端)")

    info: dict = {"easyocr_ok": False, "speed_s": 0}

    try:
        import torch
        if not torch.cuda.is_available():
            print(f"  {FAIL} CUDA 不可用，跳过")
            return info
    except ImportError:
        print(f"  {FAIL} PyTorch 未安装，跳过")
        return info

    try:
        import easyocr
    except ImportError:
        print(f"  {WARN} EasyOCR 未安装 (备选方案，非必须)")
        print(f"       安装: pip install easyocr")
        return info

    import numpy as np
    from PIL import Image

    print(f"  初始化 EasyOCR (ch_sim + en, GPU)...")
    try:
        t0 = time.time()
        reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
        init_time = time.time() - t0
        print(f"  {PASS} 初始化成功 ({init_time:.1f}s)")
    except Exception as e:
        print(f"  {FAIL} 初始化失败: {e}")
        return info

    test_img = _find_test_image()
    if test_img is None:
        img = Image.new("RGB", (1280, 900), "white")
    else:
        img = Image.open(str(test_img)).convert("RGB")

    arr = np.array(img)

    print("  运行 2 次推理...")
    times = []
    for i in range(2):
        t0 = time.time()
        result = reader.readtext(arr)
        elapsed = time.time() - t0
        times.append(elapsed)
        texts = [text for _, text, _ in result]
        print(f"    第{i+1}次: {elapsed:.2f}s, {len(result)} 行")
        if texts and i == 0:
            print(f"      前3行: {texts[:3]}")

    avg_time = sum(times) / len(times)
    info["speed_s"] = avg_time
    info["easyocr_ok"] = True

    est_total = avg_time * 5000
    print(f"\n  {PASS} EasyOCR GPU 可用!")
    print(f"  平均单张: {avg_time:.2f}s")
    print(f"  预估 5000 张: {est_total/60:.1f} 分钟")

    return info


def check_rapidocr_speed() -> dict:
    """测试现有 RapidOCR CPU 速度作为基线。"""
    sep("5. RapidOCR CPU 基线速度")

    info: dict = {"rapidocr_speed_s": 0}

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        print(f"  {WARN} RapidOCR 未安装")
        return info

    import numpy as np
    from PIL import Image

    test_img = _find_test_image()
    if test_img is None:
        img = Image.new("RGB", (1280, 900), "white")
    else:
        img = Image.open(str(test_img)).convert("RGB")

    arr = np.array(img)

    ocr = RapidOCR()
    times = []
    for i in range(3):
        t0 = time.time()
        result, _ = ocr(arr)
        elapsed = time.time() - t0
        times.append(elapsed)
        n = len(result) if result else 0
        print(f"    第{i+1}次: {elapsed:.2f}s, {n} 行")

    avg = sum(times) / len(times)
    info["rapidocr_speed_s"] = avg
    est_total = avg * 5000
    print(f"\n  平均单张: {avg:.2f}s")
    print(f"  预估 5000 张: {est_total/60:.1f} 分钟")
    print(f"  (实际多进程会更快，此处为单进程基线)")

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_test_image() -> Path | None:
    """查找一张可用的测试图片。"""
    candidates = [
        PROJECT_ROOT / "data" / "demo" / "images" / "document_sample.png",
        PROJECT_ROOT / "data" / "demo" / "images" / "chart_sample.png",
    ]
    # 也搜索 training 目录下任意图片
    training_dir = PROJECT_ROOT / "data" / "training"
    if training_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in training_dir.rglob(ext):
                candidates.append(p)
                break
            if len(candidates) > 2:
                break

    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU OCR 兼容性诊断")
    parser.add_argument("--image-path", default=None, help="指定测试图片路径")
    args = parser.parse_args()

    print("=" * 60)
    print("  GPU OCR 兼容性诊断")
    print("  目标: RTX 5090 / AutoDL 环境")
    print("=" * 60)

    # ---- 1. CUDA 基础 ----
    cuda_info = check_cuda()
    if not cuda_info["cuda_ok"]:
        print("\n" + "!" * 60)
        print("  CUDA 不可用！请选择包含 GPU 驱动的 AutoDL 镜像。")
        print("!" * 60)
        return

    # ---- 2. PaddlePaddle GPU ----
    paddle_info = check_paddle_gpu()

    # ---- 3. PaddleOCR GPU 测试 ----
    paddleocr_info: dict = {"paddleocr_ok": False, "speed_s": 0}
    if paddle_info["paddle_ok"]:
        paddleocr_info = check_paddleocr_gpu()

    # ---- 4. EasyOCR GPU (备选) ----
    easyocr_info: dict = {"easyocr_ok": False, "speed_s": 0}
    if not paddleocr_info.get("paddleocr_ok"):
        print("\n  (PaddleOCR GPU 不可用，尝试 EasyOCR 备选...)")
        easyocr_info = check_easyocr_gpu()

    # ---- 5. RapidOCR 基线 ----
    rapidocr_info = check_rapidocr_speed()

    # ---- 总结与建议 ----
    sep("总结与建议")

    print(f"  环境: {cuda_info['gpu_name']} ({cuda_info['vram_gb']:.1f} GB, sm_{cuda_info['sm']})")

    if paddleocr_info.get("paddleocr_ok"):
        print(f"\n  *** 推荐使用 PaddleOCR GPU ***")
        print(f"  单张速度: {paddleocr_info['speed_s']:.2f}s")
        print(f"  预计 5000 张耗时: {paddleocr_info['speed_s'] * 5000 / 60:.1f} 分钟")
        print(f"\n  运行命令:")
        print(f"    python scripts/precompute_all_ocr.py --gpu --gpu-backend paddle")
    elif easyocr_info.get("easyocr_ok"):
        print(f"\n  *** PaddleOCR GPU 不可用，使用 EasyOCR GPU 备选 ***")
        print(f"  单张速度: {easyocr_info['speed_s']:.2f}s")
        print(f"  预计 5000 张耗时: {easyocr_info['speed_s'] * 5000 / 60:.1f} 分钟")
        print(f"\n  运行命令:")
        print(f"    python scripts/precompute_all_ocr.py --gpu --gpu-backend easyocr")
    else:
        print(f"\n  {WARN} GPU OCR 不可用，将回退到 CPU RapidOCR 多进程")
        print(f"  PaddlePaddle GPU 问题排查:")
        print(f"    1. 确认 AutoDL 镜像 CUDA 版本 >= 12.8 (RTX 5090 Blackwell 需要)")
        print(f"    2. pip install paddlepaddle-gpu (不要安装 paddlepaddle)")
        print(f"    3. 检查 PaddlePaddle 官网是否已支持 sm_{cuda_info['sm']}")
        print(f"  如果 PaddlePaddle 暂不支持 Blackwell，请使用 EasyOCR 备选:")
        print(f"    pip install easyocr")
        print(f"    python scripts/precompute_all_ocr.py --gpu --gpu-backend easyocr")
        print(f"\n  或继续使用 CPU 版本 (慢):")
        print(f"    python scripts/precompute_all_ocr.py")

    print(f"\n  速度对比:")
    p_speed = paddleocr_info.get("speed_s", 0)
    e_speed = easyocr_info.get("speed_s", 0)
    r_speed = rapidocr_info.get("rapidocr_speed_s", 0)
    if p_speed > 0:
        print(f"    PaddleOCR GPU: {p_speed:.2f}s/张")
    if e_speed > 0:
        print(f"    EasyOCR GPU:   {e_speed:.2f}s/张")
    if r_speed > 0:
        print(f"    RapidOCR CPU:  {r_speed:.2f}s/张 (单进程)")
    print()


if __name__ == "__main__":
    main()
