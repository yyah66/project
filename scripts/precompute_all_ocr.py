"""
对所有 JSONL 数据集预计算 OCR，输出增强版 JSONL + 合并 train/val/test。

GPU 加速模式（推荐 — 利用 RTX 5090）:
  python scripts/precompute_all_ocr.py --gpu

指定后端:
  python scripts/precompute_all_ocr.py --gpu --gpu-backend paddle     # PaddleOCR GPU
  python scripts/precompute_all_ocr.py --gpu --gpu-backend easyocr    # EasyOCR GPU

传统 CPU 多进程模式（无 GPU 时使用）:
  python scripts/precompute_all_ocr.py
  python scripts/precompute_all_ocr.py --workers 96
  python scripts/precompute_all_ocr.py --max-side 1024

运行前先诊断 GPU OCR 兼容性:
  python scripts/check_gpu_ocr.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MAX_SIDE = 1600

# EasyOCR 模型下载地址（国内访问 GitHub 不通时用 HF 镜像）
_EASYOCR_ALT_BASE = "https://hf-mirror.com/EasyOCR/EasyOCR/resolve/main"


def _ensure_easyocr_models() -> None:
    """确保 EasyOCR 模型已下载。如果默认下载失败，尝试 HF 镜像。"""
    import urllib.request

    target = os.path.expanduser("~/.EasyOCR/model")
    os.makedirs(target, exist_ok=True)

    models = ["craft_mlt_25k.pth", "zh_sim_g2.pth", "english_g2.pth"]
    for name in models:
        path = os.path.join(target, name)
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            continue
        url = f"{_EASYOCR_ALT_BASE}/{name}"
        print(f"  预下载 EasyOCR 模型: {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"    完成 ({os.path.getsize(path) / 1024 / 1024:.1f}MB)")
        except Exception:
            pass  # 下载失败不阻塞，让 EasyOCR 自己再试一次

# ---------------------------------------------------------------------------
# CPU 多进程 worker（保持原有逻辑不变）
# ---------------------------------------------------------------------------

_worker_ocr = None


def _worker_init(project_root_str: str) -> None:
    """fork 后子进程首先执行：设 sys.path、限制 OpenMP 线程、加载 CPU OCR。"""
    import sys as _sys
    _sys.path.insert(0, project_root_str)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

    global _worker_ocr
    from src.ocr import OCRService
    _worker_ocr = OCRService(providers=["CPUExecutionProvider"])

    # 预热：跑一次空图消除首张 ONNX 编译延迟
    from PIL import Image as _Image
    try:
        _worker_ocr._extract_with_rapidocr(
            _Image.new("RGB", (640, 480), "white"))
    except Exception:
        pass


def _ocr_worker(payload: tuple) -> tuple[int, dict | None, str]:
    """处理单张图片（在子进程中调用）。返回 (原始索引, item_dict, 状态)。"""
    idx, image_path_str, question, answer, evidence, max_ocr_lines, max_side = payload
    global _worker_ocr

    from src.ocr import OCRResult, ImageAnalysis
    from PIL import Image

    img_path = Path(image_path_str)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path
    if not img_path.exists():
        alt = Path(PROJECT_ROOT) / image_path_str
        if alt.exists():
            img_path = alt
        else:
            return idx, None, "missing"

    try:
        img = Image.open(str(img_path))
        img.load()
    except Exception:
        return idx, None, "corrupt"

    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    try:
        lines = _worker_ocr._extract_with_rapidocr(img)
        lines = _worker_ocr._normalize_lines(lines)
        analysis = _worker_ocr.analyze_image(img, lines)
        ocr_result = OCRResult(
            backend="rapidocr-onnxruntime" if lines else "rapidocr-onnxruntime",
            lines=lines, analysis=analysis,
        )
    except Exception:
        ocr_result = OCRResult(
            backend="none", lines=[],
            analysis=ImageAnalysis(
                scene_type="未知", text_density=0.0,
                likely_table=False, likely_chart=False, likely_document=False,
                language_hint="", notes=["OCR 全部失败"],
            ),
        )

    item = _build_item(image_path_str, question, answer, ocr_result, max_ocr_lines)
    if evidence:
        item["evidence"] = evidence
    return idx, item, "ok"


# ---------------------------------------------------------------------------
# GPU OCR 引擎
# ---------------------------------------------------------------------------

_gpu_ocr_engine = None
_gpu_backend_name = ""


def _init_paddleocr_gpu() -> object:
    """初始化 PaddleOCR GPU 引擎。"""
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="ch")


def _init_easyocr_gpu(skip_download: bool = False) -> object:
    """初始化 EasyOCR GPU 引擎（预下载模型以加速首次加载）。"""
    if not skip_download:
        _ensure_easyocr_models()
    import easyocr
    return easyocr.Reader(["ch_sim", "en"], gpu=True)


def _init_gpu_ocr(backend: str, max_side: int, skip_easyocr_download: bool = False) -> tuple[object, str]:
    """初始化 GPU OCR 引擎，返回 (engine, backend_name)。失败时 raise。"""
    global _gpu_ocr_engine, _gpu_backend_name

    print(f"初始化 GPU OCR 引擎 (backend={backend})...")

    # 对 PaddlePaddle 设置环境变量以减少 ONNX 子进程干扰
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    if backend == "paddle":
        try:
            engine = _init_paddleocr_gpu()
            _gpu_ocr_engine = engine
            _gpu_backend_name = "paddleocr-gpu"
            # 预热
            from PIL import Image
            import numpy as np
            _ = engine.ocr(
                np.array(Image.new("RGB", (640, 480), "white")), cls=True)
            print(f"  PaddleOCR GPU 初始化完成")
            return engine, _gpu_backend_name
        except Exception as e:
            raise RuntimeError(
                f"PaddleOCR GPU 初始化失败: {e}\n"
                f"请先运行 scripts/check_gpu_ocr.py 诊断，"
                f"或尝试 --gpu-backend easyocr"
            ) from e

    elif backend == "easyocr":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用")
            engine = _init_easyocr_gpu(skip_download=skip_easyocr_download)
            _gpu_ocr_engine = engine
            _gpu_backend_name = "easyocr-gpu"
            print(f"  EasyOCR GPU 初始化完成")
            return engine, _gpu_backend_name
        except Exception as e:
            raise RuntimeError(
                f"EasyOCR GPU 初始化失败: {e}\n"
                f"请确保已安装: pip install easyocr"
            ) from e

    else:
        raise ValueError(f"未知 GPU 后端: {backend}，可选: paddle, easyocr")


def _run_paddleocr_gpu(engine: object, img_array) -> list:
    """用 PaddleOCR GPU 推理单张图片，返回 OCRLine 列表。"""
    from src.ocr import OCRLine, OCRService

    result = engine.ocr(img_array, cls=True)
    if not result or not result[0]:
        return []

    lines: list = []
    for item in result[0]:
        if not item or len(item) < 2:
            continue
        box = item[0]
        text_info = item[1]
        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
            text = str(text_info[0]).strip()
            try:
                confidence = float(text_info[1])
            except Exception:
                confidence = None
        else:
            text = str(text_info).strip()
            confidence = None
        if not text:
            continue
        bbox = OCRService._bbox_from_polygon(box)
        lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence))

    return lines


def _run_easyocr_gpu(reader: object, img_array) -> list:
    """用 EasyOCR GPU 推理单张图片，返回 OCRLine 列表。"""
    from src.ocr import OCRLine

    result = reader.readtext(img_array)
    lines: list = []
    for bbox, text, confidence in result:
        text = str(text).strip()
        if not text:
            continue
        try:
            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
        except Exception:
            x_min = y_min = x_max = y_max = 0
        lines.append(OCRLine(
            text=text,
            bbox=(x_min, y_min, x_max, y_max),
            confidence=confidence,
        ))
    return lines


def _process_single_gpu(
    sample: dict,
    engine: object,
    backend: str,
    max_ocr_lines: int,
    max_side: int,
) -> dict | None:
    """GPU 模式下处理单张图片，返回增强后的 item 或 None。"""
    from src.ocr import OCRResult, ImageAnalysis, OCRService
    from PIL import Image
    import numpy as np

    image_path_str = sample["image_path"]
    question = sample.get("question", sample.get("query", ""))
    answer = sample.get("answer", "")
    evidence = sample.get("evidence", None)

    img_path = Path(image_path_str)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path
    if not img_path.exists():
        alt = Path(PROJECT_ROOT) / image_path_str
        if alt.exists():
            img_path = alt
        else:
            return None

    try:
        img = Image.open(str(img_path))
        img.load()
    except Exception:
        return None

    img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    arr = np.array(img)

    try:
        if backend == "paddle":
            lines = _run_paddleocr_gpu(engine, arr)
        elif backend == "easyocr":
            lines = _run_easyocr_gpu(engine, arr)
        else:
            return None
    except Exception:
        lines = []

    lines = OCRService._normalize_lines(lines)
    analysis = OCRService.analyze_image(img, lines)

    ocr_result = OCRResult(
        backend=_gpu_backend_name,
        lines=lines,
        analysis=analysis,
    )

    item = _build_item(image_path_str, question, answer, ocr_result, max_ocr_lines)
    if evidence:
        item["evidence"] = evidence
    return item


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def _build_item(
    image_path: str,
    question: str,
    answer: str,
    ocr_result,
    max_ocr_lines: int,
) -> dict:
    """从 OCR 结果构建增强后的 item dict。"""
    return {
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "ocr_lines": [
            {
                "text": line.text,
                "bbox": list(line.bbox) if line.bbox else None,
                "confidence": line.confidence,
            }
            for line in ocr_result.lines[:max_ocr_lines]
        ],
        "ocr_analysis": {
            "scene_type": ocr_result.analysis.scene_type if ocr_result.analysis else "未知",
            "text_density": ocr_result.analysis.text_density if ocr_result.analysis else 0.0,
            "likely_table": ocr_result.analysis.likely_table if ocr_result.analysis else False,
            "likely_chart": ocr_result.analysis.likely_chart if ocr_result.analysis else False,
            "likely_document": ocr_result.analysis.likely_document if ocr_result.analysis else False,
            "language_hint": ocr_result.analysis.language_hint if ocr_result.analysis else "",
            "notes": ocr_result.analysis.notes if ocr_result.analysis else [],
        },
    }


# ---------------------------------------------------------------------------
# Helpers (main process)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def discover_jsonl_files(root: str) -> list[Path]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted(p.rglob("*.jsonl"))


# ---------------------------------------------------------------------------
# CPU 多进程路径（原有逻辑）
# ---------------------------------------------------------------------------

def precompute_single_jsonl_cpu(
    input_path: Path,
    output_path: Path,
    max_ocr_lines: int,
    max_side: int,
    workers: int,
) -> int:
    samples = load_jsonl(input_path)
    if not samples:
        write_jsonl(output_path, [])
        return 0

    tasks: list[tuple] = []
    for i, s in enumerate(samples):
        tasks.append((
            i,
            s["image_path"],
            s.get("question", s.get("query", "")),
            s.get("answer", ""),
            s.get("evidence", None),
            max_ocr_lines,
            max_side,
        ))

    enhanced: list[dict | None] = [None] * len(tasks)
    skipped = 0
    corrupt = 0

    if workers <= 1:
        _worker_init(str(PROJECT_ROOT))
        from tqdm import tqdm as _tqdm
        for t in _tqdm(tasks, desc=input_path.name):
            idx, item, status = _ocr_worker(t)
            if status == "missing":
                skipped += 1
            elif status == "corrupt":
                corrupt += 1
            else:
                enhanced[idx] = item
    else:
        import multiprocessing as mp
        from tqdm import tqdm as _tqdm
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(PROJECT_ROOT),),
        ) as executor:
            futures = {executor.submit(_ocr_worker, t): t[0] for t in tasks}
            for fut in _tqdm(as_completed(futures), total=len(tasks), desc=input_path.name):
                idx, item, status = fut.result()
                if status == "missing":
                    skipped += 1
                elif status == "corrupt":
                    corrupt += 1
                else:
                    enhanced[idx] = item

    result = [item for item in enhanced if item is not None]
    write_jsonl(output_path, result)
    print(f"  → {len(result)} 条 (跳过缺失 {skipped}, 损坏 {corrupt}) → {output_path}")
    return len(result)


# ---------------------------------------------------------------------------
# GPU 路径（新增）
# ---------------------------------------------------------------------------

def precompute_single_jsonl_gpu(
    input_path: Path,
    output_path: Path,
    max_ocr_lines: int,
    max_side: int,
    gpu_backend: str,
    checkpoint_every: int = 100,
    skip_easyocr_download: bool = False,
) -> int:
    """GPU 单进程加速预计算单个 JSONL。"""
    samples = load_jsonl(input_path)
    if not samples:
        write_jsonl(output_path, [])
        return 0

    # 初始化 GPU 引擎（只做一次）
    engine, backend_name = _init_gpu_ocr(gpu_backend, max_side,
                                             skip_easyocr_download=skip_easyocr_download)

    # 断点续跑：检查已完成的
    completed_indices: set[int] = set()
    if output_path.exists():
        existing = load_jsonl(output_path)
        # 用 (image_path, question) 匹配
        existing_keys = {(e["image_path"], e["question"]) for e in existing}
        for i, s in enumerate(samples):
            key = (s["image_path"], s.get("question", s.get("query", "")))
            if key in existing_keys:
                completed_indices.add(i)
        if completed_indices:
            print(f"  断点续跑: 已完成 {len(completed_indices)}/{len(samples)} 条")

    from tqdm import tqdm as _tqdm

    results: list[dict] = (
        load_jsonl(output_path) if output_path.exists() and completed_indices
        else []
    )
    skipped = 0
    corrupt = 0

    pbar = _tqdm(total=len(samples), desc=input_path.name, initial=len(completed_indices))

    for i, sample in enumerate(samples):
        if i in completed_indices:
            continue

        item = _process_single_gpu(sample, engine, gpu_backend, max_ocr_lines, max_side)
        if item is None:
            corrupt += 1
        else:
            results.append(item)

        pbar.update(1)

        # 定期保存检查点
        if (i + 1) % checkpoint_every == 0:
            write_jsonl(output_path, results)

    pbar.close()
    write_jsonl(output_path, results)

    print(f"  → {len(results)} 条 (损坏/缺失 {corrupt})  [{backend_name}] → {output_path}")
    return len(results)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(description="预计算所有数据集 OCR 并合并")
    parser.add_argument("--output-dir", default="data/training/merged")
    parser.add_argument("--max-ocr-lines", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                        help=f"OCR 时图片最大边长，默认 {DEFAULT_MAX_SIDE}，降至 1024 可提速")
    parser.add_argument("--workers", type=int, default=min(cpu_count // 4, 64),
                        help=f"CPU 模式并行 worker 数，默认 min(cpu_count/4, 64)={min(cpu_count // 4, 64)}")
    parser.add_argument("--gpu", action="store_true",
                        help="启用 GPU 加速 OCR（推荐：利用 RTX 5090）")
    parser.add_argument("--gpu-backend", default="easyocr", choices=["paddle", "easyocr"],
                        help="GPU OCR 后端: easyocr (PyTorch) 或 paddle (PaddleOCR, 需 3.0+)，默认 easyocr")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="GPU 模式下每 N 张图片保存一次检查点，默认 100")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-easyocr-download", action="store_true",
                        help="跳过 EasyOCR 模型自动下载（已手动下载模型时使用）")
    parser.add_argument("--single", default=None,
                        help="只处理单个 JSONL 文件（调试用）")
    args = parser.parse_args()

    use_gpu = args.gpu
    gpu_backend = args.gpu_backend

    # ---- 打印配置 ----
    if use_gpu:
        print(f"配置: GPU 模式, backend={gpu_backend}, max_side={args.max_side}, "
              f"max_ocr_lines={args.max_ocr_lines}")
        print(f"目标 GPU: RTX 5090 32GB")
        if gpu_backend == "paddle":
            print("注意: 确保 PaddlePaddle-GPU 已安装且支持当前 CUDA 版本")
            print("  如果 PaddleOCR 初始化失败，请尝试 --gpu-backend easyocr")
            print("  或运行 scripts/check_gpu_ocr.py 诊断")
    else:
        print(f"配置: CPU 多进程模式, workers={args.workers}, max_side={args.max_side}, "
              f"max_ocr_lines={args.max_ocr_lines}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 单文件模式 ----
    if args.single:
        inp = Path(args.single)
        if not inp.exists():
            raise SystemExit(f"文件不存在: {inp}")
        out = outdir / f"{inp.stem}_enhanced.jsonl"
        if use_gpu:
            precompute_single_jsonl_gpu(inp, out, args.max_ocr_lines,
                                        args.max_side, gpu_backend,
                                        args.checkpoint_every,
                                        skip_easyocr_download=args.skip_easyocr_download)
        else:
            precompute_single_jsonl_cpu(inp, out, args.max_ocr_lines,
                                        args.max_side, args.workers)
        return

    # ---- 仅处理 merged train/val/test ----
    chartqa_dir = Path("data/training/chartqa")
    jsonl_files: list[Path] = sorted([
        chartqa_dir / "merged_train.jsonl",
        chartqa_dir / "merged_val.jsonl",
        chartqa_dir / "merged_test.jsonl",
    ])
    jsonl_files = [f for f in jsonl_files if f.exists()]

    print(f"\n处理 {len(jsonl_files)} 个 JSONL:")
    for f in jsonl_files:
        samples = load_jsonl(f)
        print(f"  - {f}  ({len(samples)} 条)")

    # ---- 逐文件处理 ----
    t_start = time.time()
    enhanced_files: list[Path] = []

    for inp in jsonl_files:
        out = outdir / f"{inp.stem}_enhanced.jsonl"

        # 断点续跑：检查是否已完成
        if out.exists():
            existing = load_jsonl(out)
            original = load_jsonl(inp)
            existing_keys = {
                (e["image_path"], e.get("question", e.get("query", "")))
                for e in existing
            }
            original_keys = {
                (s["image_path"], s.get("question", s.get("query", "")))
                for s in original
            }
            if existing_keys == original_keys:
                print(f"\n跳过（已完成且无变化）: {out}")
                enhanced_files.append(out)
                continue
            elif existing_keys:
                print(f"\n断点续跑: {inp} （已完成 {len(existing_keys)}/{len(original_keys)}）")

        print(f"\n处理: {inp}")

        if use_gpu:
            count = precompute_single_jsonl_gpu(
                inp, out, args.max_ocr_lines, args.max_side,
                gpu_backend, args.checkpoint_every,
                skip_easyocr_download=args.skip_easyocr_download,
            )
        else:
            count = precompute_single_jsonl_cpu(
                inp, out, args.max_ocr_lines, args.max_side, args.workers,
            )

        if count > 0:
            enhanced_files.append(out)

    elapsed = time.time() - t_start
    print(f"\nOCR 预计算总耗时: {elapsed/60:.1f} 分钟")

    # ---- 合并与切分 ----
    all_samples: list[dict] = []
    for ef in enhanced_files:
        all_samples.extend(load_jsonl(ef))

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for s in all_samples:
        key = (s["image_path"], s["question"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    print(f"\n合并: {len(all_samples)} → {len(deduped)} 条 "
          f"(去重 {len(all_samples) - len(deduped)})")

    random.seed(args.seed)
    random.shuffle(deduped)
    n_test = max(1, int(len(deduped) * args.test_split))
    n_val = max(1, int(len(deduped) * args.val_split))
    test = deduped[:n_test]
    val = deduped[n_test:n_test + n_val]
    train = deduped[n_test + n_val:]

    train_path = outdir / "merged_train_enhanced.jsonl"
    val_path = outdir / "merged_val_enhanced.jsonl"
    test_path = outdir / "merged_test_enhanced.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    print(f"\n训练集: {len(train)} → {train_path}")
    print(f"验证集: {len(val)} → {val_path}")
    print(f"测试集: {len(test)} → {test_path}")
    print(f"完成。总耗时: {elapsed/60:.1f} 分钟")

    pkg = {
        "total": len(deduped),
        "train": len(train), "val": len(val), "test": len(test),
        "datasets": {ef.stem: len(load_jsonl(ef)) for ef in enhanced_files},
        "ocr_backend": "paddleocr-gpu" if (use_gpu and gpu_backend == "paddle")
                       else "easyocr-gpu" if (use_gpu and gpu_backend == "easyocr")
                       else "rapidocr-cpu",
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    (outdir / "merged_package.json").write_text(
        json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
