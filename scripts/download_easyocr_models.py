#!/usr/bin/env python3
"""EasyOCR 模型下载脚本 — 穷举国内所有可用源。

用法:
  python3 scripts/download_easyocr_models.py
"""
import os
import sys
import subprocess
from pathlib import Path

FILES = {
    "craft_mlt_25k.pth": "检测模型",
    "zh_sim_g2.pth":     "中文识别模型",
    "english_g2.pth":    "英文识别模型",
}

TARGET = Path.home() / ".EasyOCR" / "model"

# GitHub Releases 原始路径
GITHUB_BASE = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3"


def build_urls(filename: str) -> list[str]:
    """穷举所有可能的下载源（按可靠性排序）。"""
    return [
        # ── GitHub 代理（国内最常用）──
        f"https://ghproxy.com/{GITHUB_BASE}/{filename}",
        f"https://gh-proxy.com/{GITHUB_BASE}/{filename}",
        f"https://mirror.ghproxy.com/{GITHUB_BASE}/{filename}",
        f"https://gh.con.sh/{GITHUB_BASE}/{filename}",
        f"https://gh.api.99988866.xyz/{GITHUB_BASE}/{filename}",
        f"https://github.moeyy.xyz/{GITHUB_BASE}/{filename}",
        f"https://gh.ddlc.top/{GITHUB_BASE}/{filename}",
        f"https://gh2.yanqishui.work/{GITHUB_BASE}/{filename}",
        f"https://download.fastgit.org/JaidedAI/EasyOCR/releases/download/v1.3/{filename}",
        f"https://download.nju.edu.cn/github/JaidedAI/EasyOCR/releases/download/v1.3/{filename}",
        # ── HF 镜像 ──
        f"https://hf-mirror.com/JaidedAI/EasyOCR/resolve/main/{filename}",
        f"https://huggingface.co/JaidedAI/EasyOCR/resolve/main/{filename}",
        # ── ModelScope ──
        f"https://modelscope.cn/models/easyocr/EasyOCR/resolve/master/{filename}",
        # ── GitHub raw 替代 ──
        f"https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/{filename}",
        f"https://raw.gitmirror.com/JaidedAI/EasyOCR/master/{filename}",
        # ── 直连（最后手段）──
        f"{GITHUB_BASE}/{filename}",
    ]


def try_wget(url: str, dest: Path) -> bool:
    try:
        r = subprocess.run(
            ["wget", "-q", "--show-progress", "--timeout=15", "--tries=2",
             "-O", str(dest), url],
            timeout=45, stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0 and dest.stat().st_size > 10000
    except Exception:
        return False


def try_curl(url: str, dest: Path) -> bool:
    try:
        r = subprocess.run(
            ["curl", "-L", "--connect-timeout", "10", "--max-time", "60",
             "-o", str(dest), url],
            timeout=90, stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0 and dest.stat().st_size > 10000
    except Exception:
        return False


def try_modelscope_sdk(filename: str, dest: Path) -> bool:
    """通过 ModelScope Python SDK 下载（国内最可靠）。"""
    try:
        import modelscope
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "modelscope", "-q"],
            timeout=60,
        )
        try:
            import modelscope  # noqa: F811
        except ImportError:
            return False

    try:
        from modelscope.hub.file_download import model_file_download
        local = model_file_download(
            model_id="easyocr/EasyOCR",
            file_path=filename,
            cache_dir=str(TARGET.parent),
        )
        if local and Path(local).exists():
            # 复制到 EasyOCR 期望的路径
            import shutil
            shutil.copy2(local, dest)
            return dest.stat().st_size > 10000
    except Exception:
        pass
    return False


def download_one(name: str) -> bool:
    dest = TARGET / name
    urls = build_urls(name)

    # 策略 1: 逐个 URL 尝试 wget → curl
    for i, url in enumerate(urls):
        label = url.split("/")[2].split(".")[-2]  # 域名简写
        if i < 9:
            label = f"代理{i+1}"
        elif i < 11:
            label = "HF镜像"
        elif i < 12:
            label = "ModelScope直链"
        elif i < 14:
            label = "Raw镜像"
        else:
            label = "直连"

        print(f"    [{label}] {url[:90]}...", end=" ", flush=True)
        if try_wget(url, dest) or try_curl(url, dest):
            print(f"OK ({dest.stat().st_size/1024/1024:.1f}MB)")
            return True
        print("失败")

    # 策略 2: ModelScope SDK
    print("    [ModelScope SDK] 尝试 API 下载...", end=" ", flush=True)
    if try_modelscope_sdk(name, dest):
        print(f"OK ({dest.stat().st_size/1024/1024:.1f}MB)")
        return True
    print("失败")

    return False


def main() -> int:
    print("=" * 60)
    print(f"  EasyOCR 模型下载")
    print(f"  源: 17 个国内代理 + ModelScope SDK")
    print(f"  目标: {TARGET}")
    print("=" * 60)

    TARGET.mkdir(parents=True, exist_ok=True)

    failed = []
    for name, desc in FILES.items():
        dest = TARGET / name
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"\n[跳过] {name} ({dest.stat().st_size/1024/1024:.1f}MB)")
            continue

        print(f"\n[下载] {name} — {desc}")
        if not download_one(name):
            dest.unlink(missing_ok=True)
            print(f"  ✗ 全部源均失败")
            failed.append(name)

    print(f"\n{'='*60}")
    if failed:
        print(f"  失败 {len(failed)}/{len(FILES)} 个文件")
        print(f"  最后方案: 从本地上传")
        for f in failed:
            print(f"    {TARGET / f}")
        print(f"{'='*60}")
        return 1
    else:
        print(f"  全部就绪！")
        print(f"  运行: python3 scripts/check_gpu_ocr.py")
        print(f"{'='*60}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
