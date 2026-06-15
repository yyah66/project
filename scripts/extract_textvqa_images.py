"""
从 HF 缓存中提取 TextVQA JSONL 所需的图片（不会重复下载，直接用缓存）
用法: python3 scripts/extract_textvqa_images.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

JSONL_PATH = "data/public/textvqa_val_5000.jsonl"
IMG_DIR = Path("data/public/images/textvqa")


def main() -> None:
    if not Path(JSONL_PATH).exists():
        raise SystemExit(f"JSONL 不存在: {JSONL_PATH}")

    needed: set[str] = set()
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            fn = row["image_path"].rsplit("/", 1)[-1]
            if fn.endswith(".jpg"):
                needed.add(fn[:-4])

    print(f"需要 {len(needed)} 张图片")

    ds = load_dataset("textvqa", split="validation", trust_remote_code=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing = 0
    for row in ds:
        img_id = str(row["image_id"])
        if img_id not in needed:
            continue
        fpath = IMG_DIR / f"{img_id}.jpg"
        if not fpath.exists():
            row["image"].save(fpath)
            saved += 1

        if (saved + missing) % 500 == 0 or saved + missing == len(needed):
            print(f"  进度 {saved + missing}/{len(needed)} (新存 {saved})")

    print(f"完成: {saved} 张 -> {IMG_DIR}")


if __name__ == "__main__":
    main()
