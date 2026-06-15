"""
从本地 TextVQA JSON 文件生成训练 JSONL，按需下载 OpenImages 图片。

用法（在 AutoDL 上跑）:
  python3 scripts/prepare_textvqa_local.py \
    --annotations-json /root/autodl-tmp/datasets/textvqa/TextVQA_0.5.1_val.json \
    --num-samples 5000 \
    --output-dir data/public
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

# OpenImages v6 图片的 Google Cloud Storage HTTP 地址
GCS_IMAGE_URL = "https://storage.googleapis.com/openimages/v6/images/{image_id}.jpg"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_image(img_dir: Path, image_id: str) -> bool:
    fname = f"{image_id}.jpg"
    fpath = img_dir / fname
    if fpath.exists():
        return True
    url = GCS_IMAGE_URL.format(image_id=image_id)
    result = subprocess.run(
        ["wget", "-q", "-O", str(fpath), url],
        timeout=30,
    )
    return result.returncode == 0


def extract_answer(record: dict[str, Any]) -> str:
    answers = record.get("answers", [])
    if answers:
        return answers[0].strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地 TextVQA JSON 生成训练子集")
    parser.add_argument("--annotations-json", required=True, help="TextVQA_0.5.1_val.json 路径")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--no-download", action="store_true", help="跳过图片下载，只生成 JSONL")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    img_dir = output_dir / "images" / "textvqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注
    data = load_json(Path(args.annotations_json))
    entries = data.get("data", data)

    random.seed(args.seed)
    if args.num_samples < len(entries):
        entries = random.sample(entries, args.num_samples)

    records: list[dict[str, Any]] = []
    need_images: set[str] = set()
    skipped = 0

    for entry in entries:
        answer = extract_answer(entry)
        if not answer:
            skipped += 1
            continue
        question = entry.get("question", "").strip()
        if not question:
            skipped += 1
            continue
        image_id = str(entry.get("image_id", ""))
        if not image_id:
            skipped += 1
            continue
        need_images.add(image_id)
        records.append({
            "image_path": f"data/public/images/textvqa/{image_id}.jpg",
            "question": question,
            "answer": answer,
            "evidence": entry.get("ocr_tokens", []),
            "dataset": "textvqa",
            "sample_id": str(entry.get("question_id", "")),
        })

    print(f"有效 {len(records)} 条 (跳过 {skipped}), 需 {len(need_images)} 张图片")

    # 下载图片
    if not args.no_download:
        todo = sorted(i for i in need_images if not (img_dir / f"{i}.jpg").exists())
        print(f"已有 {len(need_images) - len(todo)} 张, 待下载 {len(todo)} 张")

        success = 0
        for idx, image_id in enumerate(todo):
            ok = download_image(img_dir, image_id)
            if ok:
                success += 1
            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(todo)}  (成功 {success})")
        print(f"下载完成: {success}/{len(todo)}, 共 {sum(1 for _ in img_dir.glob('*.jpg'))} 张")
    else:
        print("跳过图片下载 (--no-download)")

    # 写 JSONL
    output_path = output_dir / "textvqa_val_5000.jsonl"
    write_jsonl(output_path, records)
    print(f"JSONL -> {output_path}")


if __name__ == "__main__":
    main()
