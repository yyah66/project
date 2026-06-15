"""
从本地 VQA-v2 JSON 文件抽取 2000 条，按需下载 COCO 图片，生成训练 JSONL。

用法（在 AutoDL 上跑）:
  python3 scripts/prepare_vqa_v2_local.py \
    --questions-json /root/autodl-tmp/datasets/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations-json /root/autodl-tmp/datasets/vqa_v2/v2_mscoco_val2014_annotations.json \
    --num-samples 2000 \
    --output-dir data/public
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

COCO_BASE_URL = "http://images.cocodataset.org/val2014"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_image(img_dir: Path, img_id: int) -> bool:
    fname = f"COCO_val2014_{img_id:012d}.jpg"
    fpath = img_dir / fname
    if fpath.exists():
        return True
    url = f"{COCO_BASE_URL}/{fname}"
    result = subprocess.run(
        ["wget", "-q", "-O", str(fpath), url],
        timeout=30,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地 VQA-v2 JSON 生成训练子集")
    parser.add_argument("--questions-json", required=True, help="v2_OpenEnded_mscoco_val2014_questions.json 路径")
    parser.add_argument("--annotations-json", required=True, help="v2_mscoco_val2014_annotations.json 路径")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--no-download", action="store_true", help="跳过图片下载，只生成 JSONL")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    img_dir = output_dir / "images" / "vqa_v2"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    questions_data = load_json(Path(args.questions_json))
    annotations_data = load_json(Path(args.annotations_json))

    ans_map: dict[int, dict] = {}
    for a in annotations_data["annotations"]:
        ans_map[a["question_id"]] = a

    # 抽样
    random.seed(args.seed)
    all_questions = questions_data["questions"]
    if args.num_samples > len(all_questions):
        args.num_samples = len(all_questions)
    samples = random.sample(all_questions, args.num_samples)

    records: list[dict[str, Any]] = []
    need_images: set[int] = set()
    skipped = 0

    for q in samples:
        a = ans_map.get(q["question_id"])
        if a is None:
            skipped += 1
            continue
        img_id = q["image_id"]
        need_images.add(img_id)
        records.append({
            "image_path": f"data/public/images/vqa_v2/COCO_val2014_{img_id:012d}.jpg",
            "question": q["question"].strip(),
            "answer": a["multiple_choice_answer"].strip(),
            "evidence": [],
            "dataset": "vqa_v2",
            "sample_id": str(q["question_id"]),
        })

    print(f"抽样 {len(records)} 条 (跳过 {skipped} 条无标注), 涉及 {len(need_images)} 张图片")

    # 下载图片
    if not args.no_download:
        todo = sorted(i for i in need_images if not (img_dir / f"COCO_val2014_{i:012d}.jpg").exists())
        print(f"已有 {len(need_images) - len(todo)} 张, 待下载 {len(todo)} 张")

        success = 0
        for idx, img_id in enumerate(todo):
            ok = download_image(img_dir, img_id)
            if ok:
                success += 1
            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(todo)}  (成功 {success})")
        print(f"下载完成: {success}/{len(todo)} 成功, 共 {sum(1 for _ in img_dir.glob('*.jpg'))} 张")
    else:
        print("跳过图片下载 (--no-download)")

    # 写 JSONL
    output_path = output_dir / "vqa_v2_2000.jsonl"
    write_jsonl(output_path, records)
    print(f"JSONL -> {output_path}")

    # 统计
    types = {}
    for r in records:
        t = r.get("answer_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"答案类型分布: {types}")


if __name__ == "__main__":
    main()
