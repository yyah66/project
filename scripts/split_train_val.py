"""
将 JSONL 文件随机切分为训练集和验证集
用法: python3 scripts/split_train_val.py --input data/public/docvqa/docvqa_train.jsonl --output-dir data/training/docvqa --ratio 0.9
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="切分 JSONL 为训练/验证集")
    parser.add_argument("--input", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集比例 (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"文件不存在: {input_path}")

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    random.seed(args.seed)
    random.shuffle(rows)

    split = int(len(rows) * args.ratio)
    train_rows = rows[:split]
    val_rows = rows[split:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem  # e.g. docvqa_train
    train_path = output_dir / f"{base}_train.jsonl"
    val_path = output_dir / f"{base}_val.jsonl"

    for path, data in [(train_path, train_rows), (val_path, val_rows)]:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{input_path.name}: {len(train_rows)} train + {len(val_rows)} val -> {output_dir}")


if __name__ == "__main__":
    main()
