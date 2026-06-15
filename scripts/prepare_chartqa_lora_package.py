"""
所有数据集 → LoRA 训练包（自动扫描 data/public/ 和 data/training/ 下所有 JSONL）

用法:
  python scripts/prepare_chartqa_lora_package.py \
    --output-dir data/training/chartqa \
    --val-split 0.1

输出:
  data/training/chartqa/
    chartqa_train.jsonl
    chartqa_val.jsonl
    chartqa_package.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def discover_jsonl_files(root: Path) -> list[Path]:
    """递归扫描目录下所有 .jsonl 文件。"""
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 LoRA 训练数据（自动扫描）")
    parser.add_argument("--output-dir", default="data/training/chartqa", help="输出目录")
    parser.add_argument("--prefix", default="chartqa", help="输出文件前缀")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-split", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动扫描所有 JSONL
    jsonl_files: list[Path] = []
    for scan_dir in ["data/public", "data/training"]:
        jsonl_files.extend(discover_jsonl_files(PROJECT_ROOT / scan_dir))

    print(f"发现 {len(jsonl_files)} 个 JSONL 文件:")
    for f in jsonl_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")

    all_samples: list[dict] = []
    stats: dict[str, int] = {}
    skipped_missing_image = 0

    for jsonl_path in jsonl_files:
        records = load_jsonl(jsonl_path)
        source_name = str(jsonl_path.parent.relative_to(PROJECT_ROOT)).replace("/", "_").replace("\\", "_")

        count = 0
        for record in records:
            image_path_str = record.get("image_path", "")
            if not image_path_str:
                continue

            # 解析图片路径：相对于项目根目录
            img_path = PROJECT_ROOT / image_path_str
            # 如果路径不存在，尝试在当前 JSONL 同级 images 目录下查找
            if not img_path.exists():
                alt = jsonl_path.parent / "images" / Path(image_path_str).name
                if alt.exists():
                    img_path = alt
                else:
                    # 再尝试全局 data 目录搜索
                    candidates = list((PROJECT_ROOT / "data").rglob(Path(image_path_str).name))
                    if candidates:
                        img_path = candidates[0]
                    else:
                        skipped_missing_image += 1
                        continue

            sample = {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "question": record.get("question", "").strip(),
                "answer": record.get("answer", "").strip(),
                "source": source_name,
            }
            all_samples.append(sample)
            count += 1

        stats[source_name] = count
        print(f"  {source_name}: {count} 条")

    # 去重 (按 image_path + question)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for s in all_samples:
        key = (s["image_path"], s["question"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    removed = len(all_samples) - len(deduped)
    if removed:
        print(f"  去重移除: {removed} 条")
    all_samples = deduped
    print(f"  总计: {len(all_samples)} 条")
    if skipped_missing_image:
        print(f"  跳过图片缺失: {skipped_missing_image} 条")

    # 打乱并三刀切分 train / val / test
    random.shuffle(all_samples)
    test_count = max(1, int(len(all_samples) * args.test_split))
    val_count = max(1, int(len(all_samples) * args.val_split))
    test_samples = all_samples[:test_count]
    val_samples = all_samples[test_count:test_count + val_count]
    train_samples = all_samples[test_count + val_count:]

    train_path = output_dir / f"{args.prefix}_train.jsonl"
    val_path = output_dir / f"{args.prefix}_val.jsonl"
    test_path = output_dir / f"{args.prefix}_test.jsonl"
    write_jsonl(train_path, train_samples)
    write_jsonl(val_path, val_samples)
    write_jsonl(test_path, test_samples)
    print(f"\n训练集: {len(train_samples)} 条 -> {train_path}")
    print(f"验证集: {len(val_samples)} 条 -> {val_path}")
    print(f"测试集: {len(test_samples)} 条 -> {test_path}")

    # 生成 package.json
    package = {
        "total": len(all_samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "sources": stats,
    }
    package_path = output_dir / f"{args.prefix}_package.json"
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"元数据: {package_path}")


if __name__ == "__main__":
    main()
