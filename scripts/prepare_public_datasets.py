from __future__ import annotations

import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFY", "1")

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image


DATASET_MAP = {
    "vqa_v2": "HuggingFaceM4/VQAv2",
    "textvqa": "textvqa",
    "docvqa": "nielsr/docvqa_1200_examples",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def coalesce(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            v = record[key]
            # 多语言 dict（如 query={'en': '...', 'de': '...'}），取英文
            if isinstance(v, dict) and "en" in v:
                return v["en"]
            return v
    return None


def extract_answer(record: dict[str, Any]) -> str:
    direct = coalesce(record, "answer", "multiple_choice_answer", "label", "target")
    # 如果 answer 是 dict（如 {'text': '1/8/93'})，提取 text
    if isinstance(direct, dict):
        for k in ("text", "answer", "label"):
            if isinstance(direct.get(k), str) and direct[k].strip():
                return direct[k].strip()
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    answers = coalesce(record, "answers", "answer_list", "gt_answers")
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            for key in ("answer", "text", "label"):
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    if isinstance(answers, dict):
        for key in ("answer", "text", "label"):
            value = answers.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def extract_evidence(record: dict[str, Any]) -> list[str]:
    for key in ("evidence", "ocr_tokens", "ocr_text", "ocr", "text_tokens"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        if isinstance(value, list):
            extracted: list[str] = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    extracted.append(item.strip())
                elif isinstance(item, dict):
                    for candidate_key in ("text", "token", "answer"):
                        candidate = item.get(candidate_key)
                        if isinstance(candidate, str) and candidate.strip():
                            extracted.append(candidate.strip())
                            break
            if extracted:
                return extracted
    return []


def save_image(image: Any, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(output_path)
        return str(output_path)
    if isinstance(image, str):
        source_path = Path(image)
        if source_path.exists():
            output_path.write_bytes(source_path.read_bytes())
            return str(output_path)
    if isinstance(image, (bytes, bytearray)):
        output_path.write_bytes(bytes(image))
        return str(output_path)
    if isinstance(image, dict) and "bytes" in image:
        output_path.write_bytes(image["bytes"])
        return str(output_path)
    raise ValueError(f"Unsupported image payload: {type(image)!r}")


def normalize_row(dataset_name: str, record: dict[str, Any], index: int, image_dir: Path, image_prefix: str) -> dict[str, Any] | None:
    question = coalesce(record, "question", "query", "question_text")
    if not isinstance(question, str) or not question.strip():
        return None

    answer = extract_answer(record)
    if not answer:
        return None

    image = coalesce(record, "image", "img", "page_image", "document_image")
    if image is None:
        return None

    image_path = image_dir / f"{image_prefix}_{index:06d}.png"
    saved_image_path = save_image(image, image_path)

    row = {
        "image_path": saved_image_path.replace("\\", "/"),
        "question": question.strip(),
        "answer": answer.strip(),
        "evidence": extract_evidence(record),
        "dataset": dataset_name,
        "sample_id": str(coalesce(record, "question_id", "id", "uid") or index),
    }
    return row


def load_hf_subset(dataset_key: str, split: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise SystemExit("请先安装: pip install datasets")

    dataset_name = DATASET_MAP[dataset_key]
    dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(dataset):
        if limit > 0 and i >= limit:
            break
        rows.append(row)
    print(f"  加载了 {len(rows)} 条 from {dataset_name}")
    return rows


def build_from_hf(dataset_key: str, split: str, limit: int, output_dir: Path) -> Path:
    rows = load_hf_subset(dataset_key, split, limit)
    image_dir = output_dir / "images"
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(rows, start=1):
        row = normalize_row(dataset_key, record, index, image_dir, dataset_key)
        if row is not None:
            manifest.append(row)

    output_path = output_dir / f"{dataset_key}_{split.replace('/', '_')}.jsonl"
    write_jsonl(output_path, manifest)
    return output_path


def build_from_manifest(manifest_path: Path, output_dir: Path, image_root: Path | None, limit: int) -> Path:
    if manifest_path.suffix.lower() == ".jsonl":
        records = load_jsonl(manifest_path)
    else:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("data", records.get("items", []))
    if not isinstance(records, list):
        raise SystemExit("Manifest must be a JSON array, JSONL, or a dict with 'data'/'items'.")

    if limit > 0:
        records = records[:limit]

    output_rows: list[dict[str, Any]] = []
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        question = coalesce(record, "question", "query", "question_text")
        answer = extract_answer(record)
        image_value = coalesce(record, "image_path", "image", "img", "page_image", "imgname")
        if not isinstance(question, str) or not question.strip() or not answer or image_value is None:
            continue

        if isinstance(image_value, str):
            source_path = Path(image_value)
            if image_root is not None and not source_path.is_absolute():
                source_path = image_root / source_path
            if not source_path.exists():
                continue
            target_path = image_dir / f"chartqa_{index:06d}{source_path.suffix or '.png'}"
            target_path.write_bytes(source_path.read_bytes())
            saved_image_path = str(target_path)
        elif isinstance(image_value, Image.Image):
            target_path = image_dir / f"chartqa_{index:06d}.png"
            saved_image_path = save_image(image_value, target_path)
        else:
            continue

        output_rows.append(
            {
                "image_path": saved_image_path.replace("\\", "/"),
                "question": question.strip(),
                "answer": answer.strip(),
                "evidence": extract_evidence(record),
                "dataset": "chartqa",
                "sample_id": str(coalesce(record, "question_id", "id", "uid") or index),
            }
        )

    output_path = output_dir / "chartqa_custom.jsonl"
    write_jsonl(output_path, output_rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare course-ready subsets from public VQA datasets.")
    parser.add_argument("--dataset", choices=["vqa_v2", "textvqa", "docvqa", "chartqa"], required=True)
    parser.add_argument("--split", default="train", help="Hugging Face split name for VQA-v2/TextVQA/DocVQA")
    parser.add_argument("--limit", type=int, default=200, help="Maximum samples to export")
    parser.add_argument("--output-dir", default="data/public", help="Directory to store exported samples")
    parser.add_argument("--manifest", help="Local ChartQA manifest in JSON/JSONL format")
    parser.add_argument("--image-root", help="Optional image root for ChartQA manifests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "chartqa":
        if not args.manifest:
            raise SystemExit("ChartQA needs --manifest pointing to a local JSON/JSONL annotation file.")
        manifest_path = Path(args.manifest)
        image_root = Path(args.image_root) if args.image_root else None
        output_path = build_from_manifest(manifest_path, output_dir, image_root, args.limit)
    else:
        output_path = build_from_hf(args.dataset, args.split, args.limit, output_dir)

    print(f"Exported subset to {output_path}")


if __name__ == "__main__":
    main()