from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.assistant import AssistantEngine
from src.config import AppConfig


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "", value)
    value = value.replace("：", ":")
    value = value.replace("，", ",")
    value = value.replace("。", "")
    return value


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_text(prediction) == normalize_text(answer)


def overlap_score(prediction: str, answer: str) -> float:
    pred_tokens = [token for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", normalize_text(prediction)) if token]
    ans_tokens = [token for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", normalize_text(answer)) if token]
    if not pred_tokens or not ans_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    ans_set = set(ans_tokens)
    return len(pred_set & ans_set) / len(ans_set)


def build_failure_note(record: dict, prediction: str) -> str:
    answer = str(record.get("answer", "")).strip()
    evidence = record.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if exact_match(prediction, answer):
        return ""
    if answer and any(ch.isdigit() for ch in answer) and not any(ch.isdigit() for ch in prediction):
        return "数字读取偏差"
    if answer and evidence:
        first_evidence = str(evidence[0])
        if first_evidence and first_evidence not in prediction:
            return "证据覆盖不足"
    return "答案表达不一致"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Chinese VQA assistant on a JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", required=True, help="Path to save JSONL predictions")
    parser.add_argument("--limit", type=int, default=0, help="Optional item limit")
    parser.add_argument("--adapter-path", default="", help="LoRA adapter path for fine-tuned model evaluation")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]

    engine = AssistantEngine()

    # 如果指定了 LoRA adapter，配置本地模型 + adapter 路径
    if args.adapter_path:
        config = AppConfig(
            provider="local",
            local_model_path=args.base_model,
            local_device="auto",
            lora_path=args.adapter_path,
        )
        engine = AssistantEngine(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    exact_match_count = 0
    overlap_total = 0.0
    failure_notes: dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as writer:
        for index, record in enumerate(records, start=1):
            image_path = Path(record["image_path"])
            question = record["question"]
            answer = str(record.get("answer", "")).strip()
            image = Image.open(image_path).convert("RGB")
            with image_path.open("rb") as image_file:
                image_bytes = image_file.read()
            ocr_result = engine.ocr_service.extract(image)
            answer_result = engine.answer_question(
                image=image,
                image_bytes=image_bytes,
                question=question,
                history=[],
                ocr_result=ocr_result,
            )
            output_record = {
                **record,
                "prediction": answer_result.answer,
                "evidence": answer_result.evidence,
                "confidence": answer_result.confidence,
                "uncertainty": answer_result.uncertainty,
                "provider": answer_result.provider,
                "model": answer_result.model,
                "ocr_backend": ocr_result.backend,
                "exact_match": exact_match(answer_result.answer, answer) if answer else None,
                "overlap_score": overlap_score(answer_result.answer, answer) if answer else None,
            }
            writer.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(records)}] done: {image_path.name}")

            if answer:
                total += 1
                if exact_match(answer_result.answer, answer):
                    exact_match_count += 1
                overlap_total += overlap_score(answer_result.answer, answer)
                note = build_failure_note(record, answer_result.answer)
                if note:
                    failure_notes[note] = failure_notes.get(note, 0) + 1

    if total:
        summary = {
            "samples": total,
            "exact_match": round(exact_match_count / total, 4),
            "average_overlap": round(overlap_total / total, 4),
            "failure_notes": dict(sorted(failure_notes.items(), key=lambda item: (-item[1], item[0]))),
        }
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
