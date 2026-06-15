"""
消融实验独立控制脚本 (Ablation Experiment Runner)

通过命令行参数控制：
  - OCR 开关（有无 OCR 增强）
  - 证据提示开关（是否强制附带证据）
  - 模型选择（主模型/备选模型）
  - 数据集路径

输出 JSONL 预测文件 + 汇总统计。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.assistant import AssistantEngine
from src.config import AppConfig
from src.ocr import NoOpOCRService


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_ablation(
    input_path: Path,
    output_dir: Path,
    use_ocr: bool,
    use_evidence_prompt: bool,
    model_override: str | None = None,
    limit: int = 0,
) -> None:
    """执行一组消融条件下的评测。"""
    # 构造配置：优先使用环境变量 DASHSCOPE_API_KEY / VLM_MODEL，其次用默认值
    config_kwargs: dict = {}
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        config_kwargs["api_key"] = api_key
    base_url = os.environ.get("VLM_BASE_URL", "")
    if base_url:
        config_kwargs["base_url"] = base_url
    provider = os.environ.get("VLM_PROVIDER", "")
    if provider:
        config_kwargs["provider"] = provider
    if model_override:
        config_kwargs["model"] = model_override
    elif os.environ.get("VLM_MODEL"):
        config_kwargs["model"] = os.environ["VLM_MODEL"]
    config = AppConfig(**config_kwargs)

    # 创建引擎
    engine = AssistantEngine(config)

    # 如果不启用 OCR，替换 OCR 服务为 NoOp
    if not use_ocr:
        engine.ocr_service = NoOpOCRService()

    # 如果不启用证据提示，则修改 prompt（通过替换提示词构造逻辑）
    if not use_evidence_prompt:
        engine.evidence_prompt_enabled = False
    else:
        engine.evidence_prompt_enabled = True

    records = load_jsonl(input_path)
    if limit > 0:
        records = records[:limit]

    suffix = _build_suffix(use_ocr, use_evidence_prompt, model_override)
    predictions_path = output_dir / f"predictions{suffix}.jsonl"
    summary_path = output_dir / f"summary{suffix}.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    total_em = 0
    total_overlap = 0.0
    total_samples_with_answer = 0
    error_types: dict[str, int] = {}
    records_output: list[dict] = []

    for idx, record in enumerate(records, start=1):
        image_path = Path(record.get("image_path", ""))
        question = record.get("question", "")
        answer = str(record.get("answer", "")).strip()

        if not image_path.exists():
            print(f"[{idx}/{len(records)}] 跳过：图片不存在 {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_bytes = image_path.read_bytes()
        except Exception as e:
            print(f"[{idx}/{len(records)}] 跳过：图片读取失败 {image_path} — {e}")
            continue

        ocr_result = engine.ocr_service.extract(image)
        answer_result = engine.answer_question(
            image=image,
            image_bytes=image_bytes,
            question=question,
            history=[],
            ocr_result=ocr_result,
        )

        pred_text = answer_result.answer.strip()
        em = 1 if pred_text and answer and _normalize(pred_text) == _normalize(answer) else 0
        overlap = _overlap_score(pred_text, answer) if answer else 0.0

        output_record = {
            **record,
            "prediction": pred_text,
            "evidence": answer_result.evidence,
            "confidence": answer_result.confidence,
            "uncertainty": answer_result.uncertainty,
            "provider": answer_result.provider,
            "model": answer_result.model,
            "ocr_backend": ocr_result.backend,
            "exact_match": em == 1,
            "overlap_score": round(overlap, 4),
        }
        records_output.append(output_record)

        if answer:
            total_samples_with_answer += 1
            total_em += em
            total_overlap += overlap
            if em == 0:
                error_type = _classify_error(pred_text, answer, record.get("evidence", []))
                error_types[error_type] = error_types.get(error_type, 0) + 1

        print(f"[{idx}/{len(records)}] {image_path.name}: EM={'✓' if em else '✗'} pred={pred_text[:60]}")

    # 写入预测结果
    with predictions_path.open("w", encoding="utf-8") as f:
        for item in records_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入汇总统计
    summary = {
        "config": {
            "use_ocr": use_ocr,
            "use_evidence_prompt": use_evidence_prompt,
            "model": config.model,
            "dataset": str(input_path),
        },
        "total_records": len(records_output),
        "samples_with_answer": total_samples_with_answer,
        "exact_match": round(total_em / total_samples_with_answer, 4) if total_samples_with_answer else 0,
        "average_overlap": round(total_overlap / total_samples_with_answer, 4) if total_samples_with_answer else 0,
        "exact_match_count": total_em,
        "error_distribution": dict(sorted(error_types.items(), key=lambda x: -x[1])),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"消融实验完成：{suffix}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"预测结果已保存到 {predictions_path}")
    print(f"汇总统计已保存到 {summary_path}")


def _build_suffix(use_ocr: bool, use_evidence_prompt: bool, model_override: str | None) -> str:
    parts = []
    parts.append("ocr_on" if use_ocr else "ocr_off")
    parts.append("evidence_on" if use_evidence_prompt else "evidence_off")
    model_short = model_override or "default"
    # 简化模型名
    model_short = model_short.replace("qwen", "Qwen").replace("-vl-", "-VL-").replace("-instruct", "")
    parts.append(model_short)
    return "_" + "_".join(parts)


def _normalize(text: str) -> str:
    import re
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("：", ":").replace("，", ",").replace("。", "")
    return text


def _overlap_score(prediction: str, answer: str) -> float:
    import re
    pred_tokens = [t for t in re.split(r"[^0-9A-Za-z一-鿿]+", _normalize(prediction)) if t]
    ans_tokens = [t for t in re.split(r"[^0-9A-Za-z一-鿿]+", _normalize(answer)) if t]
    if not pred_tokens or not ans_tokens:
        return 0.0
    return len(set(pred_tokens) & set(ans_tokens)) / len(set(ans_tokens))


def _classify_error(prediction: str, answer: str, evidence: list[str]) -> str:
    """对错误类型进行分类。"""
    import re
    has_digit_answer = bool(re.search(r"\d", answer))
    has_digit_pred = bool(re.search(r"\d", prediction))

    if has_digit_answer and not has_digit_pred:
        return "数字遗漏"
    if has_digit_answer and has_digit_pred:
        # 尝试提取数字对比
        ans_nums = re.findall(r"\d+(?:\.\d+)?", answer)
        pred_nums = re.findall(r"\d+(?:\.\d+)?", prediction)
        if ans_nums and pred_nums and ans_nums != pred_nums:
            return "数字偏差"
    if not prediction.strip():
        return "无输出"
    if len(prediction) < 3 and len(answer) > 5:
        return "答案过短"
    if evidence and all(ev not in prediction for ev in evidence):
        return "证据缺失"
    return "表达不一致"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="消融实验运行脚本 - 控制 OCR/证据提示/模型等变量进行对比实验"
    )
    parser.add_argument("--input", required=True, help="输入 JSONL 数据集路径")
    parser.add_argument("--output_dir", default="data/results/ablation", help="输出目录")
    parser.add_argument("--ocr", action="store_true", default=False, help="启用 OCR 增强")
    parser.add_argument("--no-ocr", dest="ocr", action="store_false", help="禁用 OCR 增强")
    parser.add_argument("--evidence", action="store_true", default=True, help="启用证据提示约束")
    parser.add_argument("--no-evidence", dest="evidence", action="store_false", help="禁用证据提示约束")
    parser.add_argument("--model", default=None, help="覆盖模型名称 (如 qwen3-vl-8b-instruct)")
    parser.add_argument("--limit", type=int, default=0, help="限制测试样本数")
    parser.add_argument("--all", action="store_true", help="运行全部消融组合 (8组)")

    args = parser.parse_args()

    if args.all:
        conditions = [
            (True, True, None),
            (True, False, None),
            (False, True, None),
            (False, False, None),
        ]
        for use_ocr, use_evidence, model in conditions:
            print(f"\n{'='*60}")
            print(f"运行消融组合: OCR={'ON' if use_ocr else 'OFF'}, "
                  f"Evidence={'ON' if use_evidence else 'OFF'}, Model={'default'}")
            print(f"{'='*60}")
            run_ablation(
                input_path=Path(args.input),
                output_dir=Path(args.output_dir),
                use_ocr=use_ocr,
                use_evidence_prompt=use_evidence,
                model_override=model,
                limit=args.limit,
            )
        # 输出汇总对比
        _print_comparison_table(Path(args.output_dir))
    else:
        run_ablation(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            use_ocr=args.ocr,
            use_evidence_prompt=args.evidence,
            model_override=args.model,
            limit=args.limit,
        )


def _print_comparison_table(output_dir: Path) -> None:
    """打印各消融组合的对比表格。"""
    print("\n" + "=" * 80)
    print("消融实验汇总对比")
    print("=" * 80)
    print(f"{'OCR':<6} {'证据提示':<10} {'模型':<20} {'EM':<10} {'AvgOverlap':<12} {'Samples':<10}")
    print("-" * 80)

    summaries = sorted(output_dir.glob("summary_*.json"))
    for summary_path in summaries:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            cfg = data.get("config", {})
            ocr = "ON" if cfg.get("use_ocr") else "OFF"
            evidence = "ON" if cfg.get("use_evidence_prompt") else "OFF"
            model = cfg.get("model", "unknown")
            em = data.get("exact_match", 0)
            overlap = data.get("average_overlap", 0)
            samples = data.get("samples_with_answer", 0)
            print(f"{ocr:<6} {evidence:<10} {model:<20} {em:<10.4f} {overlap:<12.4f} {samples:<10}")
        except Exception:
            continue
    print("=" * 80)


if __name__ == "__main__":
    main()
