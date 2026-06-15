"""
消融实验评测：OCR(ON/OFF) x Evidence(ON/OFF) x Model(LoRA/Base) = 8 组
直接读取增强 JSONL 的预计算 OCR，不再实时加载 OCR 服务。

用法:
  # 8 组消融
  python3 scripts/evaluate.py --ablation

  # 单组评测
  python3 scripts/evaluate.py --adapter outputs/lora/merged_ocr/final_adapter --test-jsonl data/training/merged/merged_test_enhanced.jsonl
"""
from __future__ import annotations

import argparse, json, os, re, sys, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ocr import OCRResult, OCRLine, ImageAnalysis
from src.prompt import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "/root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_ADAPTER = "outputs/lora/merged_ocr/final_adapter"
DEFAULT_TEST = "data/training/merged/merged_test_enhanced.jsonl"
DEFAULT_OUTDIR = "outputs/ablation"

ABLATION_GROUPS = [
    ("group1_ocr_on_evi_on_lora.json",   True,  True,  True),
    ("group2_ocr_on_evi_on_base.json",   True,  True,  False),
    ("group3_ocr_on_evi_off_lora.json",  True,  False, True),
    ("group4_ocr_on_evi_off_base.json",  True,  False, False),
    ("group5_ocr_off_evi_on_lora.json",  False, True,  True),
    ("group6_ocr_off_evi_on_base.json",  False, True,  False),
    ("group7_ocr_off_evi_off_lora.json", False, False, True),
    ("group8_ocr_off_evi_off_base.json", False, False, False),
]

# 空 OCR 结果（OCR=OFF 时复用）
EMPTY_OCR = OCRResult(
    backend="none", lines=[],
    analysis=ImageAnalysis(
        scene_type="未知", text_density=0.0,
        likely_table=False, likely_chart=False, likely_document=False,
        language_hint="", notes=["OCR 已关闭"],
    ),
)


def _reconstruct_ocr_result(sample: dict) -> OCRResult:
    """从预计算 OCR 字段重建 OCRResult（与 train_qwen2vl_lora.py 一致）。"""
    lines_data = sample.get("ocr_lines", [])
    analysis_data = sample.get("ocr_analysis")

    lines = [
        OCRLine(
            text=item["text"],
            bbox=tuple(item["bbox"]) if item.get("bbox") else None,
            confidence=item.get("confidence"),
        )
        for item in lines_data
    ]

    analysis = None
    if analysis_data:
        analysis = ImageAnalysis(
            scene_type=analysis_data.get("scene_type", "未知"),
            text_density=analysis_data.get("text_density", 0.0),
            likely_table=analysis_data.get("likely_table", False),
            likely_chart=analysis_data.get("likely_chart", False),
            likely_document=analysis_data.get("likely_document", False),
            language_hint=analysis_data.get("language_hint", ""),
            notes=analysis_data.get("notes", []),
        )

    return OCRResult(backend="precomputed", lines=lines, analysis=analysis)


def load_samples(test_jsonl: str, image_root: str) -> list[dict]:
    """从增强 JSONL 加载样本，保留预计算 OCR 字段。"""
    root = Path(image_root)
    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            p = Path(s["image_path"])
            if not p.is_absolute():
                p = root / p
            if not p.exists():
                continue
            try:
                img = Image.open(p)
                img.load()
            except Exception:
                continue
            samples.append({
                "path": str(p),
                "question": s["question"],
                "answer": s.get("answer", ""),
                "ocr_result": _reconstruct_ocr_result(s),
            })
    return samples


def run_eval(model, processor, samples, ocr_on: bool, evidence_enabled: bool,
             lora_enabled: bool) -> dict:
    correct = 0
    details = []
    for i, s in enumerate(samples):
        image = Image.open(s["path"]).convert("RGB")
        ocr_result = s["ocr_result"] if ocr_on else EMPTY_OCR

        system_prompt = build_system_prompt(evidence_enabled)
        user_prompt = build_user_prompt(s["question"], ocr_result, [], 24, evidence_enabled)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(
            model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = processor.decode(generated[0], skip_special_tokens=False)
        match = re.search(r"<\|im_start\|>assistant\s*\n(.*?)(?:<\|im_end\|>|$)",
                          response, re.DOTALL)
        pred = match.group(1).strip() if match else response.strip()

        # 始终尝试解析 JSON（训练和推理时都要求 JSON 格式输出）
        answer_pred = pred
        evidence = []
        confidence = ""
        uncertainty = ""
        try:
            parsed = json.loads(pred)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer_pred = str(parsed["answer"])
                evidence = parsed.get("evidence", [])
                confidence = parsed.get("confidence", "")
                uncertainty = parsed.get("uncertainty", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        is_correct = answer_pred.lower() == s["answer"].strip().lower()
        if is_correct:
            correct += 1
        detail = {
            "image_path": s["path"],
            "question": s["question"],
            "ground_truth": s["answer"],
            "prediction": answer_pred,
            "correct": is_correct,
        }
        if evidence:
            detail["evidence"] = evidence
        if confidence:
            detail["confidence"] = confidence
        if uncertainty:
            detail["uncertainty"] = uncertainty
        details.append(detail)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)}  EM: {correct / (i + 1) * 100:.1f}%")

    em_rate = correct / len(samples) if samples else 0
    print(f"  -> EM: {correct}/{len(samples)} = {em_rate * 100:.2f}%")
    return {
        "config": {
            "ocr_enabled": ocr_on,
            "evidence_prompt_enabled": evidence_enabled,
            "model_type": "lora" if lora_enabled else "base",
            "total_samples": len(samples),
        },
        "exact_match": correct,
        "exact_match_rate": em_rate,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="消融实验评测（使用预计算 OCR）")
    parser.add_argument("--ablation", action="store_true", help="运行全部 8 组消融")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--base-only", action="store_true", help="仅基座模型")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-evidence-prompt", action="store_true")
    parser.add_argument("--save-results", default=None)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    if args.ablation:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("  消融实验：8 组（使用预计算 OCR）")
        print(f"  模型: {args.base_model}")
        print(f"  Adapter: {args.adapter}")
        print(f"  测试集: {args.test_jsonl}")
        print(f"  输出目录: {outdir}")
        print("=" * 60)

        print("\n加载基座模型...")
        processor = AutoProcessor.from_pretrained(args.base_model,
                                                  trust_remote_code=True)
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 512 * 28 * 28
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        print("模型就绪（基座 + LoRA）")

        samples = load_samples(args.test_jsonl, args.image_root)
        # 检查 OCR 是否已预计算
        has_precomputed = any(
            s["ocr_result"].lines for s in samples[:10]
        )
        if has_precomputed:
            print(f"测试样本: {len(samples)} (OCR 已预计算)")
        else:
            print(f"测试样本: {len(samples)} (⚠️  无预计算 OCR，建议使用 enhanced JSONL)")

        for filename, ocr_on, evi_on, use_lora in ABLATION_GROUPS:
            idx = filename[5]
            ocr_label = "ON" if ocr_on else "OFF"
            evi_label = "ON" if evi_on else "OFF"
            model_label = "LoRA" if use_lora else "Base"

            result_path = outdir / filename
            if result_path.exists():
                try:
                    prev = json.loads(result_path.read_text(encoding="utf-8"))
                    em = prev.get("exact_match_rate", 0)
                    total = prev.get("config", {}).get("total_samples", "?")
                    print(f"\n[{idx}/8] OCR={ocr_label}  Evidence={evi_label}  "
                          f"Model={model_label} — 已完成，跳过 (EM={em*100:.1f}%, {total}样本)")
                except Exception:
                    print(f"\n[{idx}/8] OCR={ocr_label}  Evidence={evi_label}  "
                          f"Model={model_label} — 结果文件损坏，重新运行")
                else:
                    continue

            print(f"\n[{idx}/8] OCR={ocr_label}  Evidence={evi_label}  "
                  f"Model={model_label}")

            if use_lora:
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()

            result = run_eval(model, processor, samples, ocr_on, evi_on, use_lora)
            (outdir / filename).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  已保存: {outdir / filename}")

        print("\n" + "=" * 60)
        print("  消融实验全部完成")
        print(f"  结果目录: {outdir}")
        print("=" * 60)
        print("\n运行汇总: python3 scripts/summarize_ablation.py")
        return

    # ---- 单组评测 ----
    print(f"加载模型: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    lora_enabled = False
    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
        lora_enabled = True
    model.eval()
    print(f"模型就绪 ({'LoRA' if lora_enabled else '基座'})")

    ocr_on = not args.no_ocr
    evidence_enabled = not args.no_evidence_prompt
    print(f"OCR: {'关闭' if args.no_ocr else '开启'} | "
          f"证据提示: {'关闭' if args.no_evidence_prompt else '开启'}")

    samples = load_samples(args.test_jsonl, args.image_root)
    print(f"测试样本: {len(samples)}")

    result = run_eval(model, processor, samples, ocr_on, evidence_enabled,
                      lora_enabled)

    print(f"\nExact Match: {result['exact_match']}/{len(samples)} = "
          f"{result['exact_match_rate'] * 100:.2f}%")

    if args.save_results:
        Path(args.save_results).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {args.save_results}")


if __name__ == "__main__":
    main()
