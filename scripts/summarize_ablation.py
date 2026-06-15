"""
汇总消融实验 8 组结果，输出 Markdown 表格和主效应/交互效应分析。
用法: python3 scripts/summarize_ablation.py [--results-dir outputs/ablation]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

GROUP_FILES = [
    ("group1_ocr_on_evi_on_lora.json",   "1",  "ON",  "ON",  "LoRA"),
    ("group2_ocr_on_evi_on_base.json",   "2",  "ON",  "ON",  "Base"),
    ("group3_ocr_on_evi_off_lora.json",  "3",  "ON",  "OFF", "LoRA"),
    ("group4_ocr_on_evi_off_base.json",  "4",  "ON",  "OFF", "Base"),
    ("group5_ocr_off_evi_on_lora.json",  "5",  "OFF", "ON",  "LoRA"),
    ("group6_ocr_off_evi_on_base.json",  "6",  "OFF", "ON",  "Base"),
    ("group7_ocr_off_evi_off_lora.json", "7",  "OFF", "OFF", "LoRA"),
    ("group8_ocr_off_evi_off_base.json", "8",  "OFF", "OFF", "Base"),
]


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for filename, group, ocr, evidence, model in GROUP_FILES:
        path = results_dir / filename
        if not path.exists():
            print(f"警告: {filename} 不存在，跳过")
            rows.append({
                "group": group, "ocr": ocr, "evidence": evidence, "model": model,
                "em": None, "correct": None, "total": None, "missing": True,
            })
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "group": group, "ocr": ocr, "evidence": evidence, "model": model,
            "em": data["exact_match_rate"],
            "correct": data["exact_match"],
            "total": len(data["details"]),
            "missing": False,
        })
    return rows


def format_pct(rate: float | None) -> str:
    if rate is None:
        return "N/A"
    return f"{rate * 100:.2f}%"


def compute_avg(rows: list[dict], **filters) -> float | None:
    vals = [r["em"] for r in rows
            if not r["missing"] and r["em"] is not None
            and all(r.get(k) == v for k, v in filters.items())]
    return sum(vals) / len(vals) if vals else None


def main():
    parser = argparse.ArgumentParser(description="汇总消融实验结果")
    parser.add_argument("--results-dir", default="outputs/ablation", help="结果 JSON 目录")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"错误: 目录不存在 {results_dir}")
        sys.exit(1)

    rows = load_results(results_dir)
    valid = [r for r in rows if not r["missing"]]

    if not valid:
        print("错误: 没有任何有效结果文件")
        sys.exit(1)

    # ---- 单组结果表 ----
    print("## 消融实验结果汇总\n")
    print("### 单组结果\n")
    print("| 组号 | OCR | 证据提示 | 模型 | EM | 正确/总数 |")
    print("|------|-----|----------|------|----|-----------|")
    for r in rows:
        if r["missing"]:
            print(f"| {r['group']} | {r['ocr']} | {r['evidence']} | {r['model']} | *缺失* | - |")
        else:
            print(f"| {r['group']} | {r['ocr']} | {r['evidence']} | {r['model']} | {format_pct(r['em'])} | {r['correct']}/{r['total']} |")
    print()

    # ---- 主效应分析 ----
    print("### 主效应分析\n")

    ocr_on_avg = compute_avg(valid, ocr="ON")
    ocr_off_avg = compute_avg(valid, ocr="OFF")
    evi_on_avg = compute_avg(valid, evidence="ON")
    evi_off_avg = compute_avg(valid, evidence="OFF")
    lora_avg = compute_avg(valid, model="LoRA")
    base_avg = compute_avg(valid, model="Base")

    print("| 因子 | ON/OFF 水平 | 平均 EM | 效应值 |")
    print("|------|------------|---------|--------|")
    if ocr_on_avg is not None and ocr_off_avg is not None:
        effect = ocr_on_avg - ocr_off_avg
        print(f"| OCR | ON | {format_pct(ocr_on_avg)} | {effect:+.4f} ({format_pct(effect)}) |")
        print(f"| OCR | OFF | {format_pct(ocr_off_avg)} | — |")
    print()
    if evi_on_avg is not None and evi_off_avg is not None:
        effect = evi_on_avg - evi_off_avg
        print(f"| 证据提示 | ON | {format_pct(evi_on_avg)} | {effect:+.4f} ({format_pct(effect)}) |")
        print(f"| 证据提示 | OFF | {format_pct(evi_off_avg)} | — |")
    print()
    if lora_avg is not None and base_avg is not None:
        effect = lora_avg - base_avg
        print(f"| 微调 | LoRA | {format_pct(lora_avg)} | {effect:+.4f} ({format_pct(effect)}) |")
        print(f"| 微调 | Base | {format_pct(base_avg)} | — |")
    print()

    # ---- 交互效应 ----
    print("### 双向交互效应\n")

    print("#### OCR × Model\n")
    print("| | OCR ON | OCR OFF | 差值 |")
    print("|------|--------|---------|------|")
    for model in ("LoRA", "Base"):
        on = compute_avg(valid, ocr="ON", model=model)
        off = compute_avg(valid, ocr="OFF", model=model)
        if on is not None and off is not None:
            diff = on - off
            print(f"| {model} | {format_pct(on)} | {format_pct(off)} | {diff:+.4f} ({format_pct(diff)}) |")
    print()

    print("#### Evidence × Model\n")
    print("| | Evidence ON | Evidence OFF | 差值 |")
    print("|------|-------------|--------------|------|")
    for model in ("LoRA", "Base"):
        on = compute_avg(valid, evidence="ON", model=model)
        off = compute_avg(valid, evidence="OFF", model=model)
        if on is not None and off is not None:
            diff = on - off
            print(f"| {model} | {format_pct(on)} | {format_pct(off)} | {diff:+.4f} ({format_pct(diff)}) |")
    print()

    print("#### OCR × Evidence\n")
    print("| | OCR ON | OCR OFF | 差值 |")
    print("|------|--------|---------|------|")
    for evi in ("ON", "OFF"):
        on = compute_avg(valid, ocr="ON", evidence=evi)
        off = compute_avg(valid, ocr="OFF", evidence=evi)
        if on is not None and off is not None:
            diff = on - off
            print(f"| Evidence {evi} | {format_pct(on)} | {format_pct(off)} | {diff:+.4f} ({format_pct(diff)}) |")
    print()

    # ---- 最佳/最差组合 ----
    print("### 极值\n")
    best = max(valid, key=lambda r: r["em"])
    worst = min(valid, key=lambda r: r["em"])
    print(f"- 最佳: 组{best['group']} (OCR={best['ocr']}, Evidence={best['evidence']}, Model={best['model']}) EM={format_pct(best['em'])}")
    print(f"- 最差: 组{worst['group']} (OCR={worst['ocr']}, Evidence={worst['evidence']}, Model={worst['model']}) EM={format_pct(worst['em'])}")

    missing = [r for r in rows if r["missing"]]
    if missing:
        print(f"\n警告: {len(missing)} 组结果缺失: {', '.join(r['group'] for r in missing)}")


if __name__ == "__main__":
    main()
