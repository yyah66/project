"""
多维度消融评测指标分析
支持：VQA Accuracy (EM + Relaxed)、ANLS、证据一致性、幻觉率、人工评分模板
用法: python3 scripts/evaluate_metrics.py [--results-dir outputs/ablation]
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

# ---------------------------------------------------------------------------
# ANLS (Average Normalized Levenshtein Similarity)
# Reference: ST-VQA task, ICDAR 2019
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def anls_score(pred: str, gt: str, threshold: float = 0.5) -> float:
    """单样本 ANLS：0-1 之间，低于阈值归零。"""
    p = _normalize_anls(pred)
    g = _normalize_anls(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    dist = _levenshtein(p, g)
    score = 1.0 - dist / max(len(p), len(g))
    return score if score >= threshold else 0.0


def _normalize_anls(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s一-鿿]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 证据一致性
# ---------------------------------------------------------------------------


def evidence_ocr_overlap(evidence: list[str], ocr_texts: list[str]) -> dict:
    """计算证据列表与 OCR 文本的重叠度。"""
    if not evidence:
        return {"ratio": None, "matched": 0, "total": 0}

    matched = 0
    for ev in evidence:
        ev_norm = _normalize_anls(ev)
        if not ev_norm:
            continue
        for ocr in ocr_texts:
            ocr_norm = _normalize_anls(ocr)
            # 检查证据片段是否在 OCR 中，或用词重叠 > 50%
            if ev_norm in ocr_norm or ocr_norm in ev_norm:
                matched += 1
                break
            # fallback: 词级别 Jaccard
            ev_words = set(ev_norm.split())
            ocr_words = set(ocr_norm.split())
            if ev_words and ocr_words:
                jaccard = len(ev_words & ocr_words) / len(ev_words)
                if jaccard > 0.5:
                    matched += 1
                    break
    return {"ratio": matched / len(evidence) if evidence else None,
            "matched": matched, "total": len(evidence)}


# ---------------------------------------------------------------------------
# 幻觉检测
# ---------------------------------------------------------------------------


def _extract_numbers(text: str) -> list[str]:
    """提取文本中的数值信息（数字 + 中文数字 + 单位）。"""
    nums = []
    # 阿拉伯数字（带单位）
    nums.extend(re.findall(r"\d+(?:\.\d+)?\s*(?:%|元|万|亿|个|人|台|辆|克|千克|吨|米|公里|小时|分钟|秒|岁|年|月|日|倍|[a-zA-Z]+)?", text))
    # 中文数字
    cn_nums = re.findall(r"[一二三四五六七八九十百千万亿]+", text)
    nums.extend(cn_nums)
    return nums


def check_hallucination(pred: str, ocr_texts: list[str], question: str = "") -> dict:
    """检测预测答案中的潜在幻觉：数值是否在 OCR 中能找到。"""
    nums = _extract_numbers(pred)
    if not nums:
        return {"hallucinated": False, "suspect_items": [], "reason": "无数值信息"}

    # 合并 OCR 文本为一个大池子
    ocr_pool = " ".join(ocr_texts) if ocr_texts else ""
    ocr_nums = set(_extract_numbers(ocr_pool))

    suspect = []
    for n in nums:
        n_clean = n.strip()
        if n_clean not in ocr_nums and n_clean not in ocr_pool:
            # 检查是否在问题中（如果问题提供了数字，不算幻觉）
            if n_clean in question:
                continue
            suspect.append(n_clean)

    return {
        "hallucinated": len(suspect) > 0,
        "suspect_items": suspect,
        "reason": f"答案中 {len(suspect)} 个数值未在 OCR 中找到" if suspect else "所有数值均在 OCR 中有据可查",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_ocr_index(test_jsonl: str) -> dict[str, list[str]]:
    """从增强测试集加载 OCR 文本索引（按 image_path）。"""
    index: dict[str, list[str]] = {}
    path = Path(test_jsonl)
    if not path.exists():
        print(f"警告: 测试集不存在 {test_jsonl}")
        return index
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            ocr_lines = s.get("ocr_lines", [])
            texts = [item["text"] for item in ocr_lines if isinstance(item, dict) and item.get("text")]
            index[s["image_path"]] = texts
    return index


def load_group_results(results_dir: Path) -> list[dict]:
    rows = []
    for filename, group, ocr, evidence, model in GROUP_FILES:
        path = results_dir / filename
        if not path.exists():
            rows.append({
                "group": group, "ocr": ocr, "evidence": evidence, "model": model,
                "details": [], "missing": True,
            })
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "group": group, "ocr": ocr, "evidence": evidence, "model": model,
            "details": data.get("details", []),
            "missing": False,
            "config": data.get("config", {}),
            "em_rate": data.get("exact_match_rate", 0),
            "em_count": data.get("exact_match", 0),
        })
    return rows


def compute_group_metrics(group_data: dict, ocr_index: dict[str, list[str]]) -> dict:
    """对单组计算所有指标。"""
    details = group_data["details"]
    n = len(details)
    if n == 0:
        return {"n": 0}

    em_correct = sum(1 for d in details if d.get("correct", False))
    relaxed_correct = 0
    anls_scores: list[float] = []
    ev_ratios: list[float] = []
    ev_matched_total = 0
    ev_total_support = 0
    halluc_count = 0
    halluc_details: list[dict] = []

    for d in details:
        pred = str(d.get("prediction", ""))
        gt = str(d.get("ground_truth", ""))
        image_path = d.get("image_path", "")
        question = d.get("question", "")

        # Relaxed accuracy: gt in pred or pred in gt
        if not d.get("correct", False):
            p_norm = _normalize_anls(pred)
            g_norm = _normalize_anls(gt)
            if g_norm in p_norm or p_norm in g_norm:
                relaxed_correct += 1

        # ANLS
        anls_scores.append(anls_score(pred, gt))

        # 证据一致性（仅 evidence=ON 的组）
        evidence = d.get("evidence", [])
        ocr_texts = ocr_index.get(image_path, [])
        if evidence and ocr_texts:
            overlap = evidence_ocr_overlap(evidence, ocr_texts)
            if overlap["ratio"] is not None:
                ev_ratios.append(overlap["ratio"])
                ev_matched_total += overlap["matched"]
                ev_total_support += overlap["total"]

        # 幻觉检测（仅 OCR=ON 的组有 OCR 文本）
        if ocr_texts:
            h = check_hallucination(pred, ocr_texts, question)
            if h["hallucinated"]:
                halluc_count += 1
                halluc_details.append({
                    "image_path": image_path,
                    "question": question,
                    "prediction": pred,
                    "ground_truth": gt,
                    "suspect_items": h["suspect_items"],
                })

    em_rate = em_correct / n if n else 0
    relaxed_rate = (em_correct + relaxed_correct) / n if n else 0
    avg_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    avg_ev_ratio = sum(ev_ratios) / len(ev_ratios) if ev_ratios else None
    halluc_rate = halluc_count / n if n else 0

    return {
        "n": n,
        "em_rate": em_rate,
        "em_count": em_correct,
        "relaxed_rate": relaxed_rate,
        "avg_anls": avg_anls,
        "avg_evidence_overlap": avg_ev_ratio,
        "evidence_samples": len(ev_ratios),
        "hallucination_rate": halluc_rate,
        "hallucination_count": halluc_count,
        "hallucination_details": halluc_details,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="多维度消融评测分析")
    parser.add_argument("--results-dir", default="outputs/ablation")
    parser.add_argument("--test-jsonl",
                        default="data/training/merged/merged_test_enhanced.jsonl")
    parser.add_argument("--output", default=None, help="输出 Markdown 报告路径")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"错误: 目录不存在 {results_dir}")
        sys.exit(1)

    print("加载 OCR 索引...")
    ocr_index = load_ocr_index(args.test_jsonl)
    has_ocr_index = len(ocr_index) > 0
    if not has_ocr_index:
        print("⚠️  未找到增强测试集，证据一致性和幻觉检测将跳过")

    groups = load_group_results(results_dir)
    valid = [g for g in groups if not g["missing"]]

    if not valid:
        print("错误: 没有任何有效结果文件")
        sys.exit(1)

    # 计算所有组的指标
    metrics = {}
    for g in valid:
        metrics[g["group"]] = compute_group_metrics(g, ocr_index)

    # ---- 汇总输出 ----
    lines: list[str] = []
    lines.append("## 多维度消融评测报告\n")

    # 表1: 各组全指标
    lines.append("### 各组综合指标\n")
    headers = ["组号", "OCR", "证据", "模型", "EM", "Relaxed", "ANLS"]
    if has_ocr_index:
        headers.extend(["证据一致性", "幻觉率"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["------"] * len(headers)) + "|")

    for g in groups:
        if g["missing"]:
            cells = [g["group"], g["ocr"], g["evidence"], g["model"], "*缺失*"] + ([""] * (len(headers) - 5))
        else:
            m = metrics[g["group"]]
            cells = [
                g["group"], g["ocr"], g["evidence"], g["model"],
                fmt_pct(m["em_rate"]),
                fmt_pct(m["relaxed_rate"]),
                f"{m['avg_anls']:.4f}",
            ]
            if has_ocr_index:
                cells.append(fmt_pct(m["avg_evidence_overlap"]) if m["evidence_samples"] > 0 else "N/A")
                cells.append(fmt_pct(m["hallucination_rate"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("*Relaxed = 宽松匹配（答案或预测互相包含）；ANLS = 归一化编辑距离相似度*\n")

    # 表2: 主效应（多指标）
    lines.append("### 主效应分析（LoRA vs Base）\n")
    lines.append("| 指标 | LoRA (平均) | Base (平均) | 提升 |")
    lines.append("|------|------------|------------|------|")
    for metric_key, metric_name in [
        ("em_rate", "EM"), ("relaxed_rate", "Relaxed Acc"), ("avg_anls", "ANLS"),
    ]:
        lora_vals = [metrics[g["group"]][metric_key] for g in valid if g["model"] == "LoRA" and not g["missing"]]
        base_vals = [metrics[g["group"]][metric_key] for g in valid if g["model"] == "Base" and not g["missing"]]
        if lora_vals and base_vals:
            lora_avg = sum(lora_vals) / len(lora_vals)
            base_avg = sum(base_vals) / len(base_vals)
            diff = lora_avg - base_avg
            if metric_key == "avg_anls":
                lines.append(f"| {metric_name} | {lora_avg:.4f} | {base_avg:.4f} | {diff:+.4f} |")
            else:
                lines.append(f"| {metric_name} | {fmt_pct(lora_avg)} | {fmt_pct(base_avg)} | {fmt_pct(diff)} |")
    if has_ocr_index:
        for metric_key, metric_name in [
            ("hallucination_rate", "幻觉率 (↓)"),
        ]:
            lora_vals = [metrics[g["group"]][metric_key] for g in valid if g["model"] == "LoRA"]
            base_vals = [metrics[g["group"]][metric_key] for g in valid if g["model"] == "Base"]
            if lora_vals and base_vals:
                lora_avg = sum(lora_vals) / len(lora_vals)
                base_avg = sum(base_vals) / len(base_vals)
                diff = lora_avg - base_avg
                lines.append(f"| {metric_name} | {fmt_pct(lora_avg)} | {fmt_pct(base_avg)} | {fmt_pct(diff)} |")
    lines.append("")

    # 表3: 证据一致性详情（仅 evidence=ON 的组）
    if has_ocr_index:
        lines.append("### 证据一致性分析\n")
        lines.append("证据一致性 = OCR 证实的证据条数 / 总证据条数\n")
        lines.append("| 组号 | 配置 | 证据覆盖率 | 有证据样本数 |")
        lines.append("|------|------|-----------|------------|")
        for g in valid:
            m = metrics[g["group"]]
            if m["evidence_samples"] > 0:
                lines.append(f"| {g['group']} | OCR={g['ocr']} Evi={g['evidence']} {g['model']} | {fmt_pct(m['avg_evidence_overlap'])} | {m['evidence_samples']} |")
        lines.append("")

    # 表4: 幻觉 Top 案例
    if has_ocr_index:
        lines.append("### 幻觉典型案例\n")
        lines.append("| 组号 | 题目 | 预测 | 标准答案 | 可疑内容 |")
        lines.append("|------|------|------|----------|---------|")
        shown = 0
        for g in valid:
            m = metrics[g["group"]]
            for h in m.get("hallucination_details", [])[:3]:
                lines.append(f"| {g['group']} | {h['question'][:40]} | {h['prediction'][:40]} | {h['ground_truth'][:40]} | {', '.join(h['suspect_items'][:3])} |")
                shown += 1
                if shown >= 15:
                    break
            if shown >= 15:
                break
        lines.append("")

    # 表5: 最佳/最差组合
    lines.append("### 极值\n")
    best = max(valid, key=lambda g: metrics[g["group"]]["em_rate"])
    worst = min(valid, key=lambda g: metrics[g["group"]]["em_rate"])
    lines.append(f"- **最佳**: 组{best['group']} (OCR={best['ocr']}, Evidence={best['evidence']}, Model={best['model']}) EM={fmt_pct(metrics[best['group']]['em_rate'])}")
    lines.append(f"- **最差**: 组{worst['group']} (OCR={worst['ocr']}, Evidence={worst['evidence']}, Model={worst['model']}) EM={fmt_pct(metrics[worst['group']]['em_rate'])}")

    # 表6: 人工评分模板
    lines.append("\n### 人工评分模板\n")
    lines.append("建议从每组随机抽 20 条，按以下维度 1-5 打分：\n")
    lines.append("| 维度 | 评分标准 |")
    lines.append("|------|---------|")
    lines.append("| 答案正确性 | 1=完全错误 5=完全正确 |")
    lines.append("| 证据质量 | 1=无关证据 5=精准引用 |")
    lines.append("| 表达自然度 | 1=生硬/格式错误 5=自然流畅 |")
    lines.append("| 有无幻觉 | 1=严重编造 5=无幻觉 |")
    lines.append("")
    lines.append("抽样命令示例：")
    for g in valid:
        fname = f"group{g['group']}_*"
        lines.append(f"- 组{g['group']}: `python3 -c \"import json,random; d=json.load(open('{args.results_dir}/{fname}'.replace('*','ocr_on_evi_on_lora' if g['group']=='1' else ''))); random.seed(42); print(json.dumps(random.sample(d['details'],20),ensure_ascii=False,indent=2))\"`")

    report = "\n".join(lines)
    print(report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n报告已保存: {args.output}")

    # 汇总 JSON
    summary_path = results_dir / "metrics_summary.json"
    summary = {
        group_id: {k: v for k, v in m.items() if k != "hallucination_details"}
        for group_id, m in metrics.items()
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标 JSON 已保存: {summary_path}")


if __name__ == "__main__":
    main()
