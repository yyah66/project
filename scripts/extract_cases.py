"""
从评测结果 JSON 中提取 5 个典型成功案例和 5 个典型失败案例
用法: python3 scripts/extract_cases.py --results outputs/lora/merged/eval_results.json
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from collections import Counter


def diversity_score(selected: list[dict], candidate: dict) -> float:
    """越少出现的关键词得分越高，保证案例多样性"""
    all_text = " ".join(s["question"] for s in selected + [candidate])
    words = all_text.lower().split()
    freq = Counter(words)
    cand_words = set(candidate["question"].lower().split())
    return sum(1.0 / max(freq.get(w, 1), 1) for w in cand_words)


def pick_diverse(candidates: list[dict], n: int) -> list[dict]:
    picked: list[dict] = []
    pool = list(candidates)
    random.shuffle(pool)
    for _ in range(n):
        best = max(pool, key=lambda c: diversity_score(picked, c))
        picked.append(best)
        pool.remove(best)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="evaluate.py 输出的 JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    correct = [d for d in data["details"] if d["correct"]]
    wrong = [d for d in data["details"] if not d["correct"]]

    random.seed(args.seed)
    top_correct = pick_diverse(correct, 5)
    top_wrong = pick_diverse(wrong, 5)

    print("=" * 60)
    print("5 个典型成功案例")
    print("=" * 60)
    for i, c in enumerate(top_correct, 1):
        print(f"\n案例 {i}")
        print(f"  图片: {c['image_path']}")
        print(f"  问题: {c['question']}")
        print(f"  标注答案: {c['ground_truth']}")
        print(f"  模型预测: {c['prediction']}")

    print("\n" + "=" * 60)
    print("5 个典型失败案例")
    print("=" * 60)
    for i, c in enumerate(top_wrong, 1):
        print(f"\n案例 {i}")
        print(f"  图片: {c['image_path']}")
        print(f"  问题: {c['question']}")
        print(f"  标注答案: {c['ground_truth']}")
        print(f"  模型预测: {c['prediction']}")

    print(f"\n总计: {len(correct)} 正确, {len(wrong)} 错误, EM={data['exact_match_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
