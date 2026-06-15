#!/bin/bash
# 消融实验一键运行脚本
# OCR(ON/OFF) × Evidence(ON/OFF) × Model(LoRA/Base) = 8 组
# 用法: bash scripts/run_ablation.sh
# 已完成的组会自动跳过（检查输出文件是否存在）

set -e

MODEL="/root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER="outputs/lora/merged_ocr/final_adapter"
TEST="data/training/merged/merged_test_enhanced.jsonl"
OUTDIR="outputs/ablation"

mkdir -p "$OUTDIR"

# 跳过已存在的文件
run_if_missing() {
  local label="$1"
  local outfile="$OUTDIR/$2"
  shift 2
  if [ -f "$outfile" ]; then
    echo ""
    echo "[$label] 已完成，跳过 ($outfile)"
    return
  fi
  echo ""
  echo "[$label]"
  python3 scripts/evaluate.py "$@" --save-results "$outfile"
}

echo "=========================================="
echo "  消融实验：8 组"
echo "  测试集: $TEST"
echo "  输出目录: $OUTDIR"
echo "=========================================="

# 组1: OCR=ON  Evidence=ON  LoRA
run_if_missing "1/8 OCR=ON Evidence=ON LoRA" "group1_ocr_on_evi_on_lora.json" \
  --adapter "$ADAPTER" --test-jsonl "$TEST" --base-model "$MODEL"

# 组2: OCR=ON  Evidence=ON  Base
run_if_missing "2/8 OCR=ON Evidence=ON Base" "group2_ocr_on_evi_on_base.json" \
  --base-only --test-jsonl "$TEST" --base-model "$MODEL"

# 组3: OCR=ON  Evidence=OFF  LoRA
run_if_missing "3/8 OCR=ON Evidence=OFF LoRA" "group3_ocr_on_evi_off_lora.json" \
  --adapter "$ADAPTER" --test-jsonl "$TEST" --base-model "$MODEL" --no-evidence-prompt

# 组4: OCR=ON  Evidence=OFF  Base
run_if_missing "4/8 OCR=ON Evidence=OFF Base" "group4_ocr_on_evi_off_base.json" \
  --base-only --test-jsonl "$TEST" --base-model "$MODEL" --no-evidence-prompt

# 组5: OCR=OFF  Evidence=ON  LoRA
run_if_missing "5/8 OCR=OFF Evidence=ON LoRA" "group5_ocr_off_evi_on_lora.json" \
  --adapter "$ADAPTER" --test-jsonl "$TEST" --base-model "$MODEL" --no-ocr

# 组6: OCR=OFF  Evidence=ON  Base
run_if_missing "6/8 OCR=OFF Evidence=ON Base" "group6_ocr_off_evi_on_base.json" \
  --base-only --test-jsonl "$TEST" --base-model "$MODEL" --no-ocr

# 组7: OCR=OFF  Evidence=OFF  LoRA
run_if_missing "7/8 OCR=OFF Evidence=OFF LoRA" "group7_ocr_off_evi_off_lora.json" \
  --adapter "$ADAPTER" --test-jsonl "$TEST" --base-model "$MODEL" --no-ocr --no-evidence-prompt

# 组8: OCR=OFF  Evidence=OFF  Base
run_if_missing "8/8 OCR=OFF Evidence=OFF Base" "group8_ocr_off_evi_off_base.json" \
  --base-only --test-jsonl "$TEST" --base-model "$MODEL" --no-ocr --no-evidence-prompt

echo ""
echo "=========================================="
echo "  消融实验全部完成"
echo "  结果目录: $OUTDIR"
echo "=========================================="
echo ""
echo "运行汇总脚本:"
echo "  python3 scripts/summarize_ablation.py"
