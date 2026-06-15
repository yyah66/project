"""
Qwen2.5-VL LoRA / QLoRA 训练脚本（OCR + Evidence 增强版）

用法:
  python scripts/train_qwen2vl_lora.py \
    --train-jsonl data/training/merged/merged_train_enhanced.jsonl \
    --val-jsonl data/training/merged/merged_val_enhanced.jsonl \
    --image-root . \
    --output-dir outputs/lora/merged_ocr

RTX 5090 32GB 推荐参数 (全精度 LoRA):
  --batch-size 4 --gradient-accumulation 4 --lora-rank 64

显存不足时启用 QLoRA 4-bit:
  --use-qlora

训练产物:
  outputs/lora/<name>/
    adapter_config.json
    adapter_model.safetensors
    trainer_state.json
    checkpoint-*/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt import build_system_prompt, build_user_prompt
from src.ocr import OCRLine, OCRResult, ImageAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reconstruct_ocr_result(sample: dict) -> OCRResult:
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


def build_assistant_json(
    answer: str,
    evidence: list[str] | None = None,
    confidence: str = "高",
    uncertainty: str = "",
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "evidence": evidence if evidence is not None else [],
            "confidence": confidence,
            "uncertainty": uncertainty,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VLADataset(Dataset):
    """Vision-language QA dataset with OCR-enhanced prompts."""

    def __init__(
        self,
        jsonl_path: Path,
        image_root: Path,
        processor: AutoProcessor,
        max_length: int = 4096,
        max_ocr_lines: int = 24,
        evidence_enabled: bool = True,
    ) -> None:
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        self.max_ocr_lines = max_ocr_lines
        self.evidence_enabled = evidence_enabled
        self.samples: list[dict] = []

        has_ocr = False
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                image_path = Path(sample["image_path"])
                if not image_path.is_absolute():
                    image_path = image_root / image_path
                if not image_path.exists():
                    logger.warning("跳过缺失图片: %s", image_path)
                    continue
                self.samples.append({
                    "image_path": str(image_path),
                    "question": sample["question"],
                    "answer": sample.get("answer", ""),
                    "ocr_lines": sample.get("ocr_lines", []),
                    "ocr_analysis": sample.get("ocr_analysis", None),
                    "evidence": sample.get("evidence", []),
                })
                if sample.get("ocr_lines"):
                    has_ocr = True

        if not has_ocr:
            logger.warning("%s 不含 ocr_lines 字段，将使用空 OCR 结果。请先运行 precompute_all_ocr.py。", jsonl_path)
        logger.info("加载 %d 条样本 from %s", len(self.samples), jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        question = sample["question"]
        answer = sample["answer"]

        ocr_result = _reconstruct_ocr_result(sample)

        system_text = build_system_prompt(self.evidence_enabled)
        user_prompt_text = build_user_prompt(
            question, ocr_result, [], self.max_ocr_lines, self.evidence_enabled,
        )
        assistant_text = build_assistant_json(
            answer,
            evidence=sample.get("evidence") if sample.get("evidence") else None,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt_text},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        assistant_tag = "<|im_start|>assistant\n"
        assistant_pos = prompt.rfind(assistant_tag)
        if assistant_pos >= 0:
            prompt_without_answer = prompt[:assistant_pos + len(assistant_tag)]
        else:
            prompt_without_answer = prompt

        inputs = self.processor(
            text=[prompt_without_answer],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        full_inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        input_ids = full_inputs["input_ids"][0]
        prompt_len = inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = torch.ones_like(input_ids)

        pv = full_inputs.get("pixel_values", None)
        if pv is not None and pv.dim() == 3:
            pv = pv.squeeze(0)
        thw = full_inputs.get("image_grid_thw", None)
        if thw is not None and thw.dim() == 2:
            thw = thw.squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pv,
            "image_grid_thw": thw,
        }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class VLADataCollator:
    processor: AutoProcessor
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_len = max(ids.shape[0] for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for i in range(len(features)):
            pad_len = max_len - input_ids[i].shape[0]
            padded_input_ids.append(
                torch.cat([input_ids[i], torch.full((pad_len,), self.pad_token_id, dtype=input_ids[i].dtype)])
            )
            padded_labels.append(
                torch.cat([labels[i], torch.full((pad_len,), -100, dtype=labels[i].dtype)])
            )
            padded_attention_mask.append(
                torch.cat([attention_mask[i], torch.zeros(pad_len, dtype=attention_mask[i].dtype)])
            )

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_mask),
        }

        pvs = [f["pixel_values"] for f in features if f["pixel_values"] is not None]
        if pvs:
            batch["pixel_values"] = torch.cat(pvs, dim=0)

        thws = [f["image_grid_thw"] for f in features if f["image_grid_thw"] is not None]
        if thws:
            batch["image_grid_thw"] = torch.stack(thws, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-VL on VQA data")
    parser.add_argument("--train-jsonl", required=True, help="训练集 JSONL 路径")
    parser.add_argument("--val-jsonl", required=True, help="验证集 JSONL 路径")
    parser.add_argument("--image-root", default=".", help="图片根目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct", help="基座模型路径或 HuggingFace ID")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-ocr-lines", type=int, default=24)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--use-qlora", action="store_true", help="启用 4-bit QLoRA（显存不足时使用）")
    parser.add_argument("--no-evidence", action="store_true", help="关闭证据提示词（消融用）")
    parser.add_argument("--resume-from", default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_jsonl = Path(args.train_jsonl)
    val_jsonl = Path(args.val_jsonl)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)

    if not train_jsonl.exists():
        raise SystemExit(f"训练集不存在: {train_jsonl}")
    if not val_jsonl.exists():
        raise SystemExit(f"验证集不存在: {val_jsonl}")

    logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("显存: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0)

    # ---- 加载 processor ----
    logger.info("加载 processor: %s", args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28

    # ---- 加载基座模型 ----
    bnb_config = None
    if args.use_qlora:
        logger.info("启用 QLoRA 4-bit 量化")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("加载基座模型: %s", args.model_path)
    try:
        import flash_attn as _  # noqa: F401
        _attn_impl = "flash_attention_2"
        logger.info("已启用 Flash Attention 2")
    except ImportError:
        _attn_impl = "sdpa"
        logger.info("Flash Attention 2 未安装，回退到 SDPA")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation=_attn_impl,
    )
    model.config.use_cache = False

    # ---- 注入 LoRA ----
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 数据集 ----
    evidence_enabled = not args.no_evidence
    train_dataset = VLADataset(
        train_jsonl, image_root, processor,
        max_length=args.max_length, max_ocr_lines=args.max_ocr_lines,
        evidence_enabled=evidence_enabled,
    )
    val_dataset = VLADataset(
        val_jsonl, image_root, processor,
        max_length=args.max_length, max_ocr_lines=args.max_ocr_lines,
        evidence_enabled=evidence_enabled,
    )

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    data_collator = VLADataCollator(processor, pad_token_id)

    # ---- 训练配置 ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        gradient_checkpointing=True,
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ---- 开始训练 ----
    logger.info("="*60)
    logger.info("开始 LoRA 训练")
    logger.info("  基座模型: %s", args.model_path)
    logger.info("  训练集: %s (%d 条)", train_jsonl, len(train_dataset))
    logger.info("  验证集: %s (%d 条)", val_jsonl, len(val_dataset))
    logger.info("  LoRA rank: %d, alpha: %d", args.lora_rank, args.lora_alpha)
    logger.info("  Effective batch size: %d", args.batch_size * args.gradient_accumulation)
    logger.info("  Max length: %d, Max OCR lines: %d", args.max_length, args.max_ocr_lines)
    logger.info("  Evidence enabled: %s", evidence_enabled)
    logger.info("  QLoRA: %s", "是" if args.use_qlora else "否")
    logger.info("="*60)

    trainer.train(resume_from_checkpoint=args.resume_from)

    # ---- 保存最终 adapter ----
    final_adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(final_adapter_path))
    logger.info("LoRA adapter 已保存到 %s", final_adapter_path)

    # ---- 保存训练信息 ----
    info = {
        "base_model": args.model_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_ocr_lines": args.max_ocr_lines,
        "evidence_enabled": evidence_enabled,
        "use_qlora": args.use_qlora,
    }
    (output_dir / "training_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("训练完成。")


if __name__ == "__main__":
    main()
