"""调试：测试单样本处理"""
import json, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

PROJ = Path(__file__).resolve().parents[1]
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

# 找第一条有图片的样本
with open(PROJ / "data/training/chartqa/merged_val.jsonl") as f:
    for line in f:
        s = json.loads(line)
        img_path = PROJ / s["image_path"]
        if img_path.exists():
            break
print(f"image: {img_path}")
print(f"question: {s['question'][:60]}")
print(f"answer: {s['answer'][:60]}")

image = Image.open(img_path).convert("RGB")
print(f"image size: {image.size}")

messages_no_asst = [
    {"role": "system", "content": [{"type": "text", "text": "你是一个中文图文问答助手。请根据图片内容简洁准确地回答问题。"}]},
    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": s["question"]}]},
]

prompt_text = processor.apply_chat_template(messages_no_asst, tokenize=False, add_generation_prompt=True)
print(f"\nprompt (last 200 chars): ...{prompt_text[-200:]}")

# 方式1: text + image
inputs1 = processor(text=[prompt_text], images=[image], return_tensors="pt")
print(f"\n方式1 (text+image):")
print(f"  input_ids len: {inputs1['input_ids'].shape}")
print(f"  pixel_values: {inputs1['pixel_values'].shape if inputs1.get('pixel_values') is not None else None}")
print(f"  image_grid_thw: {inputs1['image_grid_thw']}")

# 方式2: messages + image
inputs2 = processor(text=[messages_no_asst], images=[image], return_tensors="pt")
print(f"\n方式2 (messages+image):")
print(f"  input_ids len: {inputs2['input_ids'].shape}")
print(f"  pixel_values: {inputs2['pixel_values'].shape if inputs2.get('pixel_values') is not None else None}")
print(f"  image_grid_thw: {inputs2['image_grid_thw']}")

# 检查 image token 数
thw = inputs1["image_grid_thw"][0]
expected_tokens = int(thw[0] * thw[1] * thw[2] / processor.image_processor.merge_size ** 2)
print(f"\n  expected image tokens (merge_size={processor.image_processor.merge_size}): {expected_tokens}")
print(f"  features: {inputs1['pixel_values'].shape[0]}")

import sys; sys.exit(1)
