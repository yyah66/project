"""检查评测预测输出"""
import json, re, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

base = "/root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "outputs/lora/merged/final_adapter")
model.eval()

with open("data/training/chartqa/merged_test.jsonl") as f:
    lines = [json.loads(l) for l in f if l.strip()][:5]

for s in lines:
    p = Path(s["image_path"])
    if not p.exists():
        continue
    try:
        img = Image.open(p); img.load()
    except Exception:
        continue
    image = Image.open(p).convert("RGB")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "你是一个中文图文问答助手。请根据图片内容简洁准确地回答问题。"}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": s["question"]}]},
    ]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    resp = processor.decode(gen[0], skip_special_tokens=True)
    print(f"Q: {s['question'][:80]}")
    print(f"GT: {s['answer'][:80]}")
    print(f"RAW: {resp[-300:]}")
    print("---")
