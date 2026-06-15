import os

base = 'project-v3/--main'

# 实验报告附录A中列出的所有文件
required = [
    'outputs/lora/merged/final_adapter/adapter_config.json',
    'outputs/lora/merged/final_adapter/adapter_model.safetensors',
    'outputs/lora/merged/final_adapter/README.md',
    'outputs/lora/merged_ocr/final_adapter/adapter_config.json',
    'outputs/lora/merged_ocr/final_adapter/adapter_model.safetensors',
    'outputs/eval/baseline.jsonl',
    'outputs/eval/lora.jsonl',
    'outputs/ablation/group1_ocr_on_evi_on_lora.json',
    'outputs/ablation/group2_ocr_on_evi_on_base.json',
    'outputs/ablation/group3_ocr_on_evi_off_lora.json',
    'outputs/ablation/group4_ocr_on_evi_off_base.json',
    'outputs/ablation/group5_ocr_off_evi_on_lora.json',
    'outputs/ablation/group6_ocr_off_evi_on_base.json',
    'outputs/ablation/group7_ocr_off_evi_off_lora.json',
    'outputs/ablation/group8_ocr_off_evi_off_base.json',
    'outputs/ablation/metrics_summary.json',
    'outputs/ablation/summary.md',
    'outputs/charts/01_ablation_8groups_em.png',
    'outputs/charts/02_main_effects.png',
    'outputs/charts/03_training_loss_curve.png',
    'outputs/charts/04_error_distribution_pie.png',
    'outputs/charts/05_lora_vs_base_radar.png',
    'outputs/charts/06_interaction_heatmap.png',
    'outputs/charts/07_evidence_boost.png',
    'outputs/charts/08_dataset_composition.png',
    'outputs/charts/09_evidence_coverage.png',
    'app.py',
    'generate_charts.py',
]

ok = []
missing = []

for path in required:
    full = os.path.join(base, path)
    if os.path.exists(full):
        size = os.path.getsize(full)
        ok.append((path, size))
    else:
        missing.append(path)

print("=" * 60)
print(f"检验结果：{len(ok)} 通过, {len(missing)} 缺失")
print("=" * 60)

if missing:
    print("\n[MISSING] 缺失文件:")
    for m in missing:
        print(f"   MISSING: {m}")

print(f"\n[OK] 已存在文件 ({len(ok)}):")
for p, s in ok:
    unit = 'B'
    sz = s
    if sz >= 1024:
        sz /= 1024; unit = 'KB'
    if sz >= 1024:
        sz /= 1024; unit = 'MB'
    print(f"   {p}  ({sz:.1f} {unit})")

print()
if missing:
    print("WARNING: 存在缺失文件，需要补充")
else:
    print("SUCCESS: 所有文件完整")
