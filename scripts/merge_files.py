import shutil, os

base = 'project-v3/--main'
补充 = os.path.join(base, '补充文件')

# 1. 复制 merged/final_adapter ── 补充文件 -> outputs/lora/merged
src_merged = os.path.join(补充, 'merged', 'final_adapter')
dst_merged = os.path.join(base, 'outputs', 'lora', 'merged', 'final_adapter')
if os.path.exists(src_merged):
    os.makedirs(dst_merged, exist_ok=True)
    for f in os.listdir(src_merged):
        sf = os.path.join(src_merged, f)
        df = os.path.join(dst_merged, f)
        if not os.path.exists(df):
            shutil.copy2(sf, df)
            print(f'复制: {f} -> outputs/lora/merged/final_adapter/')

# 2. 复制 ablation JSON 文件
src_ablation = os.path.join(补充, 'ablation')
dst_ablation = os.path.join(base, 'outputs', 'ablation')
if os.path.exists(src_ablation):
    os.makedirs(dst_ablation, exist_ok=True)
    for f in os.listdir(src_ablation):
        sf = os.path.join(src_ablation, f)
        df = os.path.join(dst_ablation, f)
        if not os.path.exists(df):
            shutil.copy2(sf, df)
            print(f'复制: {f} -> outputs/ablation/')

print('合并完成')