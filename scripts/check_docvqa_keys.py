"""检查 DocVQA 数据集字段名"""
import os; os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
ds = load_dataset('nielsr/docvqa_1200_examples', split='train', streaming=True, trust_remote_code=True)
r = next(iter(ds))
print("keys:", list(r.keys()))
for k in r.keys():
    v = r[k]
    print(f"  {k}: {type(v).__name__}", end="")
    if hasattr(v, "size"):
        print(f"  size={v.size}")
    elif isinstance(v, dict):
        print(f"  dict keys={list(v.keys())}  sample={v}")
    elif isinstance(v, list) and len(v) > 0:
        print(f"  list[{len(v)}]  first={v[0]}")
    elif isinstance(v, str) and len(v) > 120:
        print(f"  = {v[:100]}...")
    elif isinstance(v, str):
        print(f"  = {v}")
    else:
        print(f"  = {v}")
