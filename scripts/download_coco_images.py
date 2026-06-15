"""
并发下载 COCO val2014 图片（从 VQA-v2 JSONL 中提取所需图片 ID）
用法: python3 scripts/download_coco_images.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://images.cocodataset.org/val2014"
JSONL_PATH = "data/public/vqa_v2_2000.jsonl"
IMG_DIR = Path("data/public/images/vqa_v2")
MAX_WORKERS = 20
TIMEOUT = 20
MAX_RETRIES = 3


def extract_ids(jsonl_path: Path) -> list[str]:
    ids: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            path = row.get("image_path", "")
            # e.g. data/public/images/vqa_v2/COCO_val2014_000000012345.jpg
            fname = path.rsplit("/", 1)[-1]
            if fname.startswith("COCO_val2014_") and fname.endswith(".jpg"):
                ids.add(fname[len("COCO_val2014_"):-len(".jpg")])
    return sorted(ids)


def download(img_id: str) -> tuple[str, str]:
    fname = f"COCO_val2014_{img_id}.jpg"
    fpath = IMG_DIR / fname
    if fpath.exists():
        return fname, "skip"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            url = f"{BASE_URL}/{fname}"
            data = urllib.request.urlopen(url, timeout=TIMEOUT).read()
            fpath.write_bytes(data)
            return fname, "ok"
        except Exception as e:
            if attempt == MAX_RETRIES:
                return fname, str(e)
            time.sleep(1)
    return fname, "fail"


def main() -> None:
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL 不存在: {jsonl_path}")

    ids = extract_ids(jsonl_path)
    print(f"需要下载 {len(ids)} 张图片")

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    total = len(ids)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download, i): i for i in ids}
        for future in as_completed(futures):
            _, status = future.result()
            if status == "ok":
                ok += 1
            elif status != "skip":
                fail += 1
            done = ok + fail
            if done % 200 == 0:
                print(f"  进度 {done}/{total} (成功 {ok})")

    print(f"完成: {ok} 成功, {fail} 失败, 总计 {total}")


if __name__ == "__main__":
    main()
