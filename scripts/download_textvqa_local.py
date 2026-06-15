"""
本地 Windows 下载 TextVQA 图片（需要代理/科学上网）
1. 把服务器上的 data/public/textvqa_val_5000.jsonl 下载到本机同路径
2. 运行本脚本
3. 把 data/public/images/textvqa/ 目录打包上传回服务器
"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GCS_URL = "https://storage.googleapis.com/openimages/v6/images/{}.jpg"
JSONL_PATH = "data/public/textvqa_val_5000.jsonl"
IMG_DIR = Path("data/public/images/textvqa")
MAX_WORKERS = 10
TIMEOUT = 30


def extract_ids() -> list[str]:
    ids: set[str] = set()
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            path = row["image_path"]  # e.g. data/public/images/textvqa/abc123.jpg
            fname = path.rsplit("/", 1)[-1]
            if fname.endswith(".jpg"):
                ids.add(fname[:-4])
    return sorted(ids)


def download(img_id: str) -> tuple[str, str]:
    fname = f"{img_id}.jpg"
    fpath = IMG_DIR / fname
    if fpath.exists():
        return fname, "skip"
    for t in range(1, 4):
        try:
            url = GCS_URL.format(img_id)
            data = urllib.request.urlopen(url, timeout=TIMEOUT).read()
            fpath.write_bytes(data)
            return fname, "ok"
        except Exception as e:
            if t == 3:
                return fname, str(e)
            time.sleep(1)
    return fname, "fail"


def main():
    ids = extract_ids()
    print(f"需要 {len(ids)} 张")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(download, i): i for i in ids}
        for f in as_completed(futs):
            _, s = f.result()
            if s == "ok":
                ok += 1
            elif s != "skip":
                fail += 1
            if (ok + fail) % 100 == 0:
                print(f"  {ok+fail}/{len(ids)} (ok {ok})")
    print(f"完成: {ok} ok, {fail} fail")
    print(f"图片在: {IMG_DIR.resolve()}")
    print("打包命令: tar -czf textvqa_images.tar.gz -C data/public/images textvqa")


if __name__ == "__main__":
    main()
