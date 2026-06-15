#!/usr/bin/env python3
"""
Markdown → PDF（Edge 无头模式，Windows 自带浏览器）
依赖: pip install markdown
用法: python md2pdf.py
"""

import sys
import io
# 强制 stdout 使用 UTF-8，避免 GBK 编码错误
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import base64
import re
import subprocess
import markdown

BASE = Path(__file__).resolve().parent
MD_PATH = BASE / "实验报告.md"
HTML_PATH = BASE / "实验报告_print.html"
PDF_PATH = BASE / "实验报告.pdf"
# Windows 短路径名避免空格解析问题
EDGE = r"C:\PROGRA~2\Microsoft\Edge\Application\msedge.exe"

# ── 打印优化 CSS（A4） ──
CSS = """
@page {
  size: A4;
  margin: 1.6cm 2cm 1.6cm 2cm;
}

body {
  font-family: "SimSun", "Microsoft YaHei", serif;
  font-size: 11.5pt;
  line-height: 1.85;
  color: #222;
}

h1 {
  font-size: 21pt;
  text-align: center;
  margin-top: 0.8cm;
  margin-bottom: 0.6cm;
}

h2 {
  font-size: 15pt;
  margin-top: 1cm;
  margin-bottom: 0.4cm;
  border-bottom: 1.5px solid #333;
  padding-bottom: 4px;
}

h3 { font-size: 12.5pt; margin-top: 0.6cm; margin-bottom: 0.3cm; }

p { margin: 0.4em 0; text-align: justify; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5cm 0;
  font-size: 10pt;
}
th, td {
  border: 1px solid #555;
  padding: 5px 8px;
  text-align: center;
  vertical-align: middle;
}
th { background-color: #eef; font-weight: bold; }

blockquote {
  background: #f9f9f9;
  border-left: 4px solid #888;
  margin: 0.5em 0;
  padding: 0.5em 1em;
  font-size: 11pt;
  color: #444;
}

pre {
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 0.6em;
  font-size: 9pt;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
code {
  font-family: "Consolas", "Courier New", monospace;
  font-size: 9.5pt;
  background: #f0f0f0;
  padding: 1px 4px;
  border-radius: 3px;
}
pre code { background: none; padding: 0; }

img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.3cm auto;
}
"""


def img_to_b64(path: Path) -> str:
    """图片转 base64 data URI"""
    ext = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def embed_images(html: str) -> str:
    """本地图片 → base64 内嵌"""
    def repl(m: re.Match):
        src = m.group(1)
        if src.startswith("http") or src.startswith("data:") or src.startswith("/"):
            return m.group(0)
        img_path = BASE / src
        if img_path.exists():
            b64 = img_to_b64(img_path)
            return f'src="{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]+)"', repl, html)


def main():
    print("[1/3] Markdown → HTML")
    md_text = MD_PATH.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    body = md.convert(md_text)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>实验报告</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    html = embed_images(html)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"  [OK] HTML -> {HTML_PATH}")

    print("[2/3] Edge 无头打印 PDF")
    html_url = HTML_PATH.resolve().as_uri()

    cmd = [
        EDGE,
        "--headless=new",
        f"--print-to-pdf={PDF_PATH.resolve()}",
        "--no-pdf-header-footer",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--no-first-run",
        "--no-default-browser-check",
        "--user-data-dir=C:\\Temp\\edge-pdf-profile",
        html_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if PDF_PATH.exists():
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"  [OK] PDF -> {PDF_PATH} ({size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] PDF 生成失败")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"  returncode: {result.returncode}")
        if result.stdout:
            print(f"  stdout: {result.stdout[:300]}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        sys.exit(1)

    print("[3/3] 清理临时文件")
    HTML_PATH.unlink(missing_ok=True)
    print(f"  [OK] 完成")


if __name__ == "__main__":
    main()