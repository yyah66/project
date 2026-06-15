#!/bin/bash
# ============================================================
# 使用 ModelScope 下载 EasyOCR 模型到数据盘
# ModelScope 国内访问稳定，不需要代理
#
# 用法:
#   bash scripts/download_easyocr_models.sh
# ============================================================

CACHE_DIR="${EASYOCR_MODULE_PATH:-/root/autodl-tmp/cache/easyocr}"
TARGET="$CACHE_DIR/model"

mkdir -p "$TARGET"

echo "=========================================="
echo "  EasyOCR 模型下载"
echo "  源: ModelScope"
echo "  目标: $TARGET"
echo "=========================================="

cd "$TARGET"

NEEDED=()
for f in craft_mlt_25k.pth zh_sim_g2.pth english_g2.pth; do
    if [ -s "$f" ]; then
        echo "[跳过] $f (已存在)"
    else
        NEEDED+=("$f")
    fi
done

if [ ${#NEEDED[@]} -eq 0 ]; then
    echo "所有模型已就绪。"
else
    echo "需要下载 ${#NEEDED[@]} 个文件: ${NEEDED[*]}"
    echo ""

    # 用 Python + modelscope 下载（国内无障碍）
    python3 -c "
import os, sys, shutil
from modelscope.hub.snapshot_download import snapshot_download

target = '$TARGET'
needed = set(${NEEDED[@]@Q})

print(f'从 ModelScope 下载 EasyOCR 模型...')
try:
    cache_dir = snapshot_download(
        'easyocr/EasyOCR',
        cache_dir=target,
        revision='master',
    )
    print(f'下载到: {cache_dir}')
    # ModelScope 下载的文件名可能不同，尝试匹配
    import glob
    for f in glob.glob(os.path.join(cache_dir, '**', '*.pth'), recursive=True):
        name = os.path.basename(f)
        dest = os.path.join(target, name)
        if name in needed and not os.path.exists(dest):
            shutil.copy2(f, dest)
            print(f'  ✓ 复制: {name}')
        elif not os.path.exists(dest):
            shutil.copy2(f, dest)
            print(f'  ✓ 复制: {name}')
except Exception as e:
    print(f'ModelScope 下载失败: {e}')
    print('')
    print('手动备选方案:')
    print('  1. 用 HF 镜像:')
    print('     pip install modelscope  # 确保已安装')
    print('  2. 从本地上传:')
    print(f'     把以下文件上传到 {target}/ :')
    for f in needed:
        print(f'       - {f}')
    sys.exit(1)
"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "  自动下载失败，手动备选方案:"
        echo "=========================================="
        echo ""
        echo "  1. 从你的 Windows 本地上传模型文件:"
        echo "     Windows 路径: C:\\Users\\<用户名>\\.EasyOCR\\model\\"
        echo "     上传到 AutoDL: $TARGET/"
        echo ""
        echo "  2. 或者尝试 HF 镜像下载 (逐个执行):"
        for f in "${NEEDED[@]}"; do
            echo "     wget https://hf-mirror.com/EasyOCR/EasyOCR/resolve/main/$f -O $TARGET/$f"
        done
        echo ""
        echo "  3. 或者用阿里云 OSS 备用链接:"
        echo "     # 这几个链接不一定有效，取决于 PaddleOCR 版本"
        echo ""
        echo "  下载完成后重新运行:"
        echo "     python3 scripts/check_gpu_ocr.py"
    fi
fi
