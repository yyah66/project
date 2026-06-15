#!/bin/bash
# ============================================================
# AutoDL 数据盘环境变量配置
# 只管大的：HF_HOME（模型 ~15GB）和 PIP 缓存
# OCR 模型（~100MB）使用默认路径，不管
#
# 用法（在 AutoDL 终端执行一次）:
#   source scripts/setup_autodl.sh
# ============================================================

DATA_ROOT="/root/autodl-tmp"
if [ ! -d "$DATA_ROOT" ]; then
    echo "未检测到 AutoDL 数据盘，跳过配置。"
    return 0 2>/dev/null || exit 0
fi

MARKER="### AUTODL HF/PIP CACHE SETUP ###"

# 先清理旧标记
if [ -f ~/.bashrc ]; then
    sed -i "/$MARKER/,### END AUTODL CACHE SETUP ###/d" ~/.bashrc
fi

# 写入新配置
cat >> ~/.bashrc << EOF
$MARKER
export HF_HOME="$DATA_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
### END AUTODL CACHE SETUP ###
EOF

# 立即生效
export HF_HOME="$DATA_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"

# 创建目录
mkdir -p "$DATA_ROOT/cache/huggingface"
mkdir -p "$DATA_ROOT/cache/pip"

echo ""
echo "  HF_HOME       = $HF_HOME"
echo "  PIP_CACHE_DIR = $PIP_CACHE_DIR"
echo ""
echo "  大文件（HuggingFace 模型、pip 缓存）→ 数据盘 ✓"
echo "  OCR 小模型（~100MB）→ 默认路径，不管"
echo ""
echo "  新终端需要: source ~/.bashrc"
