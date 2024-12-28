#!/bin/bash

# 確保腳本在遇到錯誤時退出
set -e

# 配置嵌入模型名稱（可根據需要修改）
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"DMetaSoul/Dmeta-embedding-zh"}

# 定義支持的 Python 命令列表（按常見順序檢查）
PYTHON_CANDIDATES=("python3.11" "python3.10" "python3.9" "python3.8" "python3.7" "python3")

# 設置最大版本限制
MAX_VERSION="3.12"

# 定義符合條件的 Python 版本變量
PYTHON_CMD=""

# 遍歷候選 Python 命令，檢查版本是否符合條件
echo "Checking for Python version below $MAX_VERSION..."
for cmd in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if (( $(echo "$VERSION < $MAX_VERSION" | bc -l) )); then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

# 確保找到符合條件的 Python 版本
if [[ -z $PYTHON_CMD ]]; then
    echo "Error: No Python version below $MAX_VERSION found. Please install a suitable Python version."
    exit 1
else
    echo "Using Python version: $($PYTHON_CMD --version)"
fi

# 設定虛擬環境相關路徑
OS="$(uname)"
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    ACTIVATE="./venv/bin/activate"
    PIP="./venv/bin/pip"
    PYTHON_EXEC="./venv/bin/$PYTHON_CMD"
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    ACTIVATE="./venv/Scripts/activate"
    PIP="./venv/Scripts/pip"
    PYTHON_EXEC="./venv/Scripts/$PYTHON_CMD"
else
    echo "Error: Unsupported OS: $OS"
    exit 1
fi

# 創建虛擬環境
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# 激活虛擬環境
echo "Activating virtual environment..."
source $ACTIVATE

# 安裝所需依賴
echo "Installing dependencies..."
$PIP install --upgrade pip
$PIP install torch
$PIP install sentence-transformers
$PIP install langchain_community

# 執行 HuggingFace 模型設置
echo "Running HuggingFace model setup..."
if [[ ! -d "backend" ]]; then
    echo "Error: 'backend' directory not found."
    deactivate
    exit 1
fi

$PYTHON_EXEC -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='$EMBEDDING_MODEL', cache_folder='./backend/cache')"

# 清理環境
echo "Deactivating and cleaning up..."
deactivate
rm -rf venv

echo "Setup complete! HuggingFace model is ready."
