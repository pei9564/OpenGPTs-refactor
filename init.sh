#!/bin/bash

# 設置 embedding model
EMBEDDING_MODEL="DMetaSoul/Dmeta-embedding-zh"  # 確認是否和環境變數一致
EMBEDDING_MODEL_DIR="./backend/cache/models--${EMBEDDING_MODEL//\//--}"

if [[ -d "$EMBEDDING_MODEL_DIR" ]]; then
    echo "Embedding model already exists. Skipping script execution."
else
    echo "Embedding model does not exist. Executing setup script..."
    ./load_embedding_model.sh
fi

# 啟動 docker
docker-compose build
docker-compose up
