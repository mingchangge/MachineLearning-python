from huggingface_hub import snapshot_download

# 指定模型 ID
model_id = "mlc-ai/Qwen2.5-1.5B-Instruct-q4f32_1-MLC"

# 下载到本地目录（例如 ./Qwen2.5-1.5B-Instruct-q4f32_1-MLC）
local_dir = "./models/Qwen2.5-1.5B-Instruct-q4f32_1-MLC"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)