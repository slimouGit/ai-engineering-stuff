Hier den lokalen Transformers-/Safetensors-Modellordner ablegen:

models/qwen2.5-0.5b-instruct/

Die App lädt daraus ausschließlich lokal (local_files_only=True).

py -m pip install -U huggingface_hub
py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b-instruct')"
py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v0.6', local_dir='models/tinyllama-1.1b-chat-v0.6')"