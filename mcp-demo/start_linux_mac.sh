#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull qwen3:4b
export OLLAMA_MODEL=qwen3:4b
python app.py
