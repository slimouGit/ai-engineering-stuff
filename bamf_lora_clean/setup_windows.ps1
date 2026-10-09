$ErrorActionPreference = "Stop"

Write-Host "1/4 Python prüfen"
py --version

Write-Host "2/4 Python-Pakete installieren"
py -m pip install --upgrade pip
py -m pip install torch
py -m pip install -r requirements.txt

Write-Host "3/4 llama.cpp bereitstellen"
if (-not (Test-Path "llama.cpp\convert_hf_to_gguf.py")) {
    if (Test-Path "llama.cpp") {
        Write-Host "llama.cpp-Ordner ist unvollständig. Bitte Ordner löschen und Setup erneut starten."
        exit 1
    }
    git clone https://github.com/ggml-org/llama.cpp.git
}

Write-Host "4/4 Abhängigkeiten des GGUF-Konverters installieren"
py -m pip install -r llama.cpp\requirements.txt

Write-Host ""
Write-Host "Setup fertig. Jetzt: py app.py"
