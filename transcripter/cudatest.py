import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nutze Gerät: {device}")