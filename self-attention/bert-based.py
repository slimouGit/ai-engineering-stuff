import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)

text = input("Enter a sentence: ")
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple: (num_layers, batch, num_heads, seq_len, seq_len)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# Example: use attention from the last layer, first head
attention = attentions[-1][0, 0].numpy()

print("\nTokens:")
for idx, token in enumerate(tokens):
    print(f"{idx}: {token}")

print("\nAttention-Matrix (Percent):")
print("    " + "    ".join(str(i) for i in range(len(tokens))))
for i, row in enumerate(attention):
    percents = "    ".join(f"{100 * v:5.1f}" for v in row)
    print(f"{i:2}  {percents}")