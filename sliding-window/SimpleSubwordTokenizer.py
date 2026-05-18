import re
import torch
from torch.utils.data import Dataset, DataLoader

# --- Tokenizer and helpers ---

def tokenize(text):
    # Splits text into words and punctuation using regex
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def build_vocab(tokens):
    # Builds vocabulary mappings: token to ID and ID to token
    unique_tokens = sorted(set(tokens))
    token_to_id = {token: i for i, token in enumerate(unique_tokens)}
    id_to_token = {i: token for token, i in token_to_id.items()}
    return token_to_id, id_to_token

def decode(ids, id_to_token):
    # Converts a list of token IDs back to readable text
    tokens = [id_to_token[int(i)] for i in ids]
    text = ""
    for token in tokens:
        # Attach punctuation directly, otherwise add space
        if token in [".", ",", "!", "?", ":", ";"]:
            text += token
        else:
            if text:
                text += " "
            text += token
    return text

class SlidingWindowDataset(Dataset):
    """
    Dataset for next-token prediction using a sliding window.
    Each input is a sequence of tokens, and the target is the same sequence shifted by one token.
    """
    def __init__(self, token_ids, max_length, stride):
        self.inputs = []
        self.targets = []
        # Create input and target windows for next-token prediction
        for start in range(0, len(token_ids) - max_length, stride):
            input_window = token_ids[start : start + max_length]
            target_window = token_ids[start + 1 : start + max_length + 1]
            self.inputs.append(torch.tensor(input_window, dtype=torch.long))
            self.targets.append(torch.tensor(target_window, dtype=torch.long))
    def __len__(self):
        # Number of windows in the dataset
        return len(self.inputs)
    def __getitem__(self, index):
        # Returns input and target tensors for a given window
        return self.inputs[index], self.targets[index]

# --- Main script ---

# Example German text for demonstration
text = """Python ist eine Programmiersprache. Ein Sprachmodell lernt aus Texten. Das Modell sagt das nächste Token voraus."""

# Tokenize the text
tokens = tokenize(text)

# Build vocabulary mappings
token_to_id, id_to_token = build_vocab(tokens)

# Convert tokens to their corresponding IDs
token_ids = [token_to_id[token] for token in tokens]

# Parameters for sliding window
max_length = 4    # Length of each input window
stride = 1        # Step size for the sliding window
batch_size = 2    # Number of samples per batch

# Create dataset and dataloader
dataset = SlidingWindowDataset(token_ids, max_length, stride)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Print tokens and their IDs for reference
print("Tokens:", tokens)
print("Token-IDs:", token_ids)
print(f"Number of windows: {len(dataset)}")

# Show each input-target window pair
for i in range(len(dataset)):
    input_ids, target_ids = dataset[i]
    print(f"Window {i}:")
    print("  Input IDs:", input_ids.tolist())
    print("  Target IDs:", target_ids.tolist())
    print("  Input Text:", decode(input_ids.tolist(), id_to_token))
    print("  Target Text:", decode(target_ids.tolist(), id_to_token))

# Show the first batch from the dataloader
print("\nFirst batch:")
batch_inputs, batch_targets = next(iter(dataloader))
print("Batch input shape:", batch_inputs.shape)
print("Batch target shape:", batch_targets.shape)
for i in range(len(batch_inputs)):
    print(f"Example {i+1}:")
    print("  Input:", decode(batch_inputs[i].tolist(), id_to_token))
    print("  Target:", decode(batch_targets[i].tolist(), id_to_token))