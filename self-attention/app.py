import re
import numpy as np

_token_vectors = {}

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def token_to_vector(token: str) -> np.ndarray:
    # Assign a random vector to each unique token
    if token not in _token_vectors:
        np.random.seed(hash(token) % 2**32)
        _token_vectors[token] = np.random.rand(4)
    return _token_vectors[token]

def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)

def calculate_attention(tokens: list[str]) -> np.ndarray:
    vectors = np.array([token_to_vector(token) for token in tokens])
    scores = np.dot(vectors, vectors.T)
    attention = np.apply_along_axis(softmax, 1, scores)
    return attention

if __name__ == "__main__":
    text = input("Enter a sentence: ")
    tokens = tokenize(text)
    attention = calculate_attention(tokens)

    print("\nTokens:")
    for idx, token in enumerate(tokens):
        print(f"{idx}: {token}")

    print("\nAttention-Matrix (Percent):")
    print("\t" + "\t".join(str(i) for i in range(len(tokens))))
    for i, row in enumerate(attention):
        percents = "\t".join(f"{100 * v:.1f}" for v in row)
        print(f"{i}\t{percents}")