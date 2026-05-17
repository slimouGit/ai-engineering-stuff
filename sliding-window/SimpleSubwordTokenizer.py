import re
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# (Klassen und Funktionen unverändert, nur modularisiert für Streamlit/CLI)

DEFAULT_TEXT = """
Python ist eine Programmiersprache.
Ein Sprachmodell lernt aus Texten.
Der DataLoader erzeugt Batches aus Trainingsdaten.

Moderne Sprachmodelle arbeiten meist nicht mit einzelnen Zeichen.
Sie verwenden Tokens, also Wörter oder Teilwörter.
Aus dem Satz wird eine Folge von Token-IDs.
Das Modell lernt, das nächste Token vorherzusagen.

Neuronale Netze lernen Muster.
Beim Training werden Eingaben und Zielwerte verglichen.
Der Fehler wird berechnet und das Modell wird angepasst.
"""


class SimpleSubwordTokenizer:

    def __init__(self, text: str):

        # Spezialtokens
        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<NL>"
        ]

        # Text tokenisieren
        tokens = self.tokenize(text)

        # Vokabular erzeugen
        vocab = self.special_tokens + sorted(set(tokens))

        # Token -> ID
        self.stoi = {
            token: idx
            for idx, token in enumerate(vocab)
        }

        # ID -> Token
        self.itos = {
            idx: token
            for token, idx in self.stoi.items()
        }

    def split_word(self, word: str):
        if len(word) <= 8:
            return [word]

        parts = []
        parts.append(word[:6])
        rest = word[6:]
        while rest:
            parts.append("##" + rest[:3])
            rest = rest[3:]
        return parts

    def tokenize(self, text: str):
        raw_tokens = re.findall(
            r"\n|\w+|[^\w\s]",
            text,
            flags=re.UNICODE
        )

        tokens = []
        for token in raw_tokens:
            if token == "\n":
                tokens.append("<NL>")
            elif token.isalnum():
                tokens.extend(self.split_word(token))
            else:
                tokens.append(token)
        return tokens

    def encode(self, text: str):
        tokens = self.tokenize(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]

    def decode(self, ids):
        tokens = [
            self.itos.get(int(idx), "<UNK>")
            for idx in ids
        ]

        text = ""
        for token in tokens:
            if token == "<PAD>":
                continue
            if token == "<NL>":
                text += "\n"
                continue
            if token.startswith("##"):
                text += token[2:]
                continue
            if token in [".", ",", "!", "?", ":", ";"]:
                text += token
                continue
            if text and not text.endswith((" ", "\n")):
                text += " "
            text += token
        return text.strip()


class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, max(0, len(token_ids) - max_length), stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.target_ids[idx])


def create_dataloader_v1(txt, batch_size=4, max_length=16, stride=4, shuffle=True, drop_last=True):
    tokenizer = SimpleSubwordTokenizer(txt)
    dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader, tokenizer, dataset


class TinyLanguageModel(nn.Module):

    def __init__(self, vocab_size, emb_dim=96, hidden_dim=160):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        logits = self.output(x)
        return logits


def generate_text(model, tokenizer, start_text, max_length, max_new_tokens=40):
    model.eval()
    generated_ids = tokenizer.encode(start_text)
    if not generated_ids:
        generated_ids = [tokenizer.stoi["<NL>"]]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated_ids[-max_length:]
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)
    return tokenizer.decode(generated_ids)


def predict_next_token(model, tokenizer, text, max_length, top_k=5):
    model.eval()
    token_ids = tokenizer.encode(text)
    if len(token_ids) == 0:
        return []
    token_ids = token_ids[-max_length:]
    x = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            token = tokenizer.decode([idx.item()])
            predictions.append({
                "token": token,
                "probability": float(prob.item())
            })
        return predictions


# --- Streamlit detection ---
def is_streamlit_running():
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# --- Streamlit App (original UI) ---
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="LLM Dataset Demo", layout="wide")
    st.title("Dataset, DataLoader und Tokenisierung")
    st.markdown("""
    Diese App demonstriert vereinfacht das Prinzip moderner Sprachmodelle.

    Ablauf:

    1. Text wird tokenisiert
    2. Sliding Windows erzeugen Trainingsdaten
    3. DataLoader erstellt Batches
    4. Das Modell lernt Next-Token-Prediction
    5. Das Modell generiert neuen Text
    """)

    text = st.text_area("Trainingsdaten", value=DEFAULT_TEXT, height=260)

    col1, col2, col3 = st.columns(3)
    with col1:
        max_length = st.slider("max_length", 4, 64, 16)
    with col2:
        stride = st.slider("stride", 1, 32, 4)
    with col3:
        batch_size = st.slider("batch_size", 1, 16, 4)

    if len(text.strip()) < 30:
        st.warning("Der Text ist zu kurz.")
        st.stop()

    dataloader, tokenizer, dataset = create_dataloader_v1(txt=text, batch_size=batch_size, max_length=max_length, stride=stride)
    if len(dataset) == 0:
        st.warning("Text zu kurz für die gewählte max_length.")
        st.stop()

    st.header("1. Tokenisierung")
    tokens = tokenizer.tokenize(text)
    st.write("Anzahl Tokens:", len(tokens))
    st.write("Vokabulargröße:", len(tokenizer.stoi))
    st.subheader("Erste Tokens")
    st.code(str(tokens[:80]), language="python")
    st.subheader("Erste Token-IDs")
    st.code(str(tokenizer.encode(text)[:80]), language="python")

    with st.expander("Vokabular anzeigen"):
        st.code(str(tokenizer.stoi), language="python")

    st.header("2. Dataset")
    sample_input, sample_target = dataset[0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Input IDs")
        st.code(str(sample_input.tolist()), language="python")
        st.subheader("Input Text")
        st.code(tokenizer.decode(sample_input.tolist()), language="text")
    with col_b:
        st.subheader("Target IDs")
        st.code(str(sample_target.tolist()), language="python")
        st.subheader("Target Text")
        st.code(tokenizer.decode(sample_target.tolist()), language="text")

    st.header("3. DataLoader")
    inputs, targets = next(iter(dataloader))
    st.write("Input Shape:")
    st.code(str(tuple(inputs.shape)), language="python")
    st.write("Target Shape:")
    st.code(str(tuple(targets.shape)), language="python")

    st.markdown("Shape:\n\n```text\n(batch_size, max_length)\n```")

    st.header("4. Modell - Generierung (ungefittet)")
    if st.button("Generiere Beispieltext"):
        model = TinyLanguageModel(vocab_size=len(tokenizer.stoi))
        out = generate_text(model, tokenizer, start_text="Python ist", max_length=max_length, max_new_tokens=40)
        st.code(out, language="text")


# --- CLI / PyCharm Mode ---
def run_cli_mode():
    print("Kein Streamlit-Kontext erkannt. Starte PyCharm\/CLI-Demo.\n")
    text = DEFAULT_TEXT
    print("Verwendeter Demo-Text (erste 300 Zeichen):\n")
    print(text.strip()[:300] + "\n")
    dataloader, tokenizer, dataset = create_dataloader_v1(txt=text, batch_size=4, max_length=16, stride=4)
    print(f"Anzahl Tokens: {len(tokenizer.tokenize(text))}")
    print(f"Vokabulargröße: {len(tokenizer.stoi)}")
    print("\nErste Tokens:", tokenizer.tokenize(text)[:40])
    print("\nErste Token-IDs:", tokenizer.encode(text)[:40])

    if len(dataset) == 0:
        print("\nText zu kurz für die gewählte max_length.")
        return

    sample_input, sample_target = dataset[0]
    print("\nSample Input IDs:", sample_input.tolist())
    print("Sample Input Text:\n", tokenizer.decode(sample_input.tolist()))
    print("\nSample Target IDs:", sample_target.tolist())
    print("Sample Target Text:\n", tokenizer.decode(sample_target.tolist()))

    model = TinyLanguageModel(vocab_size=len(tokenizer.stoi))
    print("\nGeneriere Beispieltext (ungefittetes Modell):\n")
    generated = generate_text(model, tokenizer, start_text="Python ist", max_length=16, max_new_tokens=30)
    print(generated)
    print("\nFertig.")


if __name__ == "__main__":
    if is_streamlit_running():
        run_streamlit_app()
    else:
        run_cli_mode()