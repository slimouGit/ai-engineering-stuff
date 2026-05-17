import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Tokenisierung
# ------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [token.strip() for token in text.split() if token.strip()]


# ------------------------------------------------------------
# 2. Demo-Embedding
# ------------------------------------------------------------

def token_to_vector(token: str) -> np.ndarray:
    token = token.lower()
    vector = np.zeros(4)
    for char in token:
        code = ord(char)
        vector[0] += code % 7
        vector[1] += code % 11
        vector[2] += code % 13
        vector[3] += code % 17
    if len(token) > 0:
        vector = vector / len(token)
    return vector


# ------------------------------------------------------------
# 3. Softmax
# ------------------------------------------------------------

def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


# ------------------------------------------------------------
# 4. Self-Attention
# ------------------------------------------------------------

def calculate_attention(tokens: list[str]) -> np.ndarray:
    vectors = np.array([token_to_vector(token) for token in tokens])
    attention_rows = []
    for i, query in enumerate(vectors):
        scores = []
        for j, key in enumerate(vectors):
            score = float(np.dot(query, key))
            if tokens[i].lower() == "sie":
                if tokens[j].lower() in ["katze", "frau", "mutter", "person"]:
                    score += 20
                if tokens[j].lower() in ["hund", "mann", "vater"]:
                    score += 10
            scores.append(score)
        attention_rows.append(softmax(np.array(scores)))
    return np.array(attention_rows)


# ------------------------------------------------------------
# Helpers: Streamlit-Check
# ------------------------------------------------------------

def is_streamlit_running() -> bool:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ------------------------------------------------------------
# Streamlit-App (wie zuvor), mit sicherer Behandlung von Pandas-Styling
# ------------------------------------------------------------

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Self-Attention Demo", layout="wide")
    st.title("Self-Attention Demo")

    st.write(
        "Diese App zeigt vereinfacht, wie ein neuronales Netz bei einem Token entscheidet, "
        "welche anderen Tokens im Satz besonders wichtig sind."
    )

    sentence = st.text_area(
        "Satz eingeben",
        value="Der Hund jagt die Katze weil sie wegrennt",
        height=100
    )

    tokens = tokenize(sentence)

    if not tokens:
        st.warning("Bitte gib einen Satz ein.")
        st.stop()

    attention = calculate_attention(tokens)

    st.subheader("1. Tokens")
    st.write(tokens)

    st.subheader("2. Attention-Matrix")
    attention_percent = np.round(attention * 100, 1)

    df = pd.DataFrame(
        attention_percent,
        index=[f"{i}: {token}" for i, token in enumerate(tokens)],
        columns=[f"{i}: {token}" for i, token in enumerate(tokens)]
    )

    st.write("**Zeile** = aktuelles Token  \n**Spalte** = Token, auf das geachtet wird  \n**Wert** = Attention-Gewicht in Prozent")

    # Sicheres Styling: falls matplotlib fehlt -> Fallback auf einfache Tabelle
    try:
        styled = df.style.background_gradient(axis=None)
        # Replace `use_container_width` with `width='stretch'` (Streamlit Warnung)
        st.dataframe(styled, width="stretch")
    except Exception:
        st.warning("Styling nicht möglich (wahrscheinlich fehlt `matplotlib`). Zeige einfache Tabelle.")
        st.dataframe(df, width="stretch")

    st.subheader("3. Einzelnes Token analysieren")

    selected_token = st.selectbox(
        "Welches Token möchtest du analysieren?",
        options=list(range(len(tokens))),
        format_func=lambda i: f"{i}: {tokens[i]}"
    )

    weights = attention[selected_token]
    analysis_df = pd.DataFrame({
        "Token": tokens,
        "Attention-Gewicht": weights,
        "Prozent": np.round(weights * 100, 1)
    }).sort_values("Attention-Gewicht", ascending=False)

    st.write(f'Das Token **{tokens[selected_token]}** achtet am stärksten auf:')
    st.bar_chart(analysis_df.set_index("Token")["Prozent"])
    st.dataframe(analysis_df[["Token", "Prozent"]], width="stretch")

    st.subheader("4. Erklärung")
    st.write(
        "In einem echten Transformer würde jedes Token gelernte Vektoren erhalten:\n\n"
        "- **Query**: Wonach sucht dieses Token?\n"
        "- **Key**: Was bietet ein anderes Token an?\n"
        "- **Value**: Welche Information wird übernommen?\n\n"
        "Vereinfachte Formel:\n\n"
    )
    st.code("score = Query · Key\nattention = softmax(scores)\ncontext = Summe(attention * Value)")


# ------------------------------------------------------------
# CLI-Fallback für PyCharm / direktes Ausführen
# ------------------------------------------------------------

def run_cli_mode():
    print("Kein Streamlit-Kontext erkannt. Starte CLI-Demo.\n")
    print("Gib einen Satz ein (oder leer lassen für Demo). `exit` zum Beenden.\n")

    default = "Der Hund jagt die Katze weil sie wegrennt"

    while True:
        try:
            s = input("Satz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBeendet.")
            return

        if not s:
            s = default
            print(f"(Demo-Satz) {s}")

        if s.lower() in ("exit", "quit", "q"):
            print("Beendet.")
            return

        tokens = tokenize(s)
        if not tokens:
            print("Leerer Satz. Bitte Text eingeben.\n")
            continue

        attention = calculate_attention(tokens)
        attention_percent = np.round(attention * 100, 1)

        print("\nTokens:")
        for i, t in enumerate(tokens):
            print(f"{i}: {t}")
        print("\nAttention-Matrix (Prozent):")
        # tabellarische Ausgabe
        hdr = " \t" + "\t".join([f"{i}" for i in range(len(tokens))])
        print(hdr)
        for i, row in enumerate(attention_percent):
            print(f"{i}\t" + "\t".join(f"{v:.1f}" for v in row))

        # Analyse eines Tokens
        try:
            sel = input("\nIndex des zu analysierenden Tokens (leer = skip): ").strip()
            if sel:
                idx = int(sel)
                if 0 <= idx < len(tokens):
                    weights = attention[idx]
                    order = np.argsort(-weights)
                    print(f'\n{tokens[idx]} achtet am stärksten auf:')
                    for i in order:
                        print(f"  {tokens[i]}: {weights[i]*100:.1f}%")
                else:
                    print("Index außerhalb Bereich.")
        except Exception:
            print("Ungültige Eingabe, weiter.\n")

        print("\n---\n")


if __name__ == "__main__":
    if is_streamlit_running():
        run_streamlit_app()
    else:
        run_cli_mode()