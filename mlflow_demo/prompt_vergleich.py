import mlflow
from openai import OpenAI

from mlflow_demo.config import MY_API_KEY, MODEL

# ============================================================
# HIER DEINEN OPENAI API KEY EINTRAGEN
# ============================================================
OPENAI_API_KEY = MY_API_KEY

if not OPENAI_API_KEY:
    raise ValueError("Bitte setze die Umgebungsvariable OPENAI_API_KEY.")


# MLflow-Server und Experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Prompt Vergleich")
mlflow.openai.autolog()

client = OpenAI(api_key=OPENAI_API_KEY)

prompts = [
    "Erkläre Machine Learning in einem Satz.",
    "Erkläre Machine Learning für einen Anfänger.",
    "Erkläre Machine Learning technisch und präzise.",
]

for i, prompt in enumerate(prompts, start=1):

    with mlflow.start_run(run_name=f"prompt_{i}"):

        mlflow.log_param("prompt", prompt)
        mlflow.log_param("model",  MODEL)

        response = client.chat.completions.create(
            model= MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        answer = response.choices[0].message.content

        mlflow.log_text(answer, "response.txt")

        print(answer)