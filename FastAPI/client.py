# python
import json
import requests

def main():
    user_input = input("Enter your message: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        return

    payload = {
        "model": "qwen2.5:1.5b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Explain your answers clearly and be concise unless more detail is requested."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "options": {
            "temperature": 1.15,
            "top_p": 0.95,
            "top_k": 80,
            "num_predict": 1024,
            "num_ctx": 8192,
            "repeat_penalty": 1.05,
            "seed": None,
            "stop": []
        },
        "think": False,
        "keep_alive": "5m",
        "logprobs": False,
        "top_logprobs": 0,
        "format": "text"
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=payload,
            stream=True,
            timeout=None,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            print(event.get("message", {}).get("content", ""), end="", flush=True)
            if event.get("done"):
                print()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()