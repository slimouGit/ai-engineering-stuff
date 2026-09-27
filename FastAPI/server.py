import json
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="Ollama API")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_URL}/api/tags")
        return {"api": "ok", "ollama": response.is_success}

@app.post("/api/chat")
async def chat(payload: dict):
    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{OLLAMA_URL}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")