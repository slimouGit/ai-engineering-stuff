import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

BASE_DIR = Path(__file__).resolve().parent
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TOOL_ROUNDS = 8


def mcp_tool_to_ollama(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def result_to_text(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=str)

    parts: list[str] = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else str(result)


def message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", ""),
    }


async def answer_question(session: ClientSession, ollama: AsyncClient, tools: list[dict[str, Any]], question: str) -> str:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Du bist ein lokaler Ticket-Assistent. Nutze die angebotenen Tools, "
                "wenn du Ticketdaten brauchst. Behaupte nie, ein Ticket gelesen oder "
                "geändert zu haben, ohne das passende Tool aufzurufen. Antworte auf Deutsch."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = await ollama.chat(model=MODEL, messages=messages, tools=tools)
        assistant = response.message
        messages.append(message_to_dict(assistant))

        tool_calls = assistant.tool_calls or []
        if not tool_calls:
            return assistant.content or "Das Modell hat keine Textantwort geliefert."

        for call in tool_calls:
            name = call.function.name
            arguments = call.function.arguments or {}
            print(f"  → MCP-Tool: {name}({arguments})")

            try:
                result = await session.call_tool(name, arguments=arguments)
                content = result_to_text(result)
            except Exception as exc:
                content = json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

            messages.append({"role": "tool", "tool_name": name, "content": content})

    return "Abbruch: Das Modell hat zu viele Tool-Aufrufe hintereinander erzeugt."


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
    )

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        listed = await session.list_tools()
        tools = [mcp_tool_to_ollama(tool) for tool in listed.tools]
        ollama = AsyncClient(host=OLLAMA_HOST)

        print("Lokale Ollama + MCP Demo")
        print(f"Modell: {MODEL}")
        print("MCP-Tools:", ", ".join(tool.name for tool in listed.tools))
        print("Beispiele:")
        print("  - Zeige mir Ticket 1001.")
        print("  - Suche Tickets zum Thema Login.")
        print("  - Setze Ticket 1001 auf RESOLVED.")
        print("Mit 'exit' beenden.\n")

        while True:
            question = input("Du: ").strip()
            if question.lower() in {"exit", "quit", "ende"}:
                break
            if not question:
                continue

            try:
                answer = await answer_question(session, ollama, tools, question)
                print(f"\nAssistent: {answer}\n")
            except Exception as exc:
                print(f"\nFehler: {exc}")
                print("Prüfe, ob Ollama läuft und das konfigurierte Modell installiert ist.\n")


if __name__ == "__main__":
    asyncio.run(main())
