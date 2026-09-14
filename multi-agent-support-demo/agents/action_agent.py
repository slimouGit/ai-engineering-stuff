import json

from ollama_client import generate


class ActionAgent:

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def process(
        self,
        ticket_info: dict
    ) -> dict:

        category = ticket_info["category"]
        summary = ticket_info["summary"]

        search_result = await self.mcp_client.call_tool(
            "search_tickets",
            {
                "search_text": (
                    f"{category} {summary}"
                )
            }
        )

        existing_tickets = self._read_result(
            search_result
        )

        decision = self._evaluate_tickets(
            summary,
            existing_tickets
        )

        if decision["found"]:

            ticket_id = decision["ticket_id"]

            ticket_result = (
                await self.mcp_client.call_tool(
                    "get_ticket",
                    {
                        "ticket_id": ticket_id
                    }
                )
            )

            ticket = self._read_result(
                ticket_result
            )

            return {
                "action": "EXISTING_TICKET",
                "ticket": ticket
            }

        create_result = (
            await self.mcp_client.call_tool(
                "create_ticket",
                {
                    "category": category,
                    "title": summary,
                    "description": summary
                }
            )
        )

        created_ticket = self._read_result(
            create_result
        )

        return {
            "action": "CREATED_TICKET",
            "ticket": created_ticket
        }

    def _evaluate_tickets(
        self,
        summary: str,
        tickets
    ) -> dict:

        if not tickets:
            return {
                "found": False,
                "ticket_id": None
            }

        tickets_json = json.dumps(
            tickets,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
Du bist ein spezialisierter IT-Action-Agent.

Prüfe, ob eines der gefundenen Tickets
inhaltlich zum aktuellen Problem passt.

Aktuelles Problem:

{summary}

Gefundene Tickets:

{tickets_json}

Wenn ein Ticket dasselbe oder ein sehr ähnliches
Problem beschreibt, gib dessen ID zurück.

Antworte ausschließlich als JSON.

Wenn eines passt:

{{
  "found": true,
  "ticket_id": 1
}}

Wenn keines passt:

{{
  "found": false,
  "ticket_id": null
}}
"""

        result = generate(prompt)

        try:
            return json.loads(result)

        except json.JSONDecodeError:

            cleaned = (
                result
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            try:
                return json.loads(cleaned)

            except json.JSONDecodeError:
                return {
                    "found": False,
                    "ticket_id": None
                }

    @staticmethod
    def _read_result(result):

        texts = []

        for content in result.content:

            text = getattr(
                content,
                "text",
                None
            )

            if text:
                texts.append(text)

        if not texts:
            return None

        result_text = "".join(texts)

        try:
            return json.loads(result_text)

        except json.JSONDecodeError:
            return result_text
