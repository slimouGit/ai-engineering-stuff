import json

import ollama_client
from ollama import generate


class TicketAgent:

    def analyze(self, user_input: str) -> dict:

        prompt = f"""
Du bist ein spezialisierter IT-Ticket-Agent.

Deine Aufgabe ist ausschließlich die Analyse
eingehender IT-Probleme.

Benutzeranfrage:

"{user_input}"

Ermittle:

- category
- priority
- summary

Erlaubte Kategorien:

LOGIN
VPN
PASSWORD
SOFTWARE
OTHER

Erlaubte Prioritäten:

LOW
MEDIUM
HIGH

Antworte ausschließlich mit gültigem JSON.

Beispiel:

{{
  "category": "LOGIN",
  "priority": "HIGH",
  "summary": "Login funktioniert nach Softwareupdate nicht"
}}
"""

        result = generate(prompt)

        return self._parse_json(result, user_input)

    @staticmethod
    def _parse_json(result: str, user_input: str) -> dict:

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
                    "category": "OTHER",
                    "priority": "MEDIUM",
                    "summary": user_input
                }
