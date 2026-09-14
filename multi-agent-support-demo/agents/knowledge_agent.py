from ollama_client import generate
from rag.rag_service import RagService


class KnowledgeAgent:

    def __init__(self):
        self.rag_service = RagService()

    def search(self, problem: str) -> dict:

        rag_result = self.rag_service.search(
            problem
        )

        prompt = f"""
Du bist ein spezialisierter IT-Knowledge-Agent.

Beantworte das IT-Problem ausschließlich anhand
des bereitgestellten internen Wissens.

Problem:

{problem}

Internes Wissen:

-------------------
{rag_result["content"]}
-------------------

Erstelle eine kurze und verständliche
Handlungsempfehlung.

Erfinde keine Informationen,
die nicht im internen Wissen stehen.
"""

        answer = generate(prompt)

        return {
            "answer": answer.strip(),
            "source": rag_result["filename"],
            "score": rag_result["score"]
        }
