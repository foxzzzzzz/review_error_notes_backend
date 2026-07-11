"""LLM analysis for wrong questions — optional, requires LLM_API_KEY to be set."""
from app.config import settings
import httpx

ANALYSIS_PROMPT = (
    "You are a tutor. Analyze the following wrong-exam question text. "
    "Return a JSON object with these fields:\n"
    "- subject: one of \"math\", \"chinese\", \"english\"\n"
    "- question_type: brief description of the question type\n"
    "- tags: array of short keyword strings\n"
    "- difficulty: integer 1-5 (5 hardest)\n"
    "Return ONLY valid JSON, no commentary."
)


async def analyze_question(text: str) -> dict:
    """Call LLM to analyze a single question text. Returns structured metadata."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": ANALYSIS_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            },
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    import json
    return json.loads(content)
