import httpx, json
from app.config import settings

ANALYSIS_PROMPT = """分析以下{subject_hint}题目，返回JSON：

{{
  "subject": "math" | "chinese" | "english",
  "question_type": "calculation" | "word_problem" | "fill_blank" | "choice" | "reading",
  "problem_schema": {{...}},
  "difficulty_params": {{...}},
  "tags": ["...", "..."],
  "difficulty": 1-5
}}

题目:
{question_text}"""

DERIVATIVE_PROMPT = """基于以下题目，生成一道难度提升的衍生题。

原题结构: {problem_schema}
当前难度: {difficulty}
目标难度: {target_difficulty}
科目: {subject}

要求：
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）
- 语文：同知识点换语境
- 英语：替换词汇、变化时态

只返回衍生题文本，不要解析。"""


async def analyze_question(question_text: str, subject_hint: str = "") -> dict:
    """调用 LLM 分析题目，返回结构化信息"""
    prompt = ANALYSIS_PROMPT.format(
        subject_hint=subject_hint or "自动判断",
        question_text=question_text,
    )
    result = await _call_llm(prompt)
    return _parse_json(result)


async def generate_derivative(
    question_text: str,
    problem_schema: dict,
    difficulty: int,
    target_difficulty: int,
    subject: str,
) -> str:
    """调用 LLM 生成衍生题"""
    prompt = DERIVATIVE_PROMPT.format(
        problem_schema=json.dumps(problem_schema, ensure_ascii=False),
        difficulty=difficulty,
        target_difficulty=target_difficulty,
        subject=subject,
    )
    return await _call_llm(prompt)


async def _call_llm(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个教育题目分析助手。只返回要求的JSON格式，不要额外解释。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {data}")
        return data["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
    return json.loads(raw)
