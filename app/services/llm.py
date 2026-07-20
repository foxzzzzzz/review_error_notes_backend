import httpx, json, re
from app.config import settings

ANALYSIS_PROMPT = """分析以下{subject_hint}题目，将结构化结果放在 <output>...</output> 标签中。

要求输出严格的JSON格式：
{{
  "subject": "math" | "chinese" | "english",
  "question_type": "calculation" | "word_problem" | "fill_blank" | "choice" | "reading",
  "problem_schema": {{...}},
  "difficulty_params": {{...}},
  "tags": ["...", "..."],
  "difficulty": 1-5
}}

题目: {question_text}

严格按照以下格式输出，不要有任何额外内容在 <output> 标签之外：
<output>
{{"subject": "...", "question_type": "...", "problem_schema": {{}}, "difficulty_params": {{}}, "tags": [], "difficulty": 3}}
</output>"""

DERIVATIVE_PROMPT = """基于以下题目，生成一道难度提升的衍生题。

原题结构: {problem_schema}
当前难度: {difficulty}
目标难度: {target_difficulty}
科目: {subject}

要求：
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）
- 语文：同知识点，替换字词或调整语境
- 英语：替换词汇、变化时态

严格按照以下格式输出，衍生题放在 <output> 标签内：
<output>
[衍生题文本]
</output>"""

OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.DOTALL)


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


def _extract_output(raw: str) -> str:
    """Extract content between <output> tags. Falls back to raw text."""
    m = OUTPUT_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


async def _call_llm(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{settings.LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 8192,
            },
        )
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error {resp.status_code}: {data}")
        # Prefer content; fall back to reasoning_content (reasoning models)
        msg = data["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        return _extract_output(raw)


def _parse_json(raw: str) -> dict:
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        raw = "\n".join(lines)
    return json.loads(raw)
