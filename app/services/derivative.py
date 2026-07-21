import re, random


async def generate_derivative_variants(
    question_text: str,
    problem_schema: dict,
    difficulty: int,
    target_difficulty: int,
    subject: str,
    count: int,
    use_llm: bool,
    llm_generator=None,
):
    """Generate the requested variants and record the method used for each."""
    if llm_generator is None:
        from app.services.llm import generate_derivative as llm_generator

    variants = []
    for _ in range(count):
        if use_llm:
            try:
                text = await llm_generator(
                    question_text=question_text,
                    problem_schema=problem_schema,
                    difficulty=difficulty,
                    target_difficulty=target_difficulty,
                    subject=subject,
                )
                variants.append((text, "llm"))
                continue
            except Exception:
                pass

        text = generate_derivative_rule(
            question_text,
            problem_schema,
            target_difficulty,
            subject,
        )
        variants.append((text, "rule"))

    return variants

def generate_derivative_rule(
    question_text: str,
    problem_schema: dict,
    target_difficulty: int,
    subject: str,
) -> str:
    """规则模板生成衍生题（不依赖 LLM）"""
    if subject == "math":
        return _math_derivative(question_text, problem_schema, target_difficulty)
    elif subject == "chinese":
        return _chinese_derivative(question_text, target_difficulty)
    elif subject == "english":
        return _english_derivative(question_text, target_difficulty)
    return question_text  # fallback: 原题

def _math_derivative(text: str, schema: dict, target_diff: int) -> str:
    """提取数字 → 乘以难度系数替换"""
    numbers = re.findall(r'\d+', text)
    if not numbers:
        return text
    factor = 1.5 + (target_diff - 1) * 0.5
    result = text
    for n in sorted(set(numbers), key=int, reverse=True):
        new_n = str(max(1, int(int(n) * factor + random.randint(0, 10))))
        result = result.replace(n, new_n, 1)  # 只替换第一个出现
    return result

def _chinese_derivative(text: str, target_diff: int) -> str:
    # 语文降级：保持原题不变（规则生成质量差，标记为原题）
    return text + "（规则生成，请人工确认）"

def _english_derivative(text: str, target_diff: int) -> str:
    # 英语降级：简单词汇替换
    word_map = {
        "apple": "banana", "cat": "dog", "big": "large",
        "happy": "glad", "run": "walk", "eat": "drink",
    }
    result = text
    for old, new in word_map.items():
        if old in result:
            result = result.replace(old, new, 1)
            break
    return result
