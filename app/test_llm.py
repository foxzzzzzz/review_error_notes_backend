"""Quick LLM connectivity + format test. Run inside container: python app/test_llm.py"""
import sys, os
sys.path.insert(0, "/app")

# Test 1: raw connectivity
print("=== Test 1: Raw API call ===")
import httpx
r = httpx.post(
    f'{os.getenv("LLM_API_BASE")}/chat/completions',
    headers={"Authorization": f'Bearer {os.getenv("LLM_API_KEY")}'},
    json={"model": os.getenv("LLM_MODEL"), "messages": [{"role":"user","content":"回复OK"}], "max_tokens": 50},
    timeout=30,
)
data = r.json()
msg = data["choices"][0]["message"]
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
print(f"Status: {r.status_code}, Model: {data.get('model')}")
print(f"Content ({len(content)} chars): {content[:100]}")
print(f"Reasoning ({len(reasoning)} chars): {reasoning[:100]}")
assert content or reasoning, "Both content and reasoning_content are empty!"
print("PASS\n")

# Test 2: analyze_question format
print("=== Test 2: analyze_question ===")
from app.services.llm import analyze_question
result = analyze_question("小明有12个苹果，吃了3个，还剩几个？", "数学")
print(f"Subject: {result.get('subject')}")
print(f"Type: {result.get('question_type')}")
print(f"Tags: {result.get('tags')}")
print(f"Difficulty: {result.get('difficulty')}")
print("PASS\n")

# Test 3: generate_derivative format
print("=== Test 3: generate_derivative ===")
from app.services.llm import generate_derivative
derived = generate_derivative(
    "小明有12个苹果，吃了3个，还剩几个？",
    {"operation": "subtraction", "operands": [12, 3]},
    difficulty=2,
    target_difficulty=4,
    subject="math",
)
print(f"Derived ({len(derived)} chars): {derived[:200]}")
assert len(derived) > 5, "Derivative too short"
print("PASS")
