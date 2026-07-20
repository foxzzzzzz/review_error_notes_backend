"""LLM connectivity + format protocol test. Run: python app/test_llm.py"""
import os, sys, json, re, httpx

def main():
    KEY = os.getenv("LLM_API_KEY", "")
    BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    URL = f"{BASE}/chat/completions"
    if not KEY:
        print("ERROR: LLM_API_KEY not set")
        return

    fail = False

    # ── Test 1: Raw connectivity ──
    print("=== Test 1: Connectivity ===")
    try:
        r = httpx.get(f"{BASE}/models",
            headers={"Authorization": f"Bearer {KEY}"}, timeout=15)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            models = [m["id"] for m in r.json()["data"]]
            print(f"  Models: {models}")
    except Exception as e:
        print(f"  FAIL: {e}")
        fail = True

    # ── Test 2: Chat (basic) ──
    print("\n=== Test 2: Chat ===")
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content":"回复OK"}], "max_tokens": 50},
            timeout=30)
        d = r.json()
        msg = d["choices"][0]["message"]
        c = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        print(f"  Status: {r.status_code}  model: {d.get('model')}")
        print(f"  content({len(c)}): {c[:120]}")
        print(f"  reasoning({len(rc)}): {rc[:120]}")
        if not c and not rc:
            print("  ❌ Both content and reasoning empty")
            fail = True
    except Exception as e:
        print(f"  FAIL: {e}")
        fail = True

    # ── Test 3: Analyze question ──
    print("\n=== Test 3: Analyze Question ===")
    PROMPT = """分析以下数学题目，将结构化结果放在 <output> 标签中，标签内必须是合法JSON。

JSON格式示例（用实际分析结果替换）：
<output>
{"subject": "math", "question_type": "calculation", "problem_schema": {"operation": "subtraction", "operands": [35, 17]}, "difficulty_params": {"num_range": [1, 100], "steps": 1}, "tags": ["减法", "两位数"], "difficulty": 3}
</output>

题目: 小明有12个苹果，吃了3个，还剩几个？"""
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content": PROMPT}],
                  "temperature": 0.3, "max_tokens": 4096},
            timeout=60)
        d = r.json()
        msg = d["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        print(f"  Status: {r.status_code}  raw: {len(raw)} chars")

        m = re.search(r"<output>\s*(.*?)\s*</output>", raw, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            inner_clean = inner
            if inner_clean.startswith("```"):
                inner_clean = inner_clean.split("\n", 1)[-1]
                if inner_clean.endswith("```"):
                    inner_clean = inner_clean[:-3]
            parsed = json.loads(inner_clean)
            print(f"  ✅ subject={parsed.get('subject')}")
            print(f"     type={parsed.get('question_type')}")
            print(f"     tags={parsed.get('tags')}")
            print(f"     difficulty={parsed.get('difficulty')}")
        else:
            print(f"  ❌ No <output> tag. First 500 chars:\n{raw[:500]}")
            fail = True
    except Exception as e:
        print(f"  FAIL: {e}")
        fail = True

    # ── Test 4: Generate derivative ──
    print("\n=== Test 4: Generate Derivative ===")
    DPROMPT = """基于以下题目，生成一道难度提升的衍生题，只把衍生题文本放在 <output> 标签内。

原题: 小明有12个苹果，吃了3个，还剩几个？
原题结构: {"operation": "subtraction", "operands": [12, 3]}
当前难度: 2 / 目标难度: 4
科目: math

输出格式示例：
<output>
小明有48颗糖，第一天吃了15颗，第二天吃了9颗，还剩几颗？
</output>"""
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content": DPROMPT}],
                  "temperature": 0.3, "max_tokens": 4096},
            timeout=60)
        d = r.json()
        msg = d["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        print(f"  Status: {r.status_code}  raw: {len(raw)} chars")

        m = re.search(r"<output>\s*(.*?)\s*</output>", raw, re.DOTALL)
        if m:
            derived = m.group(1).strip()
            print(f"  ✅ Derived ({len(derived)} chars): {derived[:200]}")
        else:
            print(f"  ❌ No <output> tag. First 500 chars:\n{raw[:500]}")
            fail = True
    except Exception as e:
        print(f"  FAIL: {e}")
        fail = True

    # ── Result ──
    print("\n" + "=" * 40)
    if fail:
        print("❌ Some tests failed")
    else:
        print("🎉 All tests passed!")


if __name__ == "__main__":
    main()
