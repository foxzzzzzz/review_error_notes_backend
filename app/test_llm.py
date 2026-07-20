"""LLM connectivity + format protocol test. Run: python app/test_llm.py"""
import os, sys, json, re, socket

def main():
    KEY = os.getenv("LLM_API_KEY", "")
    BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
    if not KEY:
        print("ERROR: LLM_API_KEY not set. Run: export LLM_API_KEY=sk-xxx")
        return
    URL = f"{BASE}/chat/completions"
    results = {}

    # ── Test 1: DNS ──
    print("=== Test 1: DNS ===")
    try:
        addr = socket.getaddrinfo("api.deepseek.com", 443)
        print(f"OK: {addr[0][4]}")
        results["dns"] = True
    except Exception as e:
        print(f"FAIL: {e}")
        results["dns"] = False

    # ── Test 2: TCP ──
    print("\n=== Test 2: TCP ===")
    try:
        s = socket.socket(); s.settimeout(10)
        s.connect(("api.deepseek.com", 443)); s.close()
        print("OK")
        results["tcp"] = True
    except Exception as e:
        print(f"FAIL: {e}")
        results["tcp"] = False

    if not results.get("tcp"):
        print("\n❌ Network unreachable. Check docker network / firewall.")
        _summary(results)
        return

    # ── Test 3: Raw API ──
    print("\n=== Test 3: Raw API ===")
    import httpx
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content":"回复OK"}], "max_tokens": 50},
            timeout=30)
        d = r.json()
        msg = d["choices"][0]["message"]
        c = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        print(f"Status:{r.status_code} model:{d.get('model')}")
        print(f"content({len(c)}): {c[:120]}")
        print(f"reasoning({len(rc)}): {rc[:120]}")
        results["api"] = bool(c or rc)
        print("OK" if results["api"] else "FAIL: empty response")
    except Exception as e:
        print(f"FAIL: {e}")
        results["api"] = False

    # ── Test 4: analyze_question ──
    print("\n=== Test 4: Analyze Question ===")
    PROMPT = """分析以下数学题目，将结构化结果放在 <output> 标签中，标签内必须是合法JSON。

JSON格式示例（用实际分析结果替换）：
<output>
{"subject": "math", "question_type": "calculation", "problem_schema": {"operation": "subtraction", "operands": [35, 17]}, "difficulty_params": {"num_range": [1, 100], "steps": 1}, "tags": ["减法", "两位数"], "difficulty": 3}
</output>

题目: 小明有12个苹果，吃了3个，还剩几个？"""
    results["analyze"] = False
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content": PROMPT}], "temperature": 0.3, "max_tokens": 4096},
            timeout=60)
        d = r.json()
        msg = d["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        print(f"raw total: {len(raw)} chars")
        m = re.search(r"<output>\s*(.*?)\s*</output>", raw, re.DOTALL)
        if not m:
            print(f"No <output> tag. First 500 chars:\n{raw[:500]}")
            _summary(results)
            return
        inner = m.group(1).strip()
        print(f"<output> extracted ({len(inner)} chars): {inner[:200]}")
        inner_clean = inner
        if inner_clean.startswith("```"):
            inner_clean = inner_clean.split("\n", 1)[-1]
            if inner_clean.endswith("```"):
                inner_clean = inner_clean[:-3]
        parsed = json.loads(inner_clean)
        print("Parsed JSON OK")
        print(f"  subject:      {parsed.get('subject')}")
        print(f"  question_type:{parsed.get('question_type')}")
        print(f"  tags:         {parsed.get('tags')}")
        print(f"  difficulty:   {parsed.get('difficulty')}")
        results["analyze"] = (
            parsed.get("subject") in ("math", "chinese", "english")
            and isinstance(parsed.get("tags"), list)
            and isinstance(parsed.get("difficulty"), int)
        )
        print("OK" if results["analyze"] else "FAIL: fields mismatch")
    except Exception as e:
        print(f"FAIL: {e}")

    # ── Test 5: generate_derivative ──
    print("\n=== Test 5: Generate Derivative ===")
    DPROMPT = """基于以下题目，生成一道难度提升的衍生题，只把衍生题文本放在 <output> 标签内。

原题: 小明有12个苹果，吃了3个，还剩几个？
原题结构: {"operation": "subtraction", "operands": [12, 3]}
当前难度: 2 / 目标难度: 4
科目: math

要求：
- 数学：增大数值、增加计算步骤、或改变问法（正向→逆向）

输出格式示例：
<output>
小明有48颗糖，第一天吃了15颗，第二天吃了9颗，还剩几颗？
</output>"""
    results["derive"] = False
    try:
        r = httpx.post(URL,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"model": MODEL, "messages": [{"role":"user","content": DPROMPT}], "temperature": 0.3, "max_tokens": 4096},
            timeout=60)
        d = r.json()
        msg = d["choices"][0]["message"]
        raw = msg.get("content") or msg.get("reasoning_content", "")
        print(f"raw total: {len(raw)} chars")
        m = re.search(r"<output>\s*(.*?)\s*</output>", raw, re.DOTALL)
        if not m:
            print(f"No <output> tag. First 500 chars:\n{raw[:500]}")
        else:
            derived = m.group(1).strip()
            print(f"Derived ({len(derived)} chars): {derived[:200]}")
            results["derive"] = len(derived) > 5
            print("OK" if results["derive"] else "FAIL: too short")
    except Exception as e:
        print(f"FAIL: {e}")

    _summary(results)


def _summary(results):
    print("\n" + "=" * 40)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for k, v in results.items():
        print(f"  {'✅' if v else '❌'} {k}")
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ {total - passed} tests failed")


if __name__ == "__main__":
    main()
