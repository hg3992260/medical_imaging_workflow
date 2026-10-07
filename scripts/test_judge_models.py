"""Test which Ollama model produces valid review JSON for the judge module."""
import os, sys, json, time, re
sys.path.insert(0, "F:/RSNA/medical_imaging_workflow")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"

MODELS = ["qwen2.5:7b", "qwen3:8b", "gemma4:e4b", "llama3:8b", "deepseek-r1:8b"]
BASE_URL = "http://localhost:11434"
TIMEOUT = 120
MAX_TOKENS = 512

PROMPT = """You are a strict peer reviewer for a medical imaging paper.
Return ONLY JSON with keys:
- score (number 0-10)
- passed (bool)
- issues (list[string], max 10)
- missing_data (list[string], max 10)
- hallucination_flags (list[string], max 10)
- suggested_edits (list[string], max 10)

[DRAFT]
This retrospective study included 120 patients with chest CT scans. The mean attenuation was 45.2 HU (SD 12.3). ROI analysis revealed an area of 156.78 mm2 with pixel count 1560."""

def test_model(model_name):
    import urllib.request, urllib.error
    payload = json.dumps({
        "model": model_name,
        "prompt": PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS},
        "format": "json",
    }).encode("utf-8")
    t0 = time.time()
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        response_text = body.get("response", "")
        if not response_text.strip():
            return {"ok": False, "error": "empty response", "time": elapsed}

        # Try JSON parse
        try:
            obj = json.loads(response_text)
            if isinstance(obj, dict):
                sc = obj.get("score", "?")
                passed = obj.get("passed", "?")
                issues = len(obj.get("issues", []))
                return {"ok": True, "score": sc, "passed": passed, "issues_count": issues, "time": elapsed, "raw": response_text[:200]}
        except Exception:
            pass
        # Try brace extraction
        idx = response_text.find("{")
        if idx >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(idx, len(response_text)):
                ch = response_text[i]
                if esc: esc = False; continue
                if ch == "\\": esc = True; continue
                if ch == '"': in_str = not in_str; continue
                if in_str: continue
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(response_text[idx:i+1])
                            if isinstance(obj, dict):
                                return {"ok": True, "score": obj.get("score", "?"), "passed": obj.get("passed", "?"), "issues_count": len(obj.get("issues", [])), "time": elapsed, "raw": response_text[:200]}
                        except:
                            pass
                        break
        # Try key:value
        patterns = ["score", "passed"]
        found = any(re.search(rf"\b{p}\b", response_text, re.IGNORECASE) for p in patterns)
        return {"ok": False, "error": "unparseable", "has_keywords": found, "time": elapsed, "raw": response_text[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e), "time": time.time() - t0}


if __name__ == "__main__":
    print("Testing judge models against sample draft...\n")
    results = []
    for m in MODELS:
        print(f"Testing {m:20s} ... ", end="", flush=True)
        r = test_model(m)
        status = "OK" if r.get("ok") else f"FAIL ({r.get('error','?')})"
        print(f"{status} in {r.get('time',0):.1f}s")
        if r.get("ok"):
            print(f"  score={r['score']} passed={r['passed']} issues={r['issues_count']}")
        results.append((m, r))

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    best = None
    for m, r in results:
        if r.get("ok"):
            best = m
            break
    if best:
        print(f"  Use: {best}")
        print(f"  Set: RSNA_JUDGE_MODEL={best}")
    else:
        # Pick model that at least has keywords
        for m, r in results:
            if r.get("has_keywords"):
                best = m
                break
        print(f"  Best available: {best or 'NONE'} (fallback mode)")
    print()
    for m, r in results:
        print(f"  {m:20s} raw: {r.get('raw','')[:100]}")
