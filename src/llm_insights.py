"""Optional local Small LLM insights via Ollama. App remains usable without it."""
import json
import os
import urllib.request
import urllib.error


def ollama_available() -> bool:
    return bool(os.getenv("OLLAMA_MODEL", "").strip())


def generate_insight(result: dict) -> tuple[str, str]:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        return result.get("business_insight", ""), "Rule-based summary (LLM not configured)"
    prompt = (
        "You are a careful customer-voice analyst. Use only the evidence below. Do not invent facts. "
        "In 2-3 short sentences, summarize customer sentiment and give one actionable recommendation. "
        "If evidence is unclear, say so.\n\n" + json.dumps({
            "category": result.get("category"), "review": result.get("review"), "sentiment": result.get("sentiment"),
            "aspects": result.get("aspects"), "positive_feedback": result.get("positive_feedback"), "pain_points": result.get("pain_points")
        }, ensure_ascii=False)
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}).encode()
    req = urllib.request.Request(os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = str(data.get("response", "")).strip()
        if text:
            return text, f"Local LLM ({model})"
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass
    return result.get("business_insight", ""), "Rule-based fallback (local LLM unavailable)"
