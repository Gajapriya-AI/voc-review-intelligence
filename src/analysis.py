"""Review analysis utilities: transparent aspect sentiment + overall sentiment."""
from __future__ import annotations
import re
from collections import Counter
from typing import Any

ASPECTS = {
    "Battery": ["battery", "battery life", "battery backup", "charge", "charging", "drains", "drain", "power life"],
    "Camera": ["camera", "photo", "photos", "picture", "pictures", "image quality", "video quality", "selfie", "lens"],
    "Performance": ["performance", "speed", "slow", "lag", "lagging", "hang", "hanging", "processor", "runs smoothly", "responsive"],
    "Display": ["display", "screen", "brightness", "resolution", "touchscreen", "touch response"],
    "Build & Design": ["design", "looks", "build", "material", "finish", "fabric", "stitching", "fit", "comfort", "style", "appearance"],
    "Quality & Durability": ["quality", "durable", "durability", "broken", "broke", "damaged", "defective", "defect", "material", "sturdy"],
    "Delivery & Packaging": ["delivery", "delivered", "shipping", "packaging", "package", "arrived", "late", "courier"],
    "Price & Value": ["price", "cost", "value for money", "expensive", "cheap", "overpriced", "worth the money", "affordable"],
    "Customer Support": ["support", "customer service", "service team", "refund", "replacement", "warranty", "response from support"],
    "Heating & Noise": ["heating", "heat", "hot", "overheat", "overheats", "noise", "noisy", "loud", "vibration"],
    "Ease of Use": ["easy to use", "difficult to use", "setup", "installation", "instructions", "user friendly", "complicated"],
    "Cleaning & Maintenance": ["cleaning", "clean", "wash", "washing", "stain", "maintenance", "cleaning performance"],
    "Audio": ["sound", "audio", "speaker", "volume", "bass", "microphone", "mic", "headphones"],
    "Software & Connectivity": ["software", "update", "updates", "wifi", "wi-fi", "bluetooth", "connectivity", "connection", "app", "network"],
    "Fit & Sizing": ["size", "sizing", "fit", "too tight", "too loose", "small", "large", "measurement"],
    "Taste & Freshness": ["taste", "flavor", "flavour", "fresh", "freshness", "stale", "smell", "texture"],
}
POSITIVE = ["excellent", "excellent quality", "great", "good", "love", "loved", "amazing", "perfect", "smooth", "comfortable", "durable", "reliable", "beautiful", "clear", "bright", "easy to use", "works well", "value for money", "worth the money", "fresh", "tasty", "quiet", "helpful", "quick delivery", "on time", "long battery", "good battery", "lasts all day", "last all day", "long-lasting battery", "sharp photos", "high quality", "recommend", "satisfied", "sturdy", "responsive"]
NEGATIVE = ["poor", "bad", "terrible", "awful", "worst", "disappointing", "disappointed", "broken", "defective", "damaged", "slow", "lag", "lags", "hangs", "drains fast", "battery drains", "short battery", "overheats", "overheating", "gets hot", "very hot", "hot while charging", "too hot", "noisy", "loud", "uncomfortable", "cheap quality", "low quality", "not working", "does not work", "doesn't work", "never arrived", "late delivery", "poor packaging", "unhelpful", "rude", "expensive", "overpriced", "stale", "rotten", "bad smell", "wrong size", "too tight", "too loose", "not worth", "waste of money", "disconnects", "keeps disconnecting", "refund denied", "cracked", "leaking", "hard to use", "difficult to use"]
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "doesn't", "didn't", "hardly"}


def _phrase_polarity(text: str) -> tuple[int, int, list[str], list[str]]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    pos_hits, neg_hits = [], []
    for phrase in sorted(POSITIVE, key=len, reverse=True):
        if phrase in t:
            # Flip simple local negations, e.g. "not good".
            idx = t.find(phrase)
            prefix = t[max(0, idx - 12):idx].split()
            if any(w in NEGATIONS for w in prefix[-2:]):
                neg_hits.append("not " + phrase)
            else:
                pos_hits.append(phrase)
    for phrase in sorted(NEGATIVE, key=len, reverse=True):
        if phrase in t and phrase not in neg_hits:
            idx = t.find(phrase)
            prefix = t[max(0, idx - 12):idx].split()
            if any(w in NEGATIONS for w in prefix[-2:]) and phrase in {"poor", "bad", "slow", "disappointing", "uncomfortable"}:
                pos_hits.append("not " + phrase)
            else:
                neg_hits.append(phrase)
    # Avoid repeated overlapping hits distorting score too much.
    return min(len(pos_hits), 3), min(len(neg_hits), 3), list(dict.fromkeys(pos_hits)), list(dict.fromkeys(neg_hits))


def _contains_term(text: str, term: str) -> bool:
    # Match complete words/phrases to avoid false matches such as "app" inside "disappointing".
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) is not None


def _aspect_polarity(text: str, terms: list[str]) -> tuple[str, list[str], list[str]] | None:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    clauses = [c.strip() for c in re.split(r"[.!?;]+|,|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b", t) if c.strip()]
    relevant = [clause for clause in clauses if any(_contains_term(clause, term) for term in terms)]
    if not relevant:
        return None
    local_pos = local_neg = 0
    evidence_pos, evidence_neg = [], []
    for clause in relevant:
        p, n, ph, nh = _phrase_polarity(clause)
        local_pos += p
        local_neg += n
        evidence_pos.extend(ph)
        evidence_neg.extend(nh)
    if local_pos and local_neg:
        sentiment = "Mixed"
    elif local_neg:
        sentiment = "Negative"
    elif local_pos:
        sentiment = "Positive"
    else:
        sentiment = "Needs review"
    return sentiment, list(dict.fromkeys(evidence_pos))[:4], list(dict.fromkeys(evidence_neg))[:4]


def analyze_review(review: str, category: str = "Other Products", model: Any = None) -> dict:
    if not isinstance(review, str) or not review.strip():
        return {"review": review if isinstance(review, str) else "", "category": category, "sentiment": "Needs review", "confidence": None, "aspects": [], "positive_feedback": [], "pain_points": [], "business_insight": "Please enter a non-empty review."}
    text = review.strip()
    pos, neg, pos_hits, neg_hits = _phrase_polarity(text)
    ml_sentiment, ml_conf = None, None
    if model is not None:
        try:
            ml_sentiment, ml_conf = model.predict_sentiment(text)
        except Exception:
            pass
    # Aspect evidence is primary for mixed reviews; this prevents a positive overall prior from hiding complaints.
    aspects = []
    for name, terms in ASPECTS.items():
        result = _aspect_polarity(text, terms)
        if result:
            sentiment, p_evidence, n_evidence = result
            aspects.append({"aspect": name, "sentiment": sentiment, "positive_evidence": p_evidence, "negative_evidence": n_evidence})
    aspect_pos = sum(1 for a in aspects if a["sentiment"] == "Positive")
    aspect_neg = sum(1 for a in aspects if a["sentiment"] == "Negative")
    aspect_mixed = sum(1 for a in aspects if a["sentiment"] == "Mixed")
    if aspect_pos and (aspect_neg or aspect_mixed):
        overall = "Mixed"
    elif aspect_neg and not aspect_pos:
        overall = "Negative"
    elif aspect_pos and not aspect_neg:
        overall = "Positive"
    elif pos and neg:
        overall = "Mixed"
    elif neg > pos:
        overall = "Negative"
    elif pos > neg:
        overall = "Positive"
    elif ml_sentiment in {"Positive", "Negative", "Neutral"}:
        overall = "Needs review" if ml_conf is not None and ml_conf < 0.58 else ("Mixed" if ml_sentiment == "Neutral" and (pos or neg) else ml_sentiment)
    else:
        overall = "Needs review"
    positives = list(dict.fromkeys(pos_hits))[:6]
    pains = list(dict.fromkeys(neg_hits))[:6]
    for a in aspects:
        if a["sentiment"] == "Positive" and a["aspect"] not in positives:
            positives.append(a["aspect"])
        if a["sentiment"] in {"Negative", "Mixed"} and a["aspect"] not in pains:
            pains.append(a["aspect"])
    if overall == "Mixed":
        insight = "Customers mention both strengths and problems. Review the aspect-level findings before prioritizing improvements."
    elif overall == "Negative":
        insight = "The review contains negative feedback. Investigate the listed pain points and validate whether they recur across more reviews."
    elif overall == "Positive":
        insight = "The review contains positive feedback. Track the praised aspects to understand what customers value."
    else:
        insight = "The review does not contain enough clear evidence for a reliable sentiment label. Consider human review."
    return {"review": text, "category": category, "sentiment": overall, "confidence": round(float(ml_conf), 3) if ml_conf is not None else None, "aspects": aspects, "positive_feedback": positives, "pain_points": pains, "business_insight": insight}
