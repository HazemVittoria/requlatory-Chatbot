from __future__ import annotations

import re


# =========================
# Query intent detection
# =========================

def is_definition_query(query: str) -> bool:
    q = query.lower().strip()
    return any(
        q.startswith(p)
        for p in ["what is ", "what's ", "define ", "definition of ", "meaning of ", "explain "]
    ) or (" definition" in q)


def is_procedural_query(query: str) -> bool:
    q = query.lower().strip()
    if q.startswith(("what is", "what's", "define", "definition of", "meaning of", "explain")):
        return False
    return any(
        kw in q
        for kw in [
            "how to",
            "how should",
            "carry out",
            "perform",
            "conduct",
            "procedure",
            "sampling",
            "testing",
            "batch",
            "validate",
            "validation",
            "verify",
            "verification",
        ]
    )


def normalize_term(query: str) -> str:
    ql = query.strip().lower()
    patterns = [
        r"^\s*what\s+is\s+(an?\s+)?(.+?)\s*\??\s*$",
        r"^\s*what\'?s\s+(an?\s+)?(.+?)\s*\??\s*$",
        r"^\s*define\s+(.+?)\s*\??\s*$",
        r"^\s*definition\s+of\s+(.+?)\s*\??\s*$",
        r"^\s*meaning\s+of\s+(.+?)\s*\??\s*$",
        r"^\s*explain\s+(.+?)\s*\??\s*$",
    ]
    for pat in patterns:
        m = re.match(pat, ql, flags=re.IGNORECASE)
        if m:
            term = m.group(m.lastindex)
            term = re.sub(r"[\"“”']", "", term).strip()
            term = re.sub(r"\s+", " ", term)
            return term
    return re.sub(r"\s+", " ", ql).strip()


# =========================
# Cleanup helpers
# =========================

_line_number_re = re.compile(r"(?:(?<=\s)|^)\d{1,4}(?=\s)")


def prettify(text: str) -> str:
    text = text or ""
    text = _line_number_re.sub("", text)

    # remove URLs / web fragments
    text = re.sub(r"\bwww\.[^\s]+\b", "", text, flags=re.IGNORECASE)

    # remove common header/title lines
    text = re.sub(
        r"(Guideline\s*on\s*the\s*requirements.*?Rev\.?|EMA/CHMP/QWP/\d+/\d+\s*Rev\.?|Page\s+\d+/\d+)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # remove boilerplate
    text = re.sub(
        r"(All\s+rights\s+reserved.*?|Unauthorized\s+copying.*?|No\s+part\s+of\s+this\s+document.*?prohibited\.?)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"copyright\s*©?\s*.*?$", "", text, flags=re.IGNORECASE)

    # fix broken hyphenation
    text = re.sub(r"(\w+)\s*-\s*(\w+)", r"\1-\2", text)

    # add spaces after punctuation if missing
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

    # normalize whitespace
    text = text.replace("\u00a0", " ")
    return " ".join(text.split()).strip()


def is_truncated(s: str) -> bool:
    st = s.strip()
    bad_endings = ("(e.", "(e", "e.", "vs.", "vs", "(", "e.g", "e.g.,")
    if st.endswith(bad_endings):
        return True
    if re.search(r"\(e\.$", st):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = text.replace("e.g.,", "e_g_")
    parts = re.split(r"(?<=[.!?])\s+", text)

    out: list[str] = []
    for p in parts:
        p = p.replace("e_g_", "e.g.,").strip()
        if len(p) < 25:
            continue
        if is_truncated(p):
            continue
        out.append(p)
    return out


def dedup_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b\d+(\.\d+)*\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180]


def is_diagram_like(s: str) -> bool:
    """
    IMPORTANT: Do NOT treat a sentence as diagram-like if it clearly contains a definition pattern.
    This prevents dropping AEQ-style heading+definition lines.
    """
    sl = s.lower()
    if (" is the " in sl) or (" is a " in sl) or (" is an " in sl) or (" defined as " in sl) or (" refers to " in sl) or (" means " in sl):
        return False

    if any(k in sl for k in ["risk assessment", "risk control", "risk review", "risk acceptance", "risk reduction"]):
        if s.count(".") == 0 and s.count(";") == 0 and s.count(":") == 0:
            return True

    caps = sum(1 for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", s))
    words = re.findall(r"\b\w+\b", s)
    if len(words) >= 10 and (caps / max(1, len(words))) > 0.55:
        return True

    if len(words) > 14 and s.count(".") == 0 and s.count(",") == 0 and " and " not in sl:
        return True

    return False


# =========================
# Definition detection / scoring
# =========================

def looks_like_definition(sentence: str, term_hint: str) -> bool:
    s = sentence.strip()
    sl = s.lower()

    if any(bad in sl for bad in ["all rights reserved", "unauthorized copying"]):
        return False
    if is_diagram_like(s):
        return False

    term = (term_hint or "").lower().strip()
    has_term = bool(term) and (term in sl)

    if has_term and any(m in sl for m in [" is the ", " is a ", " is an ", " means ", " refers to ", " defined as "]):
        return True

    if any(m in sl for m in [" is the collection of ", " is the process of ", " is the documented evidence "]):
        return True

    if "systematic process" in sl and "risk" in sl:
        return True

    return False


def definition_sentence_score(sentence: str, query: str, term_hint: str, source: str) -> float:
    sl = sentence.lower()
    score = 0.0

    # Strongly prefer sentences that contain the term being defined
    if term_hint and term_hint.lower() in sl:
        score += 4.0
    else:
        score -= 1.5  # penalize definitions of other terms (e.g., “uncertainty”)

    if any(m in sl for m in ["defined as", "refers to", "means"]):
        score += 2.0
    if re.search(r"\bis (a|an|the)\b", sl):
        score += 1.0

    for kw, w in [
        ("documented evidence", 1.6),
        ("intended purpose", 1.6),
        ("systematic process", 1.3),
        ("assessment", 0.6),
        ("control", 0.6),
        ("communication", 0.6),
        ("review", 0.6),
        ("qualification", 0.5),
    ]:
        if kw in sl:
            score += w

    if source == "ICH":
        score += 0.5
    elif source == "EMA":
        score += 0.2

    if re.search(r"\bation is the\b", sl):
        score -= 2.0

    return score


def procedural_sentence_score(sentence: str, query: str, source: str) -> float:
    s = sentence.lower()
    q = query.lower()
    score = 0.0

    for kw in [
        "validate", "validation", "verify", "verification", "demonstrate", "suitability",
        "method", "analytical", "accuracy", "precision", "specificity", "linearity", "range", "robustness",
        "acceptance criteria", "report", "protocol"
    ]:
        if kw in s:
            score += 0.4

    if source == "FDA":
        score += 1.2  # reduced: avoid FDA sampling dominating method validation
    elif source == "EMA":
        score += 1.8
    elif source == "ICH":
        score += 0.6

    for term in re.findall(r"[a-z]{4,}", q):
        if term in s:
            score += 0.12

    return score


# =========================
# Core answer builder
# =========================

def build_answer_v2(
    query: str,
    results: list[tuple[float, dict]],
    max_def_sentences: int = 2,
    max_supporting_bullets: int = 3,
    max_actions: int = 6,
) -> tuple[str, list[str]]:
    procedural = is_procedural_query(query)
    definition = is_definition_query(query)

    rescored: list[tuple[float, dict]] = []
    for sim, item in results:
        boost = 0.0
        if procedural:
            if item["source"] == "EMA":
                boost = 0.20
            elif item["source"] == "ICH":
                boost = 0.10
        rescored.append((sim + boost, item))
    rescored.sort(key=lambda x: x[0], reverse=True)

    citations: list[str] = []
    cited_set = set()

    def cite(item: dict) -> None:
        cit = f"{item['source']} | {item['file']} | page {item['page']} | {item['chunk_id']}"
        if cit not in cited_set:
            cited_set.add(cit)
            citations.append(cit)

    term_hint = normalize_term(query)

    # -------------------------
    # Definition mode
    # -------------------------
    if definition:
        candidates: list[tuple[float, str, dict]] = []
        support: list[tuple[float, str, dict]] = []

        # look deeper than before so we can find the real Q9 definition sentence
        for sim, item in rescored[:200]:
            text = prettify(item["text"])
            for sent in split_sentences(text):
                s = sent.strip()
                if not s:
                    continue

                sl = s.lower()
                if sl.startswith("guideline on the requirements") or "ema/chmp/qwp" in sl:
                    continue
                if is_diagram_like(s):
                    continue

                if looks_like_definition(s, term_hint):
                    sc = definition_sentence_score(s, query, term_hint, item["source"]) + (sim * 0.6)
                    candidates.append((sc, s, item))
                else:
                    if any(k in sl for k in ["documented", "evidence", "intended purpose", "foundation", "systematic process"]):
                        support.append((sim * 0.4, s, item))

        candidates.sort(key=lambda x: x[0], reverse=True)
        support.sort(key=lambda x: x[0], reverse=True)

        def_sents: list[tuple[str, dict]] = []
        seen = set()
        for _, s, item in candidates:
            key = dedup_key(s)
            if key in seen:
                continue
            seen.add(key)
            def_sents.append((s, item))
            if len(def_sents) >= max_def_sentences:
                break

        # if still nothing, fallback but avoid returning a bare title
        if not def_sents and rescored:
            for _, item in rescored[:10]:
                txt = prettify(item["text"])
                if len(txt.split()) >= 6:  # not just a title
                    cite(item)
                    return txt, citations[:4]
            top_item = rescored[0][1]
            cite(top_item)
            return prettify(top_item["text"]), citations[:4]

        bullets: list[tuple[str, dict]] = []
        for _, s, item in support:
            key = dedup_key(s)
            if key in seen:
                continue
            seen.add(key)
            bullets.append((s, item))
            if len(bullets) >= max_supporting_bullets:
                break

        main_def = " ".join(ds for ds, _ in def_sents)
        for _, item in def_sents:
            cite(item)
        for _, item in bullets:
            cite(item)

        out = main_def
        if bullets:
            out += "\n\nSupporting points:\n" + "\n".join(f"- {b}" for b, _ in bullets)

        return out, citations[:4]

    # -------------------------
    # Procedural mode
    # -------------------------
    if procedural:
        actions: list[str] = []
        seen = set()

        action_verbs = (
            "collect", "provide", "verify", "test", "perform", "review", "retain", "document",
            "investigate", "report", "obtain", "record", "assess", "confirm", "ensure",
            "demonstrate", "validate"
        )

        for sim, item in rescored:
            text = prettify(item["text"])
            for sent in split_sentences(text):
                s = sent.strip()
                if not s:
                    continue
                if is_diagram_like(s):
                    continue
                sl = s.lower()

                if sl.startswith("guideline on the requirements") or "ema/chmp/qwp" in sl:
                    continue
                if sl.startswith("page ") and "batch" not in sl:
                    continue

                # If question is about validation, require validation language
                ql = query.lower()
                if "validat" in ql:
                    if not any(k in sl for k in ["validat", "validation", "demonstrate", "suitability", "accuracy", "precision", "specificity"]):
                        continue

                if not any(v in sl for v in action_verbs):
                    continue

                sc = procedural_sentence_score(s, query, item["source"]) + (sim * 0.6)
                if sc < 1.2:
                    continue

                key = dedup_key(s)
                if key in seen:
                    continue
                seen.add(key)

                actions.append(s)
                cite(item)

                if len(actions) >= max_actions:
                    break
            if len(actions) >= max_actions:
                break

        if not actions and rescored:
            top_item = rescored[0][1]
            cite(top_item)
            return prettify(top_item["text"]), citations[:4]

        answer = "Key actions (regulatory guidance):\n" + "\n".join(f"- {a}" for a in actions)
        return answer, citations[:4]

    # -------------------------
    # Hybrid / general mode
    # -------------------------
    best: list[tuple[float, str, dict]] = []
    for sim, item in rescored[:80]:
        text = prettify(item["text"])
        for sent in split_sentences(text):
            s = sent.strip()
            if not s:
                continue
            if is_diagram_like(s):
                continue

            q_terms = re.findall(r"[a-z]{4,}", query.lower())
            overlap = sum(1 for t in q_terms if t in s.lower())
            if overlap == 0:
                continue

            sc = (sim * 0.7) + (overlap * 0.25)
            best.append((sc, s, item))

    best.sort(key=lambda x: x[0], reverse=True)
    if not best and rescored:
        top_item = rescored[0][1]
        cite(top_item)
        return prettify(top_item["text"]), citations[:4]

    _, main_sent, main_item = best[0]
    cite(main_item)

    bullets: list[tuple[str, dict]] = []
    seen = {dedup_key(main_sent)}
    for _, s, item in best[1:]:
        key = dedup_key(s)
        if key in seen:
            continue
        seen.add(key)
        bullets.append((s, item))
        cite(item)
        if len(bullets) >= 3:
            break

    out = main_sent
    if bullets:
        out += "\n\nRelevant points:\n" + "\n".join(f"- {b}" for b, _ in bullets)

    return out, citations[:4]
