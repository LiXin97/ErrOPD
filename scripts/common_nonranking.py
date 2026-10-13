"""Shared rules for injecting and detecting the non-ranking errors."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_ner

T3_KINDS = ("shift", "swap", "type1")
T2_KINDS = ("neq", "lim2", "fence", "selall", "trail")
KINDS = {"t3": T3_KINDS, "t2": T2_KINDS}
TYPE1_FROM, TYPE1_TO = "PER", "MISC"
_WORD_START = re.compile(r"[A-Za-z0-9]")

def t3_next_token(text: str, end: int):
    pass
    if end >= len(text):
        return None
    if text[end] not in " \n":
        raise ValueError(f"offset {end} is not a token boundary")
    j = end + 1
    k = j
    while k < len(text) and text[k] not in " \n":
        k += 1
    return (text[j:k], j, k) if k > j else None

def t3_eligible(kind: str, row: dict):
    gold, text = row["gold"], row["text"]
    if kind == "shift":
        if not gold:
            return None
        nt = t3_next_token(text, gold[0]["end"])
        if nt is None:
            return None
        w, a, b = nt
        if not _WORD_START.match(w):
            return None
        if any(m["start"] < b and a < m["end"] for m in gold):
            return None
        if any(m["text"] == w for m in gold):
            return None
        return {"wrong_text": w, "right_text": gold[0]["text"]}
    if kind == "swap":
        if len(gold) < 2 or gold[0]["text"] == gold[1]["text"]:
            return None
        return {"first": gold[0]["text"], "second": gold[1]["text"]}
    if kind == "type1":
        if not gold or gold[0]["type"] != TYPE1_FROM:
            return None
        if any(m["text"] == gold[0]["text"] and m["type"] == TYPE1_TO for m in gold):
            return None
        return {"text": gold[0]["text"], "wrong_type": TYPE1_TO}
    raise ValueError(kind)

def t3_inject_list(kind: str, row: dict):
    pass
    e = t3_eligible(kind, row)
    if e is None:
        return None
    out = [{"text": m["text"], "type": m["type"]} for m in row["gold"]]
    if kind == "shift":
        out[0]["text"] = e["wrong_text"]
    elif kind == "swap":
        out[0], out[1] = out[1], out[0]
    elif kind == "type1":
        out[0]["type"] = TYPE1_TO
    return out

def t3_inject(kind: str, row: dict):
    lst = t3_inject_list(kind, row)
    return None if lst is None else json.dumps(lst, ensure_ascii=False)

def t3_objects(out_text: str, row: dict):
    pass
    return common_ner.scan(out_text, row["text"]).objs

def t3_classify(kind: str, out_text: str, row: dict, objs=None):
    e = t3_eligible(kind, row)
    if e is None:
        return {"eligible": False, "hit": False}
    if objs is None:
        objs = t3_objects(out_text, row)
    if kind == "shift":
        hit = bool(objs) and objs[0]["text"] == e["wrong_text"]
    elif kind == "swap":
        hit = len(objs) >= 2 and objs[0]["text"] == e["second"] and objs[1]["text"] == e["first"]
    else:
        hit = bool(objs) and objs[0]["text"] == e["text"] and objs[0]["type"] == e["wrong_type"]
    return {"eligible": True, "hit": hit}

def t3_side_labels(out_text: str, finish_reason: str | None, row: dict):
    pass
    cls, parsed, _ = common_ner.classify(out_text, finish_reason, row["text"])
    objs = t3_objects(out_text, row)
    gold = row["gold"]
    if not gold:
        first = "empty_gold"
    elif not objs:
        first = "none"
    elif objs[0]["text"] == gold[0]["text"]:
        first = "gold" if objs[0]["type"] == gold[0]["type"] else "gold_text"
    elif len(gold) > 1 and objs[0]["text"] == gold[1]["text"]:
        first = "second"
    else:
        first = "other"
    rec = {"cls": cls, "first_item": first, "at_cap": finish_reason == "length"}
    if cls == "valid":
        tp, npred, ngold = common_ner.multiset_counts(parsed[1:], gold[1:])
        rec.update(rest_tp=tp, rest_n_pred=npred, rest_n_gold=ngold)
    return rec

_LIT = ("str", "dq", "num")
_NEQ = ("!=", "<>")

def _t2_tokens(sql: str):
    import common_sql
    return [t for t in common_sql.lex(sql) if t.kind != "comment"]

def t2_first_eq_literal(toks):
    pass
    for i, t in enumerate(toks[:-1]):
        if t.kind == "op" and t.text in ("=", "==") and toks[i + 1].kind in _LIT:
            return i
    return None

def t2_has_negation(toks) -> bool:
    return any((t.kind == "op" and t.text in _NEQ) or (t.kind == "ident" and t.text.upper() in ("NOT", "EXCEPT"))
               for t in toks)

def t2_limit_values(toks):
    pass
    return [(i + 1, toks[i + 1].text) for i, t in enumerate(toks[:-1]) if t.kind == "ident" and t.text.upper() == "LIMIT"]

def t2_gold(row: dict) -> str:
    return row["metadata"]["gold_sql"] if "metadata" in row else row["gold_sql"]

FENCE = "```"
FENCE_OPEN = "```sql\n"
FENCE_CLOSE = "\n```"
_FENCE_TAG = re.compile(r"(?:sqlite|sql)(?![A-Za-z0-9_])", re.IGNORECASE)

def t2_fence_open(text: str) -> bool:
    pass
    return text.lstrip().startswith(FENCE)

def t2_fence_closed(text: str) -> bool:
    pass
    s = text.lstrip()
    return s.startswith(FENCE) and FENCE in s[len(FENCE):]

def t2_unfence(text: str) -> str:
    pass
    s = text.lstrip()
    if not s.startswith(FENCE):
        return text
    body = s[len(FENCE):]
    m = _FENCE_TAG.match(body)
    if m:
        body = body[m.end():]
    body = body.rstrip()
    if body.endswith(FENCE):
        body = body[: -len(FENCE)]
    return body.strip()

def _t2_first_select(toks):
    for i, t in enumerate(toks):
        if t.kind == "ident" and t.text.upper() == "SELECT":
            return i
    return None

def _t2_is_kw(t, *words) -> bool:
    return t.kind == "ident" and t.text.upper() in words

def t2_select_all_first(toks) -> bool:
    pass
    i = _t2_first_select(toks)
    return i is not None and i + 1 < len(toks) and _t2_is_kw(toks[i + 1], "ALL")

def t2_select_all_any(toks) -> bool:
    return any(_t2_is_kw(t, "SELECT") and _t2_is_kw(toks[i + 1], "ALL") for i, t in enumerate(toks[:-1]))

TRAILER = "This query returns the requested information."
_ALNUM = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_CLAUSE_KW = ("FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "JOIN", "LEFT", "INNER", "ON", "AND", "OR",
              "UNION", "INTERSECT", "EXCEPT", "AS")

def t2_top_newline(text: str):
    pass
    s = text.lstrip()
    n, i, depth = len(s), 0, 0
    while i < n:
        c = s[i]
        if c in "'\"`[":
            close = "]" if c == "[" else c
            j = i + 1
            while j < n:
                if s[j] == close:
                    if close != "]" and j + 1 < n and s[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if s.startswith("--", i):
            j = s.find("\n", i)
            if j < 0:
                return None, None
            i = j
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "\n" and depth == 0:
            return s[:i], s[i + 1:]
        i += 1
    return None, None

INCOMPLETE_END = frozenset(("SELECT", "DISTINCT", "ALL", "FROM", "WHERE", "GROUP", "ORDER", "BY", "HAVING", "LIMIT",
                            "OFFSET", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING",
                            "AND", "OR", "NOT", "AS", "IN", "LIKE", "GLOB", "BETWEEN", "IS", "CASE", "WHEN", "THEN",
                            "ELSE", "WITH", "UNION", "INTERSECT", "EXCEPT", "ESCAPE", "EXISTS"))
CONTINUE_START = frozenset(("FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT", "RIGHT",
                            "INNER", "OUTER", "CROSS", "NATURAL", "ON", "USING", "AND", "OR", "AS", "UNION",
                            "INTERSECT", "EXCEPT", "ASC", "DESC"))

def t2_top_newlines(text: str):
    pass
    s = text.lstrip()
    n, i, depth, out = len(s), 0, 0, []
    while i < n:
        c = s[i]
        if c in "'\"`[":
            close = "]" if c == "[" else c
            j = i + 1
            while j < n:
                if s[j] == close:
                    if close != "]" and j + 1 < n and s[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            i = j + 1
            continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if s.startswith("--", i):
            j = s.find("\n", i)
            if j < 0:
                break
            i = j
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "\n" and depth == 0:
            out.append(i)
        i += 1
    return out, s

def t2_newline_kind(head: str, rest: str) -> str:
    pass
    rt = _t2_tokens(rest)
    if not rt:
        return "empty"
    ht = _t2_tokens(head)
    last = ht[-1] if ht else None
    if last is not None and ((last.kind == "op" and last.text not in (")", ";"))
                             or (last.kind == "ident" and last.text.upper() in INCOMPLETE_END)):
        return "continuation"
    first = rt[0]
    if (first.kind == "op" and first.text == ",") or (
            first.kind == "ident" and first.text.upper() in CONTINUE_START
            and (first.text.isupper() or first.text.islower())):
        return "continuation"
    return "extra"

def _t2_trail_kinds(text: str):
    nls, s = t2_top_newlines(text)
    return nls, s, [t2_newline_kind(s[:i], s[i + 1:]) for i in nls]

def t2_trail_hit(text: str) -> bool:
    return "extra" in _t2_trail_kinds(text)[2]

def t2_untrail(text: str) -> str:
    pass
    head, _ = t2_top_newline(text)
    return text if head is None else head.rstrip()

def t2_trail_labels(text: str) -> dict:
    head, rest = t2_top_newline(text)
    stmt = (text.strip() if head is None else head.rstrip())
    end = "empty" if not stmt else ("alnum" if stmt[-1] in _ALNUM else stmt[-1])
    r = (rest or "").lstrip()
    w = r.split(None, 1)[0].upper() if r else ""
    nls, s, kinds = _t2_trail_kinds(text)
    hit_idx = next((k for k, x in enumerate(kinds) if x == "extra"), None)
    after = s[nls[hit_idx] + 1:].lstrip() if hit_idx is not None else ""
    at = _t2_tokens(after) if after else []
    return {"trail_any": rest is not None and rest.strip() != "", "newline_at_end": rest is not None,
            "continuation": "continuation" in kinds,
            "trail_exact": after.startswith(TRAILER),
            "trail_this": bool(at) and at[0].text.upper() == "THIS",
            "second_query": bool(at) and at[0].kind == "ident" and at[0].text.upper() in ("SELECT", "WITH"),
            "continuation_kw": w in _CLAUSE_KW, "own_end": end}

def t2_eligible(kind: str, row: dict):
    g = t2_gold(row)
    if kind == "trail":
        if not g or g[-1] not in _ALNUM or "\n" in g:
            return None
        return {"end": len(g)}
    if kind == "fence":
        if g.strip()[:6].upper() != "SELECT" or FENCE in g:
            return None
        return {"tok": 0}
    toks = _t2_tokens(g)
    if kind == "selall":
        if len(toks) < 2 or not _t2_is_kw(toks[0], "SELECT") or _t2_is_kw(toks[1], "DISTINCT", "ALL"):
            return None
        return {"tok": 0, "start": toks[0].start, "end": toks[0].end}
    if kind == "neq":
        i = t2_first_eq_literal(toks)
        if i is None or t2_has_negation(toks):
            return None
        return {"tok": i, "start": toks[i].start, "end": toks[i].end}
    if kind == "lim2":
        lv = t2_limit_values(toks)
        if len(lv) != 1 or toks[lv[0][0]].kind != "num" or lv[0][1] != "1":
            return None
        i = lv[0][0]
        return {"tok": i, "start": toks[i].start, "end": toks[i].end}
    raise ValueError(kind)

def t2_inject(kind: str, row: dict):
    pass
    e = t2_eligible(kind, row)
    if e is None:
        return None
    g = t2_gold(row)
    if kind == "fence":
        return FENCE_OPEN + g + FENCE_CLOSE
    if kind == "trail":
        return g + "\n" + TRAILER
    if kind == "selall":
        return g[: e["end"]] + " ALL" + g[e["end"]:]
    return g[: e["start"]] + ("!=" if kind == "neq" else "2") + g[e["end"]:]

def t2_classify(kind: str, out_text: str, row: dict):
    if t2_eligible(kind, row) is None:
        return {"eligible": False, "hit": False}
    if kind == "fence":
        return {"eligible": True, "hit": t2_fence_open(out_text)}
    if kind == "trail":
        return {"eligible": True, "hit": t2_trail_hit(out_text)}
    if kind == "selall":
        return {"eligible": True, "hit": t2_select_all_first(_t2_tokens(t2_unfence(out_text)))}
    toks = _t2_tokens(out_text)
    if kind == "neq":
        hit = any(t.kind == "op" and t.text in _NEQ and toks[i + 1].kind in _LIT for i, t in enumerate(toks[:-1]))
    else:
        hit = any(toks[i].kind == "num" and v == "2" for i, v in t2_limit_values(toks))
    return {"eligible": True, "hit": hit}

def t2_side_labels(out_text: str, row: dict):
    pass
    toks = _t2_tokens(out_text)
    neq_any = [i for i, t in enumerate(toks) if t.kind == "op" and t.text in _NEQ]
    neq_lit = [i for i in neq_any if i + 1 < len(toks) and toks[i + 1].kind in _LIT]
    utoks = _t2_tokens(t2_unfence(out_text))
    return {"neq_any": bool(neq_any), "neq_nonliteral": len(neq_any) > len(neq_lit),
            "limit_values": [v for _, v in t2_limit_values(toks)],
            "fence_open": t2_fence_open(out_text), "fence_closed": t2_fence_closed(out_text),
            "fence_any": FENCE in out_text, "select_all": t2_select_all_first(utoks),
            "select_all_any": t2_select_all_any(utoks), **t2_trail_labels(out_text)}

def eligible(task: str, kind: str, row: dict):
    return (t3_eligible if task == "t3" else t2_eligible)(kind, row)

def inject(task: str, kind: str, row: dict):
    return (t3_inject if task == "t3" else t2_inject)(kind, row)

def classify(task: str, kind: str, out_text: str, row: dict):
    return (t3_classify if task == "t3" else t2_classify)(kind, out_text, row)
