"""CoNLL-2003 JSON entity-extraction format, validator and micro-F1."""
from __future__ import annotations

import json
from collections import Counter

TYPES = ("PER", "ORG", "LOC", "MISC")
FAIL_CLASSES = ("truncated", "unterminated", "extra_prose", "malformed", "schema", "non_verbatim")

SYSTEM_PROMPT = (
    "You are a named-entity extraction system. Given a text, extract every named-entity mention in the order the "
    "mentions appear. Entity types: PER (person), ORG (organization), LOC (location), MISC (other names, e.g. "
    "nationalities, events, titles of works, products). Return strict JSON: a list of objects with keys text and "
    "type, one object per mention (an entity mentioned twice appears twice), where text is copied exactly from the "
    "input. Return [] if the text contains no named entities."
)

def user_prompt(text: str) -> str:
    return "Text:\n" + text

def messages_for(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(text)}]

def gold_response(mentions: list[dict]) -> str:
    return json.dumps([{"text": m["text"], "type": m["type"]} for m in mentions], ensure_ascii=False)

_BOUND = (" ", "\n")

def is_token_aligned(span: str, text: str) -> bool:
    if not span or span != span.strip() or "\n" in span:
        return False
    start = 0
    while True:
        i = text.find(span, start)
        if i < 0:
            return False
        j = i + len(span)
        if (i == 0 or text[i - 1] in _BOUND) and (j == len(text) or text[j] in _BOUND):
            return True
        start = i + 1

def is_token_aligned_prefix(partial: str, text: str) -> bool:
    pass
    if partial == "":
        return True
    if partial[0] in _BOUND or "\n" in partial:
        return False
    start = 0
    while True:
        i = text.find(partial, start)
        if i < 0:
            return False
        if i == 0 or text[i - 1] in _BOUND:
            return True
        start = i + 1

def _is_incomplete_array(s: str) -> bool:
    pass
    try:
        json.loads(s)
        return False
    except json.JSONDecodeError as e:
        if e.pos >= len(s.rstrip()):
            return True
        if e.msg.startswith("Unterminated string"):
            return True
        return False

def classify(pred_text: str, finish_reason: str | None, input_text: str):
    pass
    if finish_reason == "length":
        return "truncated", None, "hit token cap"
    s = pred_text.strip()
    parsed = None
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
            if _is_incomplete_array(s):
                return "unterminated", None, "incomplete JSON array"
            try:
                obj, end = json.JSONDecoder().raw_decode(s)
                if isinstance(obj, list) and s[end:].strip():
                    return "extra_prose", None, "text after the array"
            except json.JSONDecodeError:
                pass
            return "malformed", None, "json parse error"
    else:
        i = s.find("[")
        if i < 0:
            return "malformed", None, "no array"
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[i:])
            if isinstance(obj, list):
                return "extra_prose", None, "text before the array"
        except json.JSONDecodeError:
            pass
        return "malformed", None, "no parseable array"
    if not isinstance(parsed, list):
        return "schema", None, "top level is not a list"
    for k, o in enumerate(parsed):
        if not isinstance(o, dict) or set(o.keys()) != {"text", "type"}:
            return "schema", None, f"item {k} keys"
        if not isinstance(o["text"], str) or not o["text"] or not isinstance(o["type"], str):
            return "schema", None, f"item {k} value types"
        if o["type"] not in TYPES:
            return "schema", None, f"item {k} type {o['type']!r}"
    for k, o in enumerate(parsed):
        if not is_token_aligned(o["text"], input_text):
            return "non_verbatim", None, f"item {k} text {o['text']!r}"
    return "valid", parsed, ""

def pairs(lst):
    return [(o["text"], o["type"]) for o in lst]

def multiset_counts(pred, gold):
    pass
    cp, cg = Counter(pairs(pred)), Counter(pairs(gold))
    tp = sum(min(c, cg[k]) for k, c in cp.items())
    return tp, sum(cp.values()), sum(cg.values())

def set_counts(pred, gold):
    sp, sg = set(pairs(pred)), set(pairs(gold))
    return len(sp & sg), len(sp), len(sg)

def prf(tp, npred, ngold):
    p = tp / npred if npred else (1.0 if ngold == 0 else 0.0)
    r = tp / ngold if ngold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f

def lcs_align(pred, gold):
    pass
    a, b = pairs(pred), pairs(gold)
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out

def aligned_pointer(pred, text):
    pass
    ptr = 0
    for o in pred:
        span = o["text"]
        start = ptr
        while True:
            i = text.find(span, start)
            if i < 0:
                break
            j = i + len(span)
            if (i == 0 or text[i - 1] in _BOUND) and (j == len(text) or text[j] in _BOUND):
                ptr = j
                break
            start = i + 1
    return ptr

def gold_remaining(pred, gold, text):
    pass
    ptr = aligned_pointer(pred, text)
    return sum(1 for g in gold if g["start"] >= ptr)

def positional(pred, gold, text):
    pass
    al = lcs_align(pred, gold)
    mg = {j for _, j in al}
    last_p = max(i for i, _ in al) if al else -1
    m = len(gold)
    thirds = [(min(2, (3 * j) // m), int(j in mg)) for j in range(m)] if m else []
    return {"missing_tail": gold_remaining(pred, gold, text), "overrun": len(pred) - 1 - last_p,
            "n_aligned": len(al), "thirds": thirds}

S_START, S_A0, S_O0, S_KEY, S_COLON, S_V0, S_VS, S_AFTER_V, S_NEXTKEY, S_AFTER_OBJ, S_A1, S_END = range(12)
_WS = " \t\n\r"
_JSON_VALUE_START = '"-0123456789tfn[{'
_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

class Scan:
    __slots__ = ("state", "viol", "objs", "cur", "keys", "key", "buf", "esc", "uni")

    def __init__(self):
        self.state, self.viol, self.objs = S_START, None, []
        self.cur, self.keys, self.key, self.buf, self.esc, self.uni = {}, set(), "", "", False, None

def scan(out_text, input_text):
    pass
    sc = Scan()
    for ch in out_text:
        _step(sc, ch, input_text)
        if sc.viol:
            break
    return sc

def clone(sc):
    c = Scan()
    c.state, c.viol, c.objs = sc.state, sc.viol, list(sc.objs)
    c.cur, c.keys, c.key, c.buf, c.esc, c.uni = dict(sc.cur), set(sc.keys), sc.key, sc.buf, sc.esc, sc.uni
    return c

def advance(sc, more_text, input_text):
    pass
    for ch in more_text:
        if sc.viol:
            break
        _step(sc, ch, input_text)
    return sc

def _fail(sc, lab):
    sc.viol = lab

def _step(sc, ch, text):
    st = sc.state
    if st == S_START:
        if ch in _WS: return
        if ch == "[": sc.state = S_A0; return
        return _fail(sc, "extra_prose")
    if st in (S_A0, S_A1):
        if ch in _WS: return
        if ch == "{":
            sc.state, sc.cur, sc.keys = S_O0, {}, set(); return
        if ch == "]" and st == S_A0:
            sc.state = S_END; return
        if ch == "]":
            return _fail(sc, "malformed")
        return _fail(sc, "schema" if ch in _JSON_VALUE_START else "malformed")
    if st in (S_O0, S_NEXTKEY):
        if ch in _WS: return
        if ch == '"':
            sc.state, sc.key = S_KEY, ""; return
        if ch == "}" and st == S_O0:
            return _fail(sc, "schema")
        return _fail(sc, "malformed")
    if st == S_KEY:
        if ch == '"':
            if sc.key not in ("text", "type") or sc.key in sc.keys:
                return _fail(sc, "schema")
            sc.keys.add(sc.key); sc.state = S_COLON; return
        if ch == "\\" or ch < " ":
            return _fail(sc, "schema" if ch == "\\" else "malformed")
        sc.key += ch
        if not any(k.startswith(sc.key) for k in ("text", "type") if k not in sc.keys):
            return _fail(sc, "schema")
        return
    if st == S_COLON:
        if ch in _WS: return
        if ch == ":": sc.state = S_V0; return
        return _fail(sc, "malformed")
    if st == S_V0:
        if ch in _WS: return
        if ch == '"':
            sc.state, sc.buf, sc.esc, sc.uni = S_VS, "", False, None; return
        return _fail(sc, "schema" if ch in _JSON_VALUE_START else "malformed")
    if st == S_VS:
        if sc.uni is not None:
            if ch not in "0123456789abcdefABCDEF":
                return _fail(sc, "malformed")
            sc.uni += ch
            if len(sc.uni) == 4:
                c = chr(int(sc.uni, 16)); sc.uni = None
                return _add_char(sc, c, text)
            return
        if sc.esc:
            sc.esc = False
            if ch == "u":
                sc.uni = ""; return
            if ch not in _ESC:
                return _fail(sc, "malformed")
            return _add_char(sc, _ESC[ch], text)
        if ch == "\\":
            sc.esc = True; return
        if ch == '"':
            return _close_value(sc, text)
        if ch < " ":
            return _fail(sc, "malformed")
        return _add_char(sc, ch, text)
    if st == S_AFTER_V:
        if ch in _WS: return
        if ch == ",":
            sc.state = S_NEXTKEY; return
        if ch == "}":
            if sc.keys != {"text", "type"}:
                return _fail(sc, "schema")
            sc.objs.append(dict(sc.cur)); sc.state = S_AFTER_OBJ; return
        return _fail(sc, "malformed")
    if st == S_AFTER_OBJ:
        if ch in _WS: return
        if ch == ",": sc.state = S_A1; return
        if ch == "]": sc.state = S_END; return
        return _fail(sc, "malformed")
    if st == S_END:
        if ch in _WS: return
        return _fail(sc, "extra_prose")

def _add_char(sc, c, text):
    sc.buf += c
    if sc.key == "text":
        if not is_token_aligned_prefix(sc.buf, text):
            _fail(sc, "non_verbatim")
    else:
        if not any(t.startswith(sc.buf) for t in TYPES):
            _fail(sc, "schema")

def _close_value(sc, text):
    if sc.key == "text":
        if not sc.buf:
            return _fail(sc, "schema")
        if not is_token_aligned(sc.buf, text):
            return _fail(sc, "non_verbatim")
    elif sc.buf not in TYPES:
        return _fail(sc, "schema")
    sc.cur[sc.key] = sc.buf
    sc.state = S_AFTER_V

def completable(prefix_text, input_text):
    return scan(prefix_text, input_text).viol is None

def eos_violation(prefix_text, input_text):
    pass
    sc = scan(prefix_text, input_text)
    if sc.viol:
        return sc.viol
    return None if sc.state == S_END else "unterminated"

def is_close_step(before, after):
    pass
    if after.state == S_END and before.state != S_END:
        return True
    return after.state == S_AFTER_OBJ and before.state not in (S_AFTER_OBJ, S_END)

def first_violation(pred_text, finish_reason, input_text, gold=None):
    pass
    sc = scan(pred_text, input_text)
    if sc.viol:
        return sc.viol
    if sc.state != S_END:
        return "truncated" if finish_reason == "length" else "unterminated"
    if gold is not None and gold_remaining(sc.objs, gold, input_text) > 0:
        return "early_close"
    return None

def read_conll_docs(path):
    pass
    docs, doc, sent = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("-DOCSTART-"):
            if sent and doc is not None:
                doc.append(sent)
            sent = []
            doc = []
            docs.append(doc)
            continue
        if not line.strip():
            if sent:
                doc.append(sent)
            sent = []
            continue
        parts = line.split(" ")
        sent.append((parts[0], parts[-1]))
    if sent:
        doc.append(sent)
    return [d for d in docs if d]

def sentence_mentions(sent):
    pass
    out, cur = [], None
    for k, (_, tag) in enumerate(sent):
        if tag == "O":
            if cur:
                out.append(cur); cur = None
            continue
        pre, typ = tag.split("-", 1)
        if pre == "B" or cur is None or cur[2] != typ:
            if cur:
                out.append(cur)
            cur = [k, k + 1, typ]
        else:
            cur[1] = k + 1
    if cur:
        out.append(cur)
    return [tuple(x) for x in out]

def chunk_text_and_mentions(sents):
    pass
    lines, mentions, off = [], [], 0
    for s in sents:
        toks = [t for t, _ in s]
        starts, p = [], 0
        for t in toks:
            starts.append(p); p += len(t) + 1
        line = " ".join(toks)
        for a, b, typ in sentence_mentions(s):
            txt = " ".join(toks[a:b])
            mentions.append({"text": txt, "type": typ, "start": off + starts[a], "end": off + starts[a] + len(txt)})
        lines.append(line)
        off += len(line) + 1
    text = "\n".join(lines)
    for m in mentions:
        assert text[m["start"]:m["end"]] == m["text"]
    return text, mentions
