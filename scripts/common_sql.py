#!/usr/bin/env python3
"""Spider SQL execution (sqlite), execution accuracy and strict output validity."""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LX_ROOT = "/path/to"

SYSTEM_PROMPT = ("You are a text-to-SQL system. Given the schema of a SQLite database and a question, write one SQLite "
                 "query that answers the question. Output only the SQL query on a single line: no explanation, no "
                 "code fences, no text before or after the query.")

PROGRESS_N = 1000
OP_BUDGET_TICKS = 100_000
WALL_CEILING_S = 30.0
ROW_CAP = 200_000

SQLITE_KEYWORDS = set("""ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN
BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT CURRENT_DATE
CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DO DROP EACH ELSE END
ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS EXPLAIN FAIL FILTER FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP
GROUPS HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LAST
LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF OFFSET ON OR ORDER OTHERS OUTER OVER
PARTITION PLAN PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE
RESTRICT RETURNING RIGHT ROLLBACK ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER
UNBOUNDED UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT""".split())
SPECIAL_NAMES = {"rowid", "oid", "_rowid_", "true", "false"}
CLAUSE_END = {"WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "INTERSECT", "EXCEPT", "ON", "USING", "WINDOW",
              "SELECT"}
TABLE_POS_KW = {"FROM", "JOIN"}

LABELS_TOKEN = ["prose_or_markup", "extra_after_sql", "syntax", "unknown_table", "unknown_column", "wrong_table_column",
                "unknown_function"]
LABELS_EOS = ["early_stop", "ambiguous_column", "semantic_other", "timeout", "empty"]
LABELS = LABELS_TOKEN + LABELS_EOS + ["truncated"]

_ALLOWED_ACTIONS = {20, 21, 31, 33}

_PLAIN_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _qid(name: str) -> str:
    return name if _PLAIN_ID.match(name) and name.upper() not in SQLITE_KEYWORDS else '"' + name.replace('"', '""') + '"'

def schema_ddl(db_path: str) -> str:
    pass
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    try:
        tabs = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        out = []
        for t in tabs:
            cols = con.execute(f"PRAGMA table_info({_qid(t)})").fetchall()
            fks = con.execute(f"PRAGMA foreign_key_list({_qid(t)})").fetchall()
            lines = []
            for _cid, name, ctype, _nn, _d, _pk in cols:
                ctype = (ctype or "").strip()
                lines.append(f"  {_qid(name)}{(' ' + ctype) if ctype else ''}")
            pks = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5] > 0]
            if pks:
                lines.append(f"  PRIMARY KEY ({', '.join(_qid(p) for p in pks)})")
            by_id = defaultdict(list)
            for fk in fks:
                by_id[fk[0]].append(fk)
            for _i, grp in sorted(by_id.items()):
                grp = sorted(grp, key=lambda f: f[1])
                src = ", ".join(_qid(f[3]) for f in grp)
                dst_t = grp[0][2]
                dst = [f[4] for f in grp]
                ref = f"{_qid(dst_t)}" + (f" ({', '.join(_qid(d) for d in dst)})" if all(dst) else "")
                lines.append(f"  FOREIGN KEY ({src}) REFERENCES {ref}")
            out.append(f"CREATE TABLE {_qid(t)} (\n" + ",\n".join(lines) + "\n);")
        return "\n".join(out)
    finally:
        con.close()

def user_prompt(ddl: str, question: str) -> str:
    return f"Database schema:\n{ddl}\n\nQuestion: {question.strip()}"

def build_messages(ddl: str, question: str, gold: str | None = None) -> list[dict]:
    m = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(ddl, question)}]
    if gold is not None:
        m.append({"role": "assistant", "content": gold})
    return m

def normalize_sql(q: str) -> str:
    pass
    out, i, n = [], 0, len(q)
    while i < n:
        c = q[i]
        if c in "'\"`[":
            close = "]" if c == "[" else c
            j = i + 1
            while j < n:
                if q[j] == close:
                    if close != "]" and j + 1 < n and q[j + 1] == close:
                        j += 2
                        continue
                    break
                j += 1
            out.append(q[i:j + 1])
            i = j + 1
        elif c.isspace():
            j = i
            while j < n and q[j].isspace():
                j += 1
            out.append(" ")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out).strip()

def resolve_db_path(rel_or_abs: str) -> str:
    pass
    p = rel_or_abs if os.path.isabs(rel_or_abs) else str(REPO / rel_or_abs)
    db_root = str(REPO / "data" / "erropd_t2" / "db")
    if p.startswith(db_root + "/"):
        key = db_root[len(LX_ROOT) + 1:].replace("/", "__") if db_root.startswith(LX_ROOT + "/") else None
        stage = os.environ.get("ERROPD_STAGE_DIR", "/tmp/erropd/stage")
        if key and os.path.exists(os.path.join(stage, key, ".stage_complete")):
            return os.path.join(stage, key, p[len(db_root) + 1:])
    return p

class DB:
    pass
    _cache: dict[str, "DB"] = {}

    @classmethod
    def get(cls, path: str) -> "DB":
        path = resolve_db_path(path)
        if path not in cls._cache:
            cls._cache[path] = DB(path)
        return cls._cache[path]

    def __init__(self, path: str):
        self.path = path
        self.con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        c = self.con
        self.tables = {r[0].lower() for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")}
        self.cols: dict[str, set] = {}
        for t in self.tables:
            try:
                self.cols[t] = {r[1].lower() for r in c.execute(f"PRAGMA table_info({_qid(t)})")}
            except sqlite3.Error:
                self.cols[t] = set()
        self.allcols = set().union(*self.cols.values()) if self.cols else set()
        try:
            self.funcs = {r[0].lower() for r in c.execute("SELECT name FROM pragma_function_list")}
        except sqlite3.Error:
            self.funcs = set()
        c.set_authorizer(lambda action, *a: 0 if action in _ALLOWED_ACTIONS else 1)
        self._memo: dict[str, str | None] = {}

    def execute(self, sql: str):
        pass
        ticks = [0]
        t0 = time.monotonic()

        def handler():
            ticks[0] += 1
            return 1 if (ticks[0] > OP_BUDGET_TICKS or time.monotonic() - t0 > WALL_CEILING_S) else 0

        self.con.set_progress_handler(handler, PROGRESS_N)
        self.last_ticks = ticks
        try:
            cur = self.con.execute(sql)
            rows, trunc = [], False
            while True:
                chunk = cur.fetchmany(10000)
                if not chunk:
                    break
                rows.extend(chunk)
                if len(rows) > ROW_CAP:
                    trunc = True
                    break
            return "ok", rows, trunc
        except sqlite3.Error as e:
            msg = str(e)
            if msg == "interrupted":
                return "timeout", msg, False
            return "error", msg, False
        except Exception as e:
            return "error", f"{type(e).__name__}: {e}", False
        finally:
            self.con.set_progress_handler(None, 0)

    def explain(self, sql: str) -> str | None:
        pass
        try:
            self.con.execute("EXPLAIN " + sql).fetchmany(1)
            return None
        except Exception as e:
            return str(e)

class Tok:
    __slots__ = ("kind", "text", "start", "end", "closed")

    def __init__(self, kind, text, start, end, closed):
        self.kind, self.text, self.start, self.end, self.closed = kind, text, start, end, closed

    def __repr__(self):
        return f"Tok({self.kind},{self.text!r},{self.closed})"

_OPS2 = {"<=", ">=", "<>", "!=", "==", "||", "<<", ">>", "->"}

def _is_id_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_$" or ord(ch) > 127

def lex(s: str) -> list[Tok]:
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if s.startswith("--", i):
            j = s.find("\n", i)
            j = n if j < 0 else j
            toks.append(Tok("comment", s[i:j], i, j, j < n))
            i = j
            continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            if j < 0:
                toks.append(Tok("comment", s[i:], i, n, False))
                break
            toks.append(Tok("comment", s[i:j + 2], i, j + 2, True))
            i = j + 2
            continue
        if c in "'\"`[":
            close = "]" if c == "[" else c
            kind = {"'": "str", '"': "dq", "`": "qident", "[": "qident"}[c]
            j = i + 1
            closed = False
            while j < n:
                if s[j] == close:
                    if close != "]" and j + 1 < n and s[j + 1] == close:
                        j += 2
                        continue
                    closed = True
                    break
                j += 1
            end = j + 1 if closed else n
            toks.append(Tok(kind, s[i:end], i, end, closed))
            i = end
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            m = re.compile(r"(?:0[xX][0-9A-Fa-f]*|\d*\.?\d*(?:[eE][+-]?\d*)?)").match(s, i)
            j = max(m.end(), i + 1)
            toks.append(Tok("num", s[i:j], i, j, j < n))
            i = j
            continue
        if _is_id_char(c):
            j = i
            while j < n and _is_id_char(s[j]):
                j += 1
            toks.append(Tok("ident", s[i:j], i, j, j < n))
            i = j
            continue
        if s[i:i + 2] in _OPS2:
            toks.append(Tok("op", s[i:i + 2], i, i + 2, True))
            i += 2
            continue
        toks.append(Tok("op", c, i, i + 1, not (i + 1 == n and c in "<>!|=-")))
        i += 1
    return toks

def _unq(t: Tok) -> str:
    x = t.text
    if t.kind == "qident":
        x = x[1:-1] if t.closed else x[1:]
    return x.lower()

def _is_kw(t: Tok) -> bool:
    return t.kind == "ident" and t.text.upper() in SQLITE_KEYWORDS

def _surface_ok(s: str) -> bool:
    u = s[:6].upper()
    return any(u.startswith(k) or k.startswith(u) for k in ("SELECT", "WITH"))

def _name_violation(toks: list[Tok], db: DB, n_text: int) -> str | None:
    sig = [t for t in toks if t.kind != "comment"]
    if not sig:
        return None
    aliases, ctes = set(), set()
    bind: dict[str, set] = defaultdict(set)
    depth, in_from = 0, {0: False}
    table_pos = [False] * len(sig)
    for k, t in enumerate(sig):
        prev = sig[k - 1] if k else None
        nxt = sig[k + 1] if k + 1 < len(sig) else None
        if t.kind == "op" and t.text == "(":
            depth += 1
            in_from[depth] = False
            continue
        if t.kind == "op" and t.text == ")":
            in_from[depth] = False
            depth = max(0, depth - 1)
            continue
        if _is_kw(t):
            u = t.text.upper()
            if u == "FROM":
                in_from[depth] = True
            elif u in CLAUSE_END:
                in_from[depth] = False
            continue
        if t.kind not in ("ident", "qident"):
            continue
        name = _unq(t)
        if prev is not None and ((prev.kind == "ident" and prev.text.upper() in TABLE_POS_KW) or
                                 (prev.kind == "op" and prev.text == "," and in_from.get(depth))):
            table_pos[k] = True
            bind[name].add(name)
            a = None
            if nxt is not None and nxt.kind == "ident" and nxt.text.upper() == "AS" and k + 2 < len(sig):
                a = sig[k + 2]
            elif nxt is not None and nxt.kind in ("ident", "qident") and not _is_kw(nxt):
                a = nxt
            if a is not None and a.kind in ("ident", "qident"):
                bind[_unq(a)].add(name)
        if prev is not None and prev.kind == "ident" and prev.text.upper() == "AS":
            aliases.add(name)
            if nxt is not None and nxt.kind == "op" and nxt.text == "(":
                pass
            if k >= 2 and sig[k - 2].kind == "op" and sig[k - 2].text == ")":
                bind[name].add("__sub__")
        if nxt is not None and nxt.kind == "ident" and nxt.text.upper() == "AS" and k + 2 < len(sig) \
                and sig[k + 2].kind == "op" and sig[k + 2].text == "(":
            ctes.add(name)
            bind[name].add("__sub__")
        if prev is not None and not table_pos[k] and (
                (prev.kind in ("ident", "qident") and not _is_kw(prev)) or prev.kind in ("dq", "str", "num")
                or (prev.kind == "op" and prev.text == ")")):
            aliases.add(name)
            if prev.kind == "op" and prev.text == ")" and in_from.get(depth):
                bind[name].add("__sub__")
    for k, t in enumerate(sig):
        if t.kind not in ("ident", "qident"):
            continue
        if t.kind == "ident" and _is_kw(t):
            continue
        prev = sig[k - 1] if k else None
        nxt = sig[k + 1] if k + 1 < len(sig) else None
        partial = (not t.closed) or (t.end == n_text)
        name = _unq(t)
        if not name:
            continue

        def ok_in(valid: set) -> bool:
            if partial:
                return any(v.startswith(name) for v in valid)
            return name in valid

        if nxt is not None and nxt.kind == "op" and nxt.text == "(":
            if name in db.funcs or name.upper() in SQLITE_KEYWORDS or name in ctes:
                continue
            if nxt.kind == "op":
                prev_as = prev is not None and prev.kind == "ident" and prev.text.upper() == "AS"
                if not prev_as and not table_pos[k]:
                    return "unknown_function"
            continue
        if nxt is not None and nxt.kind == "op" and nxt.text == ".":
            continue
        if prev is not None and prev.kind == "op" and prev.text == ".":
            q = _unq(sig[k - 2]) if k >= 2 and sig[k - 2].kind in ("ident", "qident") else None
            targets = bind.get(q, set()) if q else set()
            if targets and "__sub__" not in targets and all(x in db.cols for x in targets):
                valid = set().union(*(db.cols[x] for x in targets)) | SPECIAL_NAMES
                if not ok_in(valid):
                    return "wrong_table_column" if ok_in(db.allcols) else "unknown_column"
                continue
            elif targets:
                continue
            valid = db.allcols | aliases | SPECIAL_NAMES
            if not ok_in(valid):
                return "unknown_column"
            continue
        if table_pos[k] or (partial and prev is not None and (
                (prev.kind == "ident" and prev.text.upper() in TABLE_POS_KW))):
            if not ok_in(db.tables | ctes):
                return "unknown_table"
            continue
        if prev is not None and prev.kind == "ident" and prev.text.upper() == "AS":
            continue
        if name in aliases and not partial:
            continue
        if prev is not None and not partial and (
                (prev.kind in ("ident", "qident") and not _is_kw(prev)) or prev.kind in ("dq", "str", "num")
                or (prev.kind == "op" and prev.text == ")")):
            continue
        if partial or nxt is None:
            continue
        valid = db.allcols | db.tables | aliases | ctes | SPECIAL_NAMES
        if not ok_in(valid):
            return "unknown_column"
    return None

def analyze_prefix(prefix: str, db: DB) -> str | None:
    pass
    memo = db._memo
    if prefix in memo:
        return memo[prefix]
    res = _analyze(prefix, db)
    if len(memo) > 500_000:
        memo.clear()
    memo[prefix] = res
    return res

def _analyze(prefix: str, db: DB) -> str | None:
    s = prefix.lstrip()
    if not s:
        return None
    if not _surface_ok(s):
        return "prose_or_markup"
    fence = s.find("``")
    if fence >= 0:
        head = s[:fence].rstrip().rstrip(";").rstrip()
        return "extra_after_sql" if head and db.explain(head) is None else "prose_or_markup"
    toks = lex(s)
    for k, t in enumerate(toks):
        if t.kind == "op" and t.text == ";":
            if any(x.kind != "comment" for x in toks[k + 1:]):
                return "extra_after_sql"
            s = s[:t.end]
            toks = toks[:k + 1]
            break
    if len(s) < 6 and not s.upper().startswith("WITH"):
        return None
    last = toks[-1] if toks else None
    open_last = last is not None and (not last.closed or last.end == len(s))
    msg = db.explain(s)
    if msg is not None:
        low = msg.lower()
        if "incomplete input" in low:
            pass
        elif low.startswith("unrecognized token"):
            if not (open_last and last.kind in ("str", "dq", "qident", "op", "comment", "num")):
                return "syntax"
        elif "syntax error" in low:
            m = re.search(r'near "(.*)": syntax error', msg, re.S)
            near = m.group(1) if m else None
            if open_last and near is not None and s.endswith(near):
                pass
            else:
                cut = [i for i, ch in enumerate(s) if ch == "\n"]
                for i in cut:
                    head = s[:i].rstrip()
                    if head and _surface_ok(head) and db.explain(head) is None:
                        return "extra_after_sql"
                return "syntax"
    return _name_violation(toks, db, len(s))

def eos_label(text: str, db: DB, status: str, msg: str) -> str:
    pass
    if not text.strip():
        return "empty"
    if "``" in text:
        return analyze_prefix(text, db) or "prose_or_markup"
    if status == "timeout":
        return "timeout"
    low = msg.lower()
    if "incomplete input" in low or low.startswith("unrecognized token"):
        return "early_stop"
    if "no such column" in low:
        col = msg.split(":", 1)[-1].strip().split(".")[-1].strip().strip('"`[]').lower()
        return "wrong_table_column" if col in db.allcols else "unknown_column"
    if "no such table" in low:
        return "unknown_table"
    if "ambiguous column name" in low:
        return "ambiguous_column"
    if "no such function" in low:
        return "unknown_function"
    if "syntax error" in low:
        return "syntax"
    if "one statement at a time" in low:
        return "extra_after_sql"
    return "semantic_other"

def strict_check(text: str, db: DB, finish_reason: str = "stop"):
    pass
    s = text.strip()
    if finish_reason == "length":
        return {"valid": False, "status": "cap", "err": "hit max_new_tokens", "rows": None, "rows_truncated": False}
    if not s:
        return {"valid": False, "status": "error", "err": "empty", "rows": None, "rows_truncated": False}
    if not _surface_ok(s) or len(s) < 4:
        return {"valid": False, "status": "error", "err": "does not start with SELECT/WITH", "rows": None,
                "rows_truncated": False}
    st, res, trunc = db.execute(s)
    if st == "ok":
        return {"valid": True, "status": "ok", "err": None, "rows": res, "rows_truncated": trunc}
    return {"valid": False, "status": st, "err": res, "rows": None, "rows_truncated": False}

def first_violation(prefixes, db: DB):
    pass
    for idx, p in prefixes:
        lab = analyze_prefix(p, db)
        if lab is not None:
            return idx, lab
    return None, None

def classify(text: str, db: DB, finish_reason: str = "stop", token_prefixes=None) -> dict:
    pass
    chk = strict_check(text, db, finish_reason)
    if token_prefixes is None:
        token_prefixes = [(i, text[:i]) for i in range(1, len(text) + 1)]
    vi, vlab = first_violation(token_prefixes, db)
    if chk["valid"]:
        return {**chk, "label": None, "viol_index": None, "oracle_false_positive": vlab is not None}
    if vlab is None:
        if finish_reason == "length":
            vlab = "truncated"
        elif chk["status"] == "error" and chk["err"] == "does not start with SELECT/WITH":
            vlab = "prose_or_markup" if text.strip() else "empty"
        else:
            vlab = eos_label(text, db, chk["status"], chk["err"] or "")
    return {**chk, "label": vlab, "viol_index": vi, "oracle_false_positive": False}

def _permute_tuple(el, perm):
    return tuple(el[i] for i in perm)

def _unorder_row(row):
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

def _quick_rej(r1, r2, order_matters):
    s1 = [_unorder_row(r) for r in r1]
    s2 = [_unorder_row(r) for r in r2]
    return s1 == s2 if order_matters else set(s1) == set(s2)

def _multiset_eq(l1, l2):
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] += 1
    for e in l2:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True

def _constraint_perms(tab1_sets, result2, rng):
    num_cols = len(result2[0])
    cons = [set(range(num_cols)) for _ in range(num_cols)]
    if num_cols <= 3:
        return itertools.product(*cons)
    for _ in range(20):
        row = rng.choice(result2)
        for c1 in range(num_cols):
            for c2 in set(cons[c1]):
                if row[c2] not in tab1_sets[c1]:
                    cons[c1].remove(c2)
    return itertools.product(*cons)

def result_eq(r1, r2, order_matters: bool) -> bool:
    pass
    if len(r1) == 0 and len(r2) == 0:
        return True
    if len(r1) != len(r2):
        return False
    nc = len(r1[0])
    if len(r2[0]) != nc:
        return False
    if not _quick_rej(r1, r2, order_matters):
        return False
    rng = random.Random(0)
    sets = [{row[i] for row in r1} for i in range(nc)]
    for perm in _constraint_perms(sets, r2, rng):
        if len(perm) != len(set(perm)):
            continue
        r2p = r2 if nc == 1 else [_permute_tuple(e, perm) for e in r2]
        if order_matters:
            if r1 == r2p:
                return True
        elif _multiset_eq(r1, r2p):
            return True
    return False

def exec_match(pred_rows, gold_rows, gold_sql: str) -> bool:
    return result_eq(gold_rows, pred_rows, order_matters="order by" in gold_sql.lower())

if __name__ == "__main__":
    import sys
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.sqlite")
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE singer (Singer_ID int PRIMARY KEY, Name text, Age int, Country text);"
                      "CREATE TABLE concert (Concert_ID int PRIMARY KEY, Theme text, Singer_ID int,"
                      " FOREIGN KEY (Singer_ID) REFERENCES singer (Singer_ID));"
                      "INSERT INTO singer VALUES (1,'A',30,'France'),(2,'B',40,'US');"
                      "INSERT INTO concert VALUES (1,'x',1),(2,'y',2),(3,'z',1);")
    con.commit()
    con.close()
    print(schema_ddl(p))
    db = DB.get(p)
    cases = {
        "SELECT": None, "SELECT Na": None, "SELECT Name FROM singer": None, "SELECT Nme ": None, "SELECT Nme FROM": "unknown_column", "SELECT T1": None,
        "SELECT Nmx": None, "SELECT T1.Nmx": "unknown_column", "SELECT Name FROM singr ": "unknown_table", "SELECT Name FROM sin": None,
        "SELECT Name FROM singer;\nThis": "extra_after_sql", "Here is": "prose_or_markup", "```sql": "prose_or_markup",
        "SELECT Name FROM singer WHERE Age >": None, "SELECT Name FROM singer GROUP B": None,
        "SELECT Name FROM singer GROUP B ": "syntax", "SELECT T1.Name FROM singer AS T1 JOIN concert AS T2 ON T1.Singer_ID = T2.Theme": None,
        "SELECT T1.Name FROM singer AS T1 WHERE T1.Theme ": "wrong_table_column",
        "SELECT count(*) AS cnt FROM singer ORDER BY cnt": None, "SELECT Name FROM singer WHERE Name = 'ab": None,
        "SELECT YEAR(": "unknown_function", "SELECT count(*) FROM singer\nThe query ": "extra_after_sql",
        "SELECT count(*) FROM singer\n```": "extra_after_sql", "SELECT T1.Theme FROM singer AS T1": "wrong_table_column",
        "SELECT T1.Theme FROM singer AS T2": None,
        "SELECT Name, count(*) c FROM singer GROUP BY Name ORDER BY c DESC": None,
        "WITH x AS (SELECT Name FROM singer) SELECT Name FROM x": None,
    }
    bad = 0
    for q, want in cases.items():
        got = analyze_prefix(q, db)
        flag = "" if got == want else "   <-- MISMATCH"
        bad += got != want
        print(f"{q!r:90s} -> {got}{flag}")
    for text in ["SELECT Name FROM singer WHERE Age > 35", "SELECT Nme FROM singer", "SELECT T1.Theme FROM singer AS T1",
                 "SELECT Name FROM singer JOIN concert", "SELECT Singer_ID FROM singer JOIN concert", "SELECT Name FROM",
                 "SELECT Name FROM singer; SELECT 1"]:
        c = classify(text, db)
        print(f"{text!r:60s} valid={c['valid']} label={c['label']} err={c['err']}")
    g = db.execute("SELECT Name, Age FROM singer")[1]
    pr = db.execute("SELECT Age, Name FROM singer")[1]
    print("EX perm:", exec_match(pr, g, "SELECT Name, Age FROM singer"))
    sys.exit(1 if bad else 0)
