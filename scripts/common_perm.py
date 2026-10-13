#!/usr/bin/env python3
"""Parse reranking permutations."""
from __future__ import annotations

import math
import re

EOS_TOKENS = ("<|im_end|>", "<|endoftext|>")
TYPES = ("duplicate_id", "out_of_range", "overrun", "malformed",
         "early_stop_k1", "early_stop_short", "early_stop_mid", "length_cap")

class PermScanner:
    pass

    __slots__ = ("K", "phase", "digits", "used", "n", "viol", "viol_pos", "pos")

    def __init__(self, K: int):
        self.K = K
        self.phase = "start"
        self.digits = ""
        self.used: set[int] = set()
        self.n = 0
        self.viol = None
        self.viol_pos = -1
        self.pos = 0

    def copy(self) -> "PermScanner":
        s = PermScanner.__new__(PermScanner)
        s.K, s.phase, s.digits, s.used, s.n = self.K, self.phase, self.digits, set(self.used), self.n
        s.viol, s.viol_pos, s.pos = self.viol, self.viol_pos, self.pos
        return s

    def _fail(self, t: str) -> None:
        self.viol, self.viol_pos = t, self.pos

    def _id_prefix_status(self, d: str) -> str:
        pass
        if not d or d[0] == "0":
            return "range"
        cands = [i for i in range(1, self.K + 1) if str(i).startswith(d)]
        if not cands:
            return "range"
        return "ok" if any(i not in self.used for i in cands) else "dup"

    def feed(self, text: str) -> "PermScanner":
        for c in text:
            if self.viol is not None:
                return self
            ph = self.phase
            if ph == "start":
                if c.isspace():
                    pass
                elif c == "[":
                    self.phase, self.digits = "in_id", ""
                else:
                    self._fail("malformed")
            elif ph == "in_id":
                if "0" <= c <= "9":
                    st = self._id_prefix_status(self.digits + c)
                    if st == "ok":
                        self.digits += c
                    else:
                        self._fail("duplicate_id" if st == "dup" else "out_of_range")
                elif c == "]":
                    if not self.digits:
                        self._fail("malformed")
                    else:
                        i = int(self.digits)
                        if i in self.used:
                            self._fail("duplicate_id")
                        else:
                            self.used.add(i)
                            self.n += 1
                            self.phase = "after_item"
                else:
                    self._fail("malformed")
            elif ph == "after_item":
                if c.isspace():
                    pass
                elif self.n >= self.K:
                    self._fail("overrun")
                elif c == ">":
                    self.phase = "after_gt"
                else:
                    self._fail("malformed")
            elif ph == "after_gt":
                if c.isspace():
                    pass
                elif c == "[":
                    self.phase, self.digits = "in_id", ""
                else:
                    self._fail("malformed")
            self.pos += 1
        return self

    @property
    def completable(self) -> bool:
        return self.viol is None

    @property
    def complete(self) -> bool:
        pass
        return self.viol is None and self.phase == "after_item" and self.n == self.K

    def stop_type(self):
        pass
        if self.viol is not None:
            return self.viol
        if self.complete:
            return None
        if self.phase in ("after_item", "start"):
            return "early_stop_k1" if self.n == self.K - 1 else "early_stop_short"
        return "early_stop_mid"

def scan(text: str, K: int) -> PermScanner:
    return PermScanner(K).feed(text)

def completable(prefix_text: str, K: int) -> bool:
    pass
    return scan(prefix_text, K).completable

def violation_type(prefix_text: str, K: int):
    pass
    return scan(prefix_text, K).viol

def token_violation(state: PermScanner, piece: str, is_eos: bool):
    pass
    if state.viol is not None:
        return state.viol
    if is_eos:
        return state.stop_type()
    s = state.copy().feed(piece)
    return s.viol

def parse_strict(text: str, K: int):
    pass
    s = scan(text, K)
    if not s.complete:
        return None
    ids = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    assert len(ids) == K and sorted(ids) == list(range(1, K + 1)), text
    return [i - 1 for i in ids]

def first_violation(text: str, K: int, finish_reason: str):
    pass
    s = scan(text, K)
    if s.viol is not None:
        return s.viol
    if finish_reason == "length":
        return "length_cap"
    return s.stop_type()

def n_items_before_violation(text: str, K: int) -> int:
    return scan(text, K).n

def rankgpt_repair(text: str, K: int) -> list[int]:
    pass
    digits = "".join(c if c.isdigit() else " " for c in text).strip()
    resp = [int(x) - 1 for x in digits.split()] if digits else []
    seen, out = set(), []
    for r in resp:
        if 0 <= r < K and r not in seen:
            seen.add(r)
            out.append(r)
    return out + [i for i in range(K) if i not in seen]

def load_qrels(path: str) -> dict[str, dict[str, int]]:
    q: dict[str, dict[str, int]] = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, did, rel = parts[:4]
        q.setdefault(str(qid), {})[str(did)] = int(rel)
    return q

def ndcg_at_k(ranked_docids: list[str], qrel: dict[str, int], k: int = 10) -> float:
    pass
    dcg = 0.0
    for r, d in enumerate(ranked_docids[:k]):
        g = qrel.get(str(d), 0)
        if g > 0:
            dcg += g / math.log2(r + 2)
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def kendall_tau_perm(a: list[int], b: list[int]) -> float:
    pass
    K = len(a)
    ra = {x: i for i, x in enumerate(a)}
    rb = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    items = list(ra)
    for i in range(K):
        for j in range(i + 1, K):
            x, y = items[i], items[j]
            s = (ra[x] - ra[y]) * (rb[x] - rb[y])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n = K * (K - 1) / 2
    return (conc - disc) / n if n else 0.0

SYSTEM = "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."

def render_user(query: str, passages: list[str]) -> str:
    pass
    K = len(passages)
    head = (f"I will provide you with {K} passages, each indicated by a numerical identifier []. "
            f"Rank the passages based on their relevance to the search query: {query}.\n\n")
    body = "".join(f"[{i}] {p}\n" for i, p in enumerate(passages, 1))
    tail = (f"Search Query: {query}.\nRank the {K} passages above based on their relevance to the search query. "
            f"All the passages should be included and listed using identifiers, in descending order of relevance. "
            f"The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, "
            f"do not say any word or explain.")
    return head + body + tail

def format_perm(order: list[int]) -> str:
    pass
    return " > ".join(f"[{i + 1}]" for i in order)

def _selftest() -> None:
    K = 20
    good = format_perm(list(range(K))[::-1])
    assert parse_strict(good, K) == list(range(K))[::-1]
    assert first_violation(good, K, "stop") is None
    assert first_violation("  " + good + "\n", K, "stop") is None
    for i in range(len(good) + 1):
        assert completable(good[:i], K), good[:i]
    items = good.split(" > ")
    assert first_violation(" > ".join(items[:-1]), K, "stop") == "early_stop_k1"
    assert first_violation(" > ".join(items[:5]), K, "stop") == "early_stop_short"
    assert first_violation("", K, "stop") == "early_stop_short"
    assert first_violation(" > ".join(items[:5]) + " >", K, "stop") == "early_stop_mid"
    assert first_violation(" > ".join(items[:5]) + " > [", K, "stop") == "early_stop_mid"
    dup = " > ".join(items[:5] + [items[0]] + items[5:-1])
    assert first_violation(dup, K, "stop") == "duplicate_id", dup
    assert first_violation("[21] > " + good, K, "stop") == "out_of_range"
    assert first_violation("[0]", K, "stop") == "out_of_range"
    assert first_violation("[3", K, "stop") == "early_stop_mid"
    assert first_violation(good + " > [1]", K, "stop") == "overrun"
    assert first_violation("The ranking is " + good, K, "stop") == "malformed"
    assert first_violation("<think>\n\n</think>\n\n" + good, K, "stop") == "malformed"
    assert first_violation(good.replace(" > ", ", "), K, "stop") == "malformed"
    assert first_violation(good[:30], K, "length") == "length_cap"
    s = scan("[1] > [1", K)
    assert s.completable and token_violation(s, "]", False) == "duplicate_id" and token_violation(s, "0", False) is None
    s = scan("[2] > [2", K)
    assert token_violation(s, "]", False) == "duplicate_id" and token_violation(s, "1", False) == "out_of_range"
    assert token_violation(s, "0", False) is None
    assert scan("[2] > [20] > [2", K).viol == "duplicate_id"
    s = scan(" > ".join(items), K)
    assert token_violation(s, "", True) is None and token_violation(s, " >", False) == "overrun"
    s = scan(" > ".join(items[:-1]), K)
    assert token_violation(s, "", True) == "early_stop_k1" and token_violation(s, " >", False) is None
    assert rankgpt_repair("[3] > [1] > [3] > [25]", 4) == [2, 0, 1, 3]
    assert rankgpt_repair("nonsense", 3) == [0, 1, 2]
    assert abs(ndcg_at_k(["a", "b"], {"a": 3, "b": 1}) - 1.0) < 1e-12
    assert kendall_tau_perm([0, 1, 2], [0, 1, 2]) == 1.0 and kendall_tau_perm([0, 1, 2], [2, 1, 0]) == -1.0
    print("common_perm self-test OK")

if __name__ == "__main__":
    _selftest()
