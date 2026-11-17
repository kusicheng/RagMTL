                                                           

                                                                            
                                                                                 
                                                                          
   

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
import itertools
import re
from typing import Any

from .storage import DocumentStore, tokenize

_STOP = set("a an the and or to of in for on is are was were be been with by as at "
            "from that this which what who when where how does do did has have had".split())
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_MAX_QUERY_TERMS = 64
_ANCHORS_PER_DOCUMENT = 66
_MAX_ANCHORS = 666
_NOVELTY_TERMS = 128


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str = "magra"
    top_k: int = 10
    candidate_k: int = 50
    title_weight: float = 2.0
    context_bytes: int = 12_000
    initial_k: int = 10
    coverage_stop: float = 0.8
    min_gain: float = 0.001
    diversity_weight: float = 0.15
    redundancy_weight: float = 0.35
    bridge_terms: int = 6
    use_dense: bool = False
    use_reranker: bool = False
    evidence_selection: str = "auto"

    def __post_init__(self):
        if self.strategy not in {"bm25", "dense", "hybrid", "rerank", "magra", "magra_no_bridge", "magra_no_selection"}:
            raise ValueError("Unknown retrieval strategy")
        if self.evidence_selection not in {"auto", "query_focused", "prefix", "marginal_gain"}:
            raise ValueError("Unknown evidence selection method")
        if not 1 <= self.top_k <= self.candidate_k <= 1000:
            raise ValueError("Require 1 <= top_k <= candidate_k <= 1000")
        if not 1 <= self.initial_k <= self.candidate_k:
            raise ValueError("initial_k must be within candidate_k")
        if not 128 <= self.context_bytes <= 1_000_000:
            raise ValueError("context_bytes must be 128..1000000")
        if self.title_weight <= 0 or not 0 <= self.coverage_stop <= 1:
            raise ValueError("Invalid title weight or coverage threshold")
        if min(self.min_gain, self.diversity_weight, self.redundancy_weight, self.bridge_terms) < 0:
            raise ValueError("Selection weights and bridge terms must be nonnegative")


def reciprocal_rank_fusion(rankings: list[list[dict]], *, constant: float = 60) -> list[dict]:
    scores, docs = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, doc in enumerate(ranking, 1):
            if doc["id"] in seen:
                continue
            seen.add(doc["id"])
            docs.setdefault(doc["id"], doc)
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (constant + rank)
    return [dict(docs[doc_id], score=score, retriever="rrf")
            for doc_id, score in sorted(scores.items(), key=lambda t: (-t[1], t[0]))]


def _sentence_texts(doc: dict):
    pass                                                                         

                                                                                 
                                                                             
       
    ss = doc.get("metadata", {}).get("sentences")
    if isinstance(ss, list) and ss:
        for i, text in enumerate(ss):
            if text.strip():
                yield i, text
        return
    text, start, sentence_id = doc["text"], 0, 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        part = text[start:boundary.start()].strip()
        start = boundary.end()
        if part:
            yield sentence_id, part
            sentence_id += 1
    part = text[start:].strip()
    if part:
        yield sentence_id, part


def _unit(doc: dict, sentence_id: int, text: str) -> dict:
    metadata = doc.get("metadata", {})
    return {"doc_id": doc["id"], "sentence_id": sentence_id, "title": doc["title"],
            "text": text, "published_at": metadata.get("published_at", metadata.get("document_date", metadata.get("date"))),
            "fetched_at": metadata.get("fetched_at"),
            "url": metadata.get("url", metadata.get("source_url")),
            "source": metadata.get("source", "")}


def _units(doc: dict):
    for sentence_id, text in _sentence_texts(doc):
        yield _unit(doc, sentence_id, text)


def evidence_context(evidence: list[dict]) -> str:
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def _query_terms(question: str) -> set[str]:
                                                                                
                                                             
    terms = set()
    for token in _TOKEN.finditer(question.lower()):
        word = token.group()
        if word not in _STOP:
            terms.add(word)
            if len(terms) == _MAX_QUERY_TERMS:
                break
    return terms


def _body_counts(text: str, terms: set[str]) -> tuple[int, dict[str, int]]:
    counts, length = {}, 0
    for token in _TOKEN.finditer(text.lower()):
        length += 1
        word = token.group()
        if word in terms:
            counts[word] = counts.get(word, 0) + 1
    return length, counts


def _novelty_words(text: str) -> set[int]:
    pass                                                                             
    words = set()
    for token in _TOKEN.finditer(text.lower()):
        word = token.group()
        if word not in _STOP:
            words.add(int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big"))
            if len(words) == _NOVELTY_TERMS:
                break
    return words


def _keep_anchor(heap: list, item: tuple, limit: int):
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


@dataclass
class _Window:
    score: float
    tie: int
    rank: int
    keys: tuple


def _evidence_pool(question: str, candidates: list[dict], max_bytes: int):
    pass                                                                       

                                                                               
                                                                                
                                                                                 
                                                                              
                                                                               
       
    terms, documents, seen = _query_terms(question), [], set()
    for doc in candidates:
        if doc["id"] not in seen:
            documents.append(doc)
            seen.add(doc["id"])
    count, total_length, df = 0, 0, dict.fromkeys(terms, 0)
    for doc in documents:
        for _, text in _sentence_texts(doc):
            length, hits = _body_counts(text, terms) if terms else (0, {})
            count += 1
            total_length += length
            for term in hits:
                df[term] += 1
    average_length = total_length / max(1, count)
    idf = {term: max(1e-6, math.log((count - frequency + 0.5) / (frequency + 0.5)))
           for term, frequency in df.items()}
    anchors = []
    seed = hashlib.sha256(question.encode()).digest()
    for rank, doc in enumerate(documents):
        local = []
        doc_seed = seed + doc["id"].encode() + b"\0"
        for ordinal, (sentence_id, text) in enumerate(_sentence_texts(doc)):
            length, hits = _body_counts(text, terms) if terms else (0, {})
            normalizer = 1.2 * (0.25 + 0.75 * length / max(1, average_length))
            score = sum(idf[term] * frequency * 2.2 / (frequency + normalizer)
                        for term, frequency in hits.items())
            tie = int.from_bytes(hashlib.blake2b(doc_seed + str(sentence_id).encode(), digest_size=8).digest(), "big")
            _keep_anchor(local, (score, tie, rank, sentence_id, ordinal), _ANCHORS_PER_DOCUMENT)
        for anchor in local:
            _keep_anchor(anchors, anchor, _MAX_ANCHORS)
                                                                              
                                                                         
    if any(anchor[0] > 0 for anchor in anchors):
        anchors = [anchor for anchor in anchors if anchor[0] > 0]
                                                                            
    by_document = {}
    for score, tie, rank, sentence_id, ordinal in anchors:
        by_document.setdefault(rank, []).append((score, tie, sentence_id, ordinal))
    units, sizes, hits, words, windows = {}, {}, {}, {}, []
    for rank, doc_anchors in by_document.items():
        wanted = {neighbor for _, _, _, ordinal in doc_anchors
                  for neighbor in (ordinal - 1, ordinal, ordinal + 1) if neighbor >= 0}
        ordinals, too_large = {}, set()
        for ordinal, (sentence_id, text) in enumerate(_sentence_texts(documents[rank])):
            if ordinal not in wanted:
                continue
            key = (documents[rank]["id"], sentence_id)
            ordinals[ordinal] = key
            unit = _unit(documents[rank], sentence_id, text)
                                                                               
            if sum(len(value) for value in unit.values() if isinstance(value, str)) > max_bytes:
                too_large.add(key)
                continue
            size = len(evidence_context([unit]).encode()) - 2
            if size + 2 > max_bytes:
                too_large.add(key)
                continue
            units[key], sizes[key] = unit, size
            hits[key] = set(_body_counts(text, terms)[1]) if terms else set()
            words[key] = _novelty_words(text)
        for score, tie, _, ordinal in doc_anchors:
            keys = tuple(ordinals[o] for o in (ordinal - 1, ordinal, ordinal + 1) if o in ordinals)
            if keys and not any(key in too_large for key in keys):
                windows.append(_Window(score, tie, rank, keys))
    windows.sort(key=lambda window: (-window.score, -window.tie, window.rank, window.keys))
    return terms, units, sizes, hits, words, windows


def _incremental_bytes(keys, sizes: dict, selected_count: int) -> int:
    return sum(sizes[key] for key in keys) + len(keys) - (not selected_count)


def _pack_evidence(question: str, candidates: list[dict], config: RetrievalConfig, *, marginal: bool) -> dict:
    terms, units, sizes, hits, words, windows = _evidence_pool(question, candidates, config.context_bytes)
    selected, selected_keys, covered, used_docs, selected_words = [], set(), set(), set(), set()
    context_bytes = 2                                             
    while windows:
        best, best_utility = None, -float("inf")
        for i, window in enumerate(windows):
            keys = tuple(key for key in window.keys if key not in selected_keys)
            if not keys:
                continue
            cost = _incremental_bytes(keys, sizes, len(selected))
            if context_bytes + cost > config.context_bytes:
                continue
            if not marginal:
                best = i
                break
            body_terms = set().union(*(hits[key] for key in keys))
            body_words = set().union(*(words[key] for key in keys))
            new = len(body_terms - covered) / max(1, len(terms))
            relevance = len(body_terms) / max(1, len(terms))
            new_doc = keys[0][0] not in used_docs
            prior = 0.15 / (window.rank + 1) if new_doc else 0
            diversity = config.diversity_weight if new_doc else 0
            repetition = len(body_words & selected_words) / max(1, len(body_words))
            gain = new + 0.2 * relevance + prior + diversity - config.redundancy_weight * repetition
            utility = gain / max(1, math.sqrt(cost))
                                                                          
            if utility > best_utility:
                best, best_utility = i, utility
        if best is None or (marginal and best_utility < config.min_gain):
            break
        window = windows.pop(best)
        keys = tuple(key for key in window.keys if key not in selected_keys)
        context_bytes += _incremental_bytes(keys, sizes, len(selected))
        for key in keys:
            selected.append(units[key])
            selected_keys.add(key)
            covered |= hits[key]
            selected_words |= words[key]
            used_docs.add(key[0])
    return {"evidence": selected, "coverage": len(covered) / max(1, len(terms)),
            "context_bytes": context_bytes, "coverage_is_confidence": False}


def select_evidence(question: str, candidates: list[dict], config: RetrievalConfig) -> dict:
    pass                                                                  
    return _pack_evidence(question, candidates, config, marginal=True)


def query_focused_evidence(question: str, candidates: list[dict], config: RetrievalConfig) -> dict:
    pass                                                                           
    return _pack_evidence(question, candidates, config, marginal=False)


def fixed_evidence(candidates: list[dict], max_bytes: int) -> dict:
    pass                                                                  
    selected, context_bytes, considered = [], 2, 0
    for doc in candidates:
        if considered == 2000:
            break
        for unit in itertools.islice(_units(doc), 200):
            if considered == 2000:
                break
            considered += 1
            if sum(len(value) for value in unit.values() if isinstance(value, str)) > max_bytes:
                continue
            size = len(evidence_context([unit]).encode()) - 2 + bool(selected)
            if context_bytes + size <= max_bytes:
                selected.append(unit)
                context_bytes += size
    return {"evidence": selected, "coverage": None,
            "context_bytes": context_bytes,
            "coverage_is_confidence": False}


class Retriever:
    def __init__(self, store: DocumentStore, config: RetrievalConfig | None = None, *,
                 dense: Any = None, reranker: Any = None):
        self.store = store
        self.config = config or RetrievalConfig()
        self.dense, self.reranker = dense, reranker
        if (self.config.use_dense or self.config.strategy in {"dense", "hybrid", "rerank"}) and dense is None:
            raise ValueError("Requested strategy requires an explicit dense retriever")
        if (self.config.use_reranker or self.config.strategy == "rerank") and reranker is None:
            raise ValueError("Requested strategy requires an explicit cross-encoder")

    def retrieve(self, question: str, *, allowed_ids=None, exclude_ids=None, source=None,
                 id_prefix=None) -> dict:
        c = self.config
        filters = {"allowed_ids": allowed_ids, "exclude_ids": exclude_ids, "source": source,
                   "id_prefix": id_prefix}
        trace = []
        n = c.initial_k if c.strategy.startswith("magra") else c.candidate_k
        lexical = []
        if c.strategy != "dense":
            lexical = self.store.search(question, k=n, title_weight=c.title_weight, **filters)
            trace.append({"stage": "bm25", "candidates": len(lexical)})
        ranking = lexical
        initial = None
        reusable_initial = False
        if c.strategy == "dense":
            ranking = self.dense.search(question, k=c.candidate_k, **filters)
            trace.append({"stage": "dense", "candidates": len(ranking)})
        elif c.use_dense or c.strategy in {"hybrid", "rerank"}:
            dense = self.dense.search(question, k=c.candidate_k, **filters)
            ranking = reciprocal_rank_fusion([lexical, dense])[:c.candidate_k]
            trace.append({"stage": "hybrid", "candidates": len(ranking)})
        if c.strategy.startswith("magra"):
            initial = select_evidence(question, ranking[:c.top_k], c)
            reusable_initial = True
            if initial["coverage"] < c.coverage_stop or len(ranking) < min(2, c.top_k):
                                                                            
                                                                             
                reusable_initial = False
                broad = self.store.search(question, k=c.candidate_k, title_weight=c.title_weight, **filters)
                ranking = reciprocal_rank_fusion([ranking, broad])[:c.candidate_k]
                trace.append({"stage": "expand", "reason": "low lexical coverage", "candidates": len(ranking)})
                                                                                           
            if c.strategy != "magra_no_bridge" and c.bridge_terms and initial["coverage"] < c.coverage_stop:
                original = set(tokenize(question))
                bridge = list(dict.fromkeys(t for d in ranking[:2] for t in tokenize(d["title"])
                                            if t not in original and t not in _STOP))[:c.bridge_terms]
                if bridge:
                    reusable_initial = False
                    expanded = self.store.search(question + " " + " ".join(bridge),
                                                 k=c.candidate_k, title_weight=c.title_weight, **filters)
                    ranking = reciprocal_rank_fusion([ranking, expanded])[:c.candidate_k]
                    trace.append({"stage": "bridge", "terms": bridge, "candidates": len(ranking)})
        if c.use_reranker or c.strategy == "rerank":
            reusable_initial = False
            ranking = self.reranker.rerank(question, ranking)[:c.candidate_k]
            trace.append({"stage": "cross_encoder", "candidates": len(ranking)})
        ranking = ranking[:c.top_k]
        method = c.evidence_selection
        if c.strategy == "magra_no_selection":
            method = "prefix"
        elif method == "auto":
            method = "marginal_gain" if c.strategy.startswith("magra") else "query_focused"
        if method == "marginal_gain":
            chosen = initial if initial is not None and reusable_initial else select_evidence(question, ranking, c)
        elif method == "query_focused":
            chosen = query_focused_evidence(question, ranking, c)
        else:
            chosen = fixed_evidence(ranking, c.context_bytes)
        trace.append({"stage": "evidence_selection", "method": method})
        return {"ranked_doc_ids": [d["id"] for d in ranking], "documents": ranking,
                **chosen, "trace": trace, "config": asdict(c)}
