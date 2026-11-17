                                                                                

                                                                          
                                                                          
                                                                           
                                                                         
                                                                     
   

from __future__ import annotations

from collections import Counter
import math
import random
import re
import string
from typing import Iterable, Mapping, Sequence


def probability(value: object) -> float:
    pass                                                                        
    if isinstance(value, bool):
        raise ValueError("A probability must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("A probability must be numeric") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return result


def retrieval_metrics(
    ranked_doc_ids: Sequence[str], relevance: Mapping[str, float] | None, k: int = 10
) -> dict[str, float | None]:
    pass                                                                          

                                                                             
                                                                       
       
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    ranked = [str(doc_id) for doc_id in ranked_doc_ids]
    if len(set(ranked)) != len(ranked):
        raise ValueError("A ranking must contain unique document IDs")
    names = (f"ndcg@{k}", f"recall@{k}", f"mrr@{k}", f"map@{k}",
             f"precision@{k}", f"judged_fraction@{k}", f"unjudged_fraction@{k}")
    if relevance is None:
        return dict.fromkeys(names)
    judged = {}
    for doc_id, grade in relevance.items():
        score = float(grade)
        if not math.isfinite(score):
            raise ValueError("Relevance grades must be finite")
        if score >= 0:
            judged[str(doc_id)] = score
    if not judged:
        return dict.fromkeys(names)
    top = ranked[:k]
    relevant = {doc for doc, grade in judged.items() if grade > 0}
    gains = [judged.get(doc, 0.0) for doc in top]
    dcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(gains))
    ideal = sum(grade / math.log2(rank + 2)
                for rank, grade in enumerate(sorted(judged.values(), reverse=True)[:k]))
    hits = 0
    precision_sum = 0.0
    reciprocal = 0.0
    for rank, doc in enumerate(top, 1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / rank
            if not reciprocal:
                reciprocal = 1 / rank
    judged_fraction = sum(doc in judged for doc in top) / len(top) if top else None
    return {
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
        f"recall@{k}": hits / len(relevant) if relevant else 0.0,
        f"mrr@{k}": reciprocal,
        f"map@{k}": precision_sum / len(relevant) if relevant else 0.0,
        f"precision@{k}": hits / k,
        f"judged_fraction@{k}": judged_fraction,
        f"unjudged_fraction@{k}": 1 - judged_fraction if judged_fraction is not None else None,
    }


def normalize_answer(answer: str) -> str:
    answer = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def token_scores(prediction: str, reference: str) -> tuple[float, float, float]:
    pass                                                                          
    predicted, expected = normalize_answer(prediction), normalize_answer(reference)
    special = {"yes", "no", "noanswer"}
    if predicted != expected and (predicted in special or expected in special):
        return 0.0, 0.0, 0.0
    left, right = predicted.split(), expected.split()
    overlap = sum((Counter(left) & Counter(right)).values())
    if not overlap:
        return 0.0, 0.0, 0.0
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall), precision, recall


def answer_metrics(prediction: str, references: str | Sequence[str] | None) -> dict:
    names = ("answer_em", "answer_f1", "answer_precision", "answer_recall")
    if references is None:
        return dict.fromkeys(names)
    refs = [references] if isinstance(references, str) else list(references)
    if not refs:
        return dict.fromkeys(names)
    if not isinstance(prediction, str) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("Answers and reference answers must be strings")
    scores = [token_scores(prediction, reference) for reference in refs]
    return {
        "answer_em": float(any(normalize_answer(prediction) == normalize_answer(ref) for ref in refs)),
        "answer_f1": max(score[0] for score in scores),
        "answer_precision": max(score[1] for score in scores),
        "answer_recall": max(score[2] for score in scores),
    }


def _evidence_key(item: object) -> tuple[str, object | None]:
    if isinstance(item, str):
        return item, None
    if isinstance(item, Mapping):
        doc_id = item.get("doc_id", item.get("document_id", item.get("title")))
        if doc_id is None:
            raise ValueError("An evidence entry needs doc_id")
        sentence = item.get("sentence_id", item.get("sentence_index"))
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        doc_id, sentence = item
    else:
        raise ValueError("Evidence must be a document ID, a mapping, or [doc_id, sentence_id]")
    if sentence is not None and (isinstance(sentence, bool) or not isinstance(sentence, (int, str))):
        raise ValueError("sentence_id must be an integer or string")
    return str(doc_id), sentence


def _set_scores(predicted: set, expected: set) -> tuple[float, float, float, float]:
    overlap = len(predicted & expected)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, float(predicted == expected)


def evidence_metrics(citations: Iterable[object], supporting_evidence: Iterable[object] | None) -> dict:
    pass                                                                    

                                                                               
                                                                             
                                                                               
       
    names = ("citation_precision", "citation_recall", "citation_f1", "citation_em",
             "evidence_sentence_precision", "evidence_sentence_recall",
             "evidence_sentence_f1", "evidence_sentence_em")
    if supporting_evidence is None:
        return dict.fromkeys(names)
    predicted = {_evidence_key(item) for item in citations}
    expected = {_evidence_key(item) for item in supporting_evidence}
    doc_scores = _set_scores({doc for doc, _ in predicted}, {doc for doc, _ in expected})
    result = dict(zip(names[:4], doc_scores))
    if not expected or any(sentence is None for _, sentence in expected):
        result.update(dict.fromkeys(names[4:]))
    else:
                                                                               
        result.update(zip(names[4:], _set_scores(predicted, expected)))
    return result


def brier_score(confidences: Sequence[float], outcomes: Sequence[bool | int]) -> float | None:
    pairs = _probability_pairs(confidences, outcomes)
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else None


def _probability_pairs(confidences: Sequence[float], outcomes: Sequence[bool | int]) -> list:
    if len(confidences) != len(outcomes):
        raise ValueError("Confidences and outcomes must have the same length")
    if any(y not in (0, 1) for y in outcomes):
        raise ValueError("Calibration outcomes must be binary")
    return [(probability(p), int(y)) for p, y in zip(confidences, outcomes)]


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[bool | int], bins: int = 10
) -> float | None:
    pass                                                              
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    pairs = _probability_pairs(confidences, outcomes)
    if not pairs:
        return None
    buckets = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    return sum(abs(sum(p for p, _ in bucket) - sum(y for _, y in bucket))
               for bucket in buckets) / len(pairs)


def selective_metrics(correct: Sequence[bool | int], abstained: Sequence[bool]) -> dict:
    if len(correct) != len(abstained):
        raise ValueError("Correctness and abstention arrays must have the same length")
    if any(y not in (0, 1) for y in correct) or any(type(a) is not bool for a in abstained):
        raise ValueError("Correctness must be binary and abstained must be boolean")
    answered = [int(y) for y, abstain in zip(correct, abstained) if not abstain]
    return {
        "coverage": len(answered) / len(correct) if correct else None,
        "selective_accuracy": sum(answered) / len(answered) if answered else None,
        "selective_risk": 1 - sum(answered) / len(answered) if answered else None,
        "answered_count": len(answered), "eligible_count": len(correct),
    }


def abstention_metrics(answerable: Sequence[bool], abstained: Sequence[bool]) -> dict:
    if len(answerable) != len(abstained):
        raise ValueError("Answerability and abstention arrays must have the same length")
    if any(type(x) is not bool for x in [*answerable, *abstained]):
        raise ValueError("Answerability and abstention must be boolean")
    missing = sum(not value for value in answerable)
    present = sum(answerable)
    abstain_count = sum(abstained)
    correct_abstentions = sum(not a and b for a, b in zip(answerable, abstained))
    false_abstentions = sum(a and b for a, b in zip(answerable, abstained))
    return {
        "abstention_precision": correct_abstentions / abstain_count if abstain_count else None,
        "missing_evidence_abstention_recall": correct_abstentions / missing if missing else None,
        "unsupported_answer_rate": (missing - correct_abstentions) / missing if missing else None,
        "false_abstention_rate": false_abstentions / present if present else None,
        "answerability_decision_accuracy": (correct_abstentions + present - false_abstentions) / len(answerable)
        if answerable else None,
        "missing_evidence_count": missing, "answerable_count": present,
    }


def paired_bootstrap(
    candidate: Sequence[float], baseline: Sequence[float], *, samples: int = 2000,
    confidence: float = 0.95, seed: int = 0, cluster_ids: Sequence[str] | None = None,
) -> dict:
    pass                                                                 

                                                                           
                                                                  
       
    if len(candidate) != len(baseline):
        raise ValueError("Paired runs must have equal lengths")
    if not 0 < confidence < 1 or isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("Need positive samples and confidence strictly between 0 and 1")
    differences = [float(c) - float(b) for c, b in zip(candidate, baseline)]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Bootstrap values must be finite")
    ids = list(cluster_ids) if cluster_ids is not None else list(range(len(differences)))
    if len(ids) != len(differences):
        raise ValueError("Need one cluster ID per pair")
    groups = {}
    for key, difference in zip(ids, differences):
        groups.setdefault(key, []).append(difference)
    result = {"delta": sum(differences) / len(differences) if differences else None,
              "ci_low": None, "ci_high": None, "confidence": confidence,
              "pairs": len(differences), "clusters": len(groups), "samples": samples, "seed": seed,
              "status": "insufficient_pairs"}
    if len(groups) < 2:
        return result
    rng = random.Random(seed)
    units = list(groups.values())
    draws = []
    for _ in range(samples):
        draw = [rng.choice(units) for _ in units]
        draws.append(sum(sum(unit) for unit in draw) / sum(len(unit) for unit in draw))
    draws.sort()
    alpha = (1 - confidence) / 2
                                                                             
    result.update(ci_low=draws[max(0, math.ceil(alpha * samples) - 1)],
                  ci_high=draws[min(samples - 1, math.ceil((1 - alpha) * samples) - 1)],
                  status="ok")
    return result
