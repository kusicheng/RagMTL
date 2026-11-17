                                                                             

                                                                               
                                                                                
                                                                                 
                                          
   

from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta
from typing import Any, Iterable, Mapping


CASES = (
    "missing_evidence",
    "conflicting_sources",
    "stale_facts",
    "lexical_mismatch",
    "irrelevant_context",
)
GENERATOR = "magra.synthetic_failures.v1"
_ANNOTATIONS = {
    "labels", "answer", "answers", "expected_answer", "expected_behavior",
    "qrels", "relevance", "supporting_facts", "withheld_evidence",
}


def _document(value: Mapping[str, Any]) -> dict[str, Any]:
    identifier = str(value["id"])
    if not identifier:
        raise ValueError("document id must not be empty")
    metadata = copy.deepcopy(dict(value.get("metadata", {})))
    if _ANNOTATIONS.intersection(metadata):
        raise ValueError("evaluation annotations must stay outside indexed metadata")
    return {
        "id": identifier,
        "title": str(value.get("title", "")),
        "text": str(value.get("text", "")),
        "metadata": metadata,
    }


def materialize_overlay(
    documents: Iterable[Mapping[str, Any]], overlay: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pass                                                                          

                                                                               
                                                                                 
       
    excluded = {str(value) for value in overlay.get("exclude_ids", ())}
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for value in documents:
        document = _document(value)
        identifier = document["id"]
        if identifier in seen:
            raise ValueError(f"duplicate document id: {identifier}")
        seen.add(identifier)
        if identifier not in excluded:
            output.append(document)
    for value in overlay.get("add_documents", ()):
        document = _document(value)
        identifier = document["id"]
        if identifier in excluded or identifier in seen:
            raise ValueError(f"overlay document id collides: {identifier}")
        seen.add(identifier)
        output.append(document)
    return output


def build_failure_case(
    case: str,
    documents: Iterable[Mapping[str, Any]] = (),
    query: str | Mapping[str, Any] | None = None,
    relevant_ids: Iterable[str] = (),
    *,
    seed: int = 0,
    as_of: str = "2025-01-01",
) -> dict[str, Any]:
    pass                                                                        

                                                                                 
                                                                                
                                                                                 
                                                                           
       
    if case not in CASES:
        raise ValueError(f"unknown robustness case: {case}")
    reference_date = date.fromisoformat(as_of)
    base = [_document(document) for document in documents]
    if len({document["id"] for document in base}) != len(base):
        raise ValueError("duplicate document ids in background corpus")
    supplied_ids = {str(identifier) for identifier in relevant_ids}
    unknown = supplied_ids.difference(document["id"] for document in base)
    if unknown:
        raise ValueError(f"unknown relevant document ids: {sorted(unknown)}")
    fingerprint = hashlib.sha256(f"{GENERATOR}:{case}:{seed}:{as_of}".encode()).hexdigest()
    prefix = f"synthetic:{case}:{fingerprint[:16]}"
    entity = f"Magra reference unit {fingerprint[:12]}"
    code = f"{int(fingerprint[:10], 16) % 900000 + 100000:06d}"
    alternative = f"{(int(code) - 100000 + 137) % 900000 + 100000:06d}"

    def make(suffix: str, text: str, document_date: str | None = None) -> dict[str, Any]:
        return {
            "id": f"{prefix}:{suffix}",
            "title": f"{entity}: {suffix.replace('_', ' ')}",
            "text": text,
            "metadata": {
                "synthetic": True,
                "generator": GENERATOR,
                "case": case,
                "document_date": document_date or as_of,
                "domain": "synthetic_controlled",
            },
        }

    support = make("record", f"The access code assigned to {entity} is {code}.")
    additions: list[dict[str, Any]] = []
    excluded: set[str] = set()
    labels: dict[str, Any] = {
        "answerable": True,
        "expected_behavior": "answer_with_evidence",
        "expected_answer": code,
        "relevant_ids": [support["id"]],
        "supporting_ids": [support["id"]],
    }
    question = f"What is the access code assigned to {entity}?"
    if case == "missing_evidence":
        selected_texts = {
            " ".join(document["text"].split()).casefold()
            for document in base if document["id"] in supplied_ids
        }
        excluded.update(
            document["id"] for document in base
            if document["id"] in supplied_ids
            or " ".join(document["text"].split()).casefold() in selected_texts
            or entity.casefold() in (document["title"] + " " + document["text"]).casefold()
        )
        excluded.add(support["id"])
        additions.append(make("directory", f"{entity} has a directory entry. No access code is recorded in this entry."))
        labels.update(
            answerable=False, expected_behavior="abstain", expected_answer=None,
            relevant_ids=[], supporting_ids=[], withheld_evidence=[support],
            removed_background_ids=sorted(excluded.difference({support["id"]})),
            evidence_absent_by_construction=True,
        )
    elif case == "conflicting_sources":
        first = make("source_a", f"Independent record A states that the access code assigned to {entity} is {code}.")
        second = make("source_b", f"Independent record B states that the access code assigned to {entity} is {alternative}.")
        additions.extend([first, second])
        labels.update(
            answerable=False, expected_behavior="report_conflict", expected_answer=None,
            relevant_ids=[first["id"], second["id"]],
            supporting_ids=[first["id"], second["id"]],
            conflicting_ids=[first["id"], second["id"]],
            conflicting_values=[code, alternative],
            precedence="equal_authority_equal_date",
        )
    elif case == "stale_facts":
        old_date = (reference_date - timedelta(days=365)).isoformat()
        new_date = (reference_date - timedelta(days=1)).isoformat()
        old = make("old_record", f"Effective {old_date}, the access code for {entity} was {alternative}.", old_date)
        current = make("current_record", f"Effective {new_date}, the access code for {entity} changed to {code}, superseding the previous code.", new_date)
        additions.extend([old, current])
        question = f"As of {as_of}, what is the current access code for {entity}?"
        labels.update(relevant_ids=[current["id"]], supporting_ids=[current["id"]], stale_ids=[old["id"]], as_of=as_of)
    elif case == "lexical_mismatch":
        displacement = 1000 + int(fingerprint[12:16], 16) % 2000
        support = make("technical_record", f"The automobile identified as {entity} has an engine displacement of {displacement} cubic centimetres.")
        additions.append(support)
        question = f"What is the motor capacity in cc of the car identified as {entity}?"
        labels.update(
            expected_answer=f"{displacement} cubic centimetres",
            accepted_answers=[str(displacement), f"{displacement} cc", f"{displacement} cubic centimetres"],
            relevant_ids=[support["id"]], supporting_ids=[support["id"]],
            lexical_pairs=[["car", "automobile"], ["motor capacity", "engine displacement"], ["cc", "cubic centimetres"]],
        )
    elif case == "irrelevant_context":
        additions.append(support)
        for number in range(5):
            additions.append(make(
                f"distractor_{number}",
                f"Access code directory discussion for {entity}. "
                f"This document discusses directory layout, access code labels and formatting example {number}; "
                "it does not specify the unit's assigned access code.",
            ))
        labels["irrelevant_ids"] = [document["id"] for document in additions[1:]]

    overlay = {"exclude_ids": sorted(excluded), "add_documents": additions}
    candidate_documents = materialize_overlay(base, overlay)
    candidate_ids = {document["id"] for document in candidate_documents}
    if not set(labels["relevant_ids"]).issubset(candidate_ids):
        raise AssertionError("synthetic relevant evidence is absent")
    source_query = dict(query) if isinstance(query, Mapping) else query
    return {
        "case_id": prefix,
        "case": case,
        "synthetic": True,
        "track": "synthetic_robustness",
        "provenance": {"generator": GENERATOR, "seed": seed, "as_of": as_of, "source_query": source_query},
        "query": {"id": f"{prefix}:query", "text": question, "question": question,
                  "track": "synthetic_robustness", "domain": "synthetic_controlled", "as_of": as_of},
        "documents": candidate_documents,
        "labels": labels,
        "overlay": overlay,
    }


def build_failure_suite(
    documents: Iterable[Mapping[str, Any]] = (), *, seed: int = 0, as_of: str = "2025-01-01"
) -> list[dict[str, Any]]:
    pass                                                               
    base = list(documents)
    return [build_failure_case(case, base, seed=seed, as_of=as_of) for case in CASES]
