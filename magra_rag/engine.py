                                                                              

from __future__ import annotations

import time

from .retrieval import Retriever
from .index_integrity import effective_query_scope


class MagraEngine:
    def __init__(self, retriever: Retriever, generator=None, *, calibrator=None):
        self.retriever, self.generator, self.calibrator = retriever, generator, calibrator

    def run(self, query: dict, *, retrieval_only=False) -> dict:
        started = time.perf_counter()
        allowed_fields = {"id", "query_id", "question", "text", "allowed_doc_ids", "exclude_doc_ids",
                          "source", "domain", "setting", "track", "as_of", "split", "group_id",
                          "context_doc_ids", "corpus_source", "paraphrase_group"}
        allowed_fields |= {"scenario", "synthetic", "parent_id", "official_split"}
        if set(query) - allowed_fields:
            raise ValueError("Engine requires a public query record without labels")
        query_id = query.get("id", query.get("query_id"))
        question = query.get("question", query.get("text"))
        if not query_id or not isinstance(question, str) or not question.strip():
            raise ValueError("Query requires id and nonempty question")
        scope = effective_query_scope(query)
        retrieved = self.retriever.retrieve(question, **{field: scope[field] for field in
            ("allowed_ids", "exclude_ids", "source", "id_prefix")})
        result = {"query_id": query_id, "ranked_doc_ids": retrieved["ranked_doc_ids"],
                  "evidence": retrieved["evidence"], "context_bytes": retrieved["context_bytes"],
                  "lexical_coverage": retrieved["coverage"], "trace": retrieved["trace"],
                  "retrieval_config": retrieved["config"],
                  "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
                  "generation_latency_ms": None}
        if self.generator is not None and not retrieval_only:
            generation_started = time.perf_counter()
            generation = self.generator.generate(question, retrieved["evidence"], as_of=query.get("as_of"))
            result.update(generation)
            if self.calibrator is not None:
                if str(query_id) in self.calibrator.fit_query_ids:
                    raise ValueError("Cannot evaluate calibration on queries used to fit it")
                raw = result["confidence"]
                result["raw_confidence"] = raw
                result["confidence"] = self.calibrator.predict(raw)
                result["calibration_hash"] = self.calibrator.artifact_hash
                result["confidence_kind"] = "development_calibrated"
                if self.calibrator.should_abstain(raw):
                    result.update(answer="", citations=[], abstained=True,
                                  reason="Below development-calibrated acceptance threshold")
            result["generation_latency_ms"] = (time.perf_counter() - generation_started) * 1000
        result["resources"] = self.retriever.store.budget.check()
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        return result
