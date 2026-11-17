                                                                              

                                                                                
                                                                             
                                                                       
   

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .calibration import Calibrator
from .metrics import (
    abstention_metrics, answer_metrics, brier_score, evidence_metrics,
    expected_calibration_error, paired_bootstrap, probability,
    retrieval_metrics, selective_metrics,
)


def _records(rows: Iterable[Mapping] | str | Path) -> list[dict]:
    if isinstance(rows, (str, Path)):
        with Path(rows).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [dict(row) for row in rows]


def _hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _index(rows: list[dict], *, predictions: bool = False) -> dict[str, dict]:
    result = {}
    for row in rows:
        qid = row.get("query_id", row.get("id")) if predictions else row.get("id", row.get("query_id"))
        if not isinstance(qid, str) or not qid or qid in result:
            raise ValueError("Every query/prediction needs a unique, nonempty string query ID")
        result[qid] = row
    return result


def _stratum(query: Mapping) -> dict:
                                                                              
    synthetic = query.get("synthetic", False)
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be boolean")
    return {
        "domain": str(query.get("domain", "unknown")),
        "source": str(query.get("source", "unknown")),
        "track": str(query.get("track", "natural")),
        "setting": str(query.get("setting", "full_corpus")),
        "synthetic": synthetic,
        "web_mode": str(query.get("web_mode", "none")),
        "scenario": str(query.get("scenario", "standard")),
    }


def _group_key(stratum: Mapping) -> str:
    return json.dumps(stratum, sort_keys=True, separators=(",", ":"))


def _fallback_key(stratum: Mapping) -> str:
    return _group_key({key: value for key, value in stratum.items() if key not in {"domain", "source"}})


def _summary(values: Iterable[float | None], *, total: int | None = None) -> dict:
    all_values = list(values)
    known = [float(value) for value in all_values if value is not None]
    denominator = len(all_values) if total is None else total
    return {"value": sum(known) / len(known) if known else None,
            "count": len(known), "missing": denominator - len(known),
            "status": "available" if known else "unavailable"}


def _ranking(prediction: Mapping) -> list[str]:
    ranking = prediction.get("ranked_doc_ids", prediction.get("ranked_docs", []))
    if not isinstance(ranking, list):
        raise ValueError("ranked_doc_ids must be a list")
    result = []
    for item in ranking:
        if isinstance(item, Mapping):
            item = item.get("doc_id", item.get("id"))
        if not isinstance(item, str) or not item:
            raise ValueError("Ranked documents must have nonempty string IDs")
        result.append(item)
    return result


def _measurement(prediction: Mapping, name: str) -> float | None:
    value = prediction.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _references(query: Mapping):
    if "answers" in query:
        return query["answers"]
    return query.get("answer")


def _score_query(query: Mapping, prediction: Mapping | None, k: int, calibration: Calibrator | None,
                 *, generation_enabled: bool = True) -> dict:
    absent = prediction is None
    pred = prediction or {}
    abstained = pred.get("abstained", False) if generation_enabled else False
    if type(abstained) is not bool:
        raise ValueError("abstained must be boolean")
    answer = pred.get("answer", "") if generation_enabled else ""
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    references = _references(query) if generation_enabled else None
    scores = retrieval_metrics(_ranking(pred), query.get("relevance"), k)
    selected_support = query.get("supporting_evidence", query.get("supporting_facts"))
                                                                             
                                                                                
    selected = [] if absent else pred.get("evidence")
    selection_scores = evidence_metrics(selected or [], selected_support if selected is not None else None)
    selection_names = {
        "citation_precision": "selected_document_precision", "citation_recall": "selected_document_recall",
        "citation_f1": "selected_document_f1", "citation_em": "selected_document_em",
        "evidence_sentence_precision": "selected_sentence_precision",
        "evidence_sentence_recall": "selected_sentence_recall",
        "evidence_sentence_f1": "selected_sentence_f1", "evidence_sentence_em": "selected_sentence_em",
    }
    scores.update({selection_names[name]: value for name, value in selection_scores.items()})
    scores.update(answer_metrics("" if abstained or absent else answer, references))
                                                                              
    if (abstained or absent) and scores["answer_em"] is not None:
        scores.update({key: 0.0 for key in ("answer_em", "answer_f1", "answer_precision", "answer_recall")})
    support = query.get("supporting_evidence", query.get("supporting_facts")) if generation_enabled else None
    scores.update(evidence_metrics(pred.get("citations", []), support))
    if scores["answer_em"] is not None and scores["evidence_sentence_em"] is not None:
        scores["joint_em"] = scores["answer_em"] * scores["evidence_sentence_em"]
        joint_precision = scores["answer_precision"] * scores["evidence_sentence_precision"]
        joint_recall = scores["answer_recall"] * scores["evidence_sentence_recall"]
        scores["joint_f1"] = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) \
            if joint_precision + joint_recall else 0.0
    else:
        scores.update(joint_em=None, joint_f1=None)
    for field in ("latency_ms", "retrieval_latency_ms", "generation_latency_ms", "tokens", "cost_usd", "context_bytes"):
        scores[field] = _measurement(pred, field)
    answerable = query.get("answerable") if generation_enabled else None
    if answerable is not None and type(answerable) is not bool:
        raise ValueError("answerable must be boolean when supplied")
    eligible = references is not None or answerable is not None
    correctness = None
    if answerable is False:
        correctness = False                                                        
    elif scores["answer_em"] is not None:
        correctness = bool(scores["answer_em"])
                                                                   
    selective_eligible = eligible and correctness is not None
    if not generation_enabled:
        raw_confidence = calibrated_confidence = None
    elif pred.get("calibration_hash"):
        if calibration is not None and pred["calibration_hash"] != calibration.artifact_hash:
            raise ValueError("Prediction calibration hash differs from supplied calibration artifact")
        raw_confidence = probability(pred["raw_confidence"]) if pred.get("raw_confidence") is not None else None
        calibrated_confidence = probability(pred["confidence"]) if pred.get("confidence") is not None else None
    else:
        raw_confidence = probability(pred["confidence"]) if pred.get("confidence") is not None else None
        calibrated_confidence = calibration.predict(raw_confidence) \
            if calibration is not None and raw_confidence is not None else None
    return {
        "query_id": query.get("id", query.get("query_id")),
        "group_id": str(query.get("group_id", query.get("id", query.get("query_id")))),
        "stratum": _stratum(query), "split": query.get("split", "unknown"),
        "held_out": bool(query.get("held_out", query.get("heldout", False))),
        "prediction_missing": absent, "abstained": abstained if generation_enabled else None,
        "generation_scoring_enabled": generation_enabled,
        "answerable": answerable, "coverage_eligible": eligible,
        "selective_eligible": selective_eligible,
        "correct": correctness, "confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "calibration_recommends_abstention": calibration.should_abstain(raw_confidence)
        if calibration is not None and raw_confidence is not None else None,
        "metrics": scores,
    }


def _aggregate(rows: list[dict]) -> dict:
    names = sorted({name for row in rows for name in row["metrics"]})
    metrics = {name: _summary(row["metrics"].get(name) for row in rows) for name in names}
    eligible = [row for row in rows if row["selective_eligible"]]
                                                                                     
    selective = selective_metrics([row["correct"] for row in eligible],
                                  [row["abstained"] or row["prediction_missing"] for row in eligible])
    coverage_eligible = [row for row in rows if row["coverage_eligible"]]
    selective["coverage"] = sum(not row["abstained"] and not row["prediction_missing"]
                                for row in coverage_eligible) / len(coverage_eligible) if coverage_eligible else None
    selective["coverage_eligible_count"] = len(coverage_eligible)
    decisions = [row for row in rows if row["answerable"] is not None and not row["prediction_missing"]]
    abstention = abstention_metrics([row["answerable"] for row in decisions],
                                   [row["abstained"] for row in decisions])
    for name, value in selective.items():
        if not name.endswith("count"):
            denominator = len(coverage_eligible) if name == "coverage" else selective["answered_count"]
            metrics[name] = {"value": value, "count": denominator,
                             "missing": len(rows) - denominator,
                             "status": "available" if value is not None else "unavailable"}
    abstention_denominators = {
        "abstention_precision": sum(row["abstained"] for row in decisions),
        "missing_evidence_abstention_recall": abstention["missing_evidence_count"],
        "unsupported_answer_rate": abstention["missing_evidence_count"],
        "false_abstention_rate": abstention["answerable_count"],
        "answerability_decision_accuracy": len(decisions),
    }
    for name, value in abstention.items():
        if not name.endswith("count"):
            denominator = abstention_denominators[name]
            metrics[name] = {"value": value, "count": denominator,
                             "missing": len(rows) - denominator,
                             "status": "available" if value is not None else "unavailable"}
    calibration_counts = {}
    for field, prefix in (("confidence", "raw_"), ("calibrated_confidence", "calibrated_")):
        labeled = [row for row in rows if row[field] is not None and row["correct"] is not None
                   and not row["prediction_missing"] and not row["abstained"]]
        probabilities = [row[field] for row in labeled]
        outcomes = [row["correct"] for row in labeled]
        calibration_counts[prefix + "count"] = len(labeled)
        for name, value in (("brier", brier_score(probabilities, outcomes)),
                            ("ece", expected_calibration_error(probabilities, outcomes))):
            metrics[prefix + name] = {"value": value, "count": len(labeled),
                                      "missing": len(rows) - len(labeled),
                                      "status": "available" if value is not None else "unavailable"}
    return {"stratum": rows[0]["stratum"], "query_count": len(rows),
            "prediction_count": sum(not row["prediction_missing"] for row in rows),
            "metrics": metrics, "selective_counts": selective,
            "abstention_counts": {**abstention, "decision_count": len(decisions)},
            "calibration_counts": calibration_counts}


def evaluate_run(
    queries: Iterable[Mapping] | str | Path, predictions: Iterable[Mapping] | str | Path,
    *, k: int = 10, calibration: Calibrator | None = None, run_metadata: Mapping | None = None,
) -> dict:
    pass                                                                      

                                                                                
                                                                            
                                                                             
                                                                             
                                                                               
                                                                               
                                                                                
                                                                    
       
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    labels, outputs = _records(queries), _records(predictions)
    generation_enabled = (run_metadata or {}).get("generation") is not False
    query_map, output_map = _index(labels), _index(outputs, predictions=True)
    unknown = set(output_map) - set(query_map)
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} IDs outside the selected query set")
    if calibration is not None:
        query_groups = {str(query.get("group_id", qid)) for qid, query in query_map.items()}
        if set(calibration.fit_query_ids) & set(query_map) or set(calibration.fit_group_ids) & query_groups:
            raise ValueError("Evaluation queries overlap calibration development fit queries/groups")
    prediction_calibration_hashes = {row["calibration_hash"] for row in outputs if row.get("calibration_hash")}
    if len(prediction_calibration_hashes) > 1:
        raise ValueError("A run cannot mix confidence calibration artifacts")
    grouped = defaultdict(list)
    rows = []
    for qid, query in sorted(query_map.items()):
        row = _score_query(query, output_map.get(qid), k, calibration,
                           generation_enabled=generation_enabled)
        rows.append(row)
        grouped[_group_key(row["stratum"])].append(row)
    domains = sorted({row["stratum"]["domain"] for row in rows} - {"unknown"})
    return {
        "schema_version": 1, "query_count": len(labels), "prediction_count": len(outputs),
        "missing_predictions": sorted(set(query_map) - set(output_map)),
        "splits": sorted({str(query.get("split", "unknown")) for query in labels}),
        "domains": domains, "k": k,
        "generation_scoring_enabled": generation_enabled,
        "provenance": {"labels_hash": _hash([query_map[qid] for qid in sorted(query_map)]),
                       "predictions_hash": _hash([output_map[qid] for qid in sorted(output_map)]),
                       "calibration_hash": calibration.artifact_hash if calibration
                       else next(iter(prediction_calibration_hashes), None)},
        "run_metadata": dict(run_metadata or {}),
        "groups": {key: _aggregate(group) for key, group in sorted(grouped.items())},
        "per_query": rows,
        "limitations": [
            "No overall metric pools domains, settings, natural/synthetic scenarios, or web tracks.",
            "Unjudged retrieval documents receive zero gain; report judged coverage alongside relevance.",
            "Citation scores measure annotated evidence identity, not semantic entailment or factual truth.",
            "Calibration correctness uses answer exact match and includes only emitted answers.",
            "Public benchmark contamination is not excluded by private query splits.",
        ],
        "generality": {"required_domains": 5, "observed_domains": len(domains),
                       "coverage_sufficient": len(domains) >= 5,
                       "claim_supported": False,
                       "reason": "Coverage alone does not establish an improvement against feasible baselines."},
    }


def _feasible(report: Mapping, group: Mapping, constraints: Mapping[str, float]) -> tuple[bool, list[str]]:
    failures = []
    resources = report.get("run_metadata", {}).get("resources", {})
    for name, limit in constraints.items():
        if name in group["metrics"]:
            measurement = group["metrics"][name]
            value = measurement["value"] if not measurement["missing"] else None
        else:
            value = resources.get(name)
        if value is None:
            failures.append(f"{name}: missing measurement")
        elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > limit:
            failures.append(f"{name}: exceeds limit or invalid")
    return not failures, failures


def select_baselines(
    development_reports: Mapping[str, Mapping], baseline_names: Iterable[str], *,
    quality_metric: str = "answer_f1", constraints: Mapping[str, float] | None = None,
) -> dict:
    pass                                                                       

                                                                          
                                                                                
                                                                               
       
    names = list(baseline_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide unique baseline names")
    limits = dict(constraints or {})
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("Feasibility limits must be finite and nonnegative")
    for name in names:
        if name not in development_reports:
            raise ValueError(f"Missing development report for baseline {name}")
        if development_reports[name].get("splits") != ["dev"]:
            raise ValueError("Baseline selection requires reports exclusively from split='dev'")
    label_hashes = {development_reports[name].get("provenance", {}).get("labels_hash") for name in names}
    if None in label_hashes or len(label_hashes) != 1:
        raise ValueError("Development baselines must be evaluated against the same labeled query set")
    keys = sorted({key for name in names for key in development_reports[name]["groups"]})
    result = {}
    for key in keys:
        options, excluded = [], {}
        for name in names:
            report = development_reports[name]
            group = report["groups"].get(key)
            if group is None:
                excluded[name] = ["missing development stratum"]
                continue
            quality = group["metrics"].get(quality_metric, {})
            feasible, failures = _feasible(report, group, limits)
            if group["query_count"] != group["prediction_count"]:
                failures.append("incomplete development predictions")
            if quality.get("value") is None or quality.get("missing"):
                failures.append("missing development quality labels or scores")
            if not feasible or failures:
                excluded[name] = failures
                continue
            cost = group["metrics"].get("cost_usd", {}).get("value")
            options.append((-quality["value"], cost if cost is not None else math.inf, name))
        options.sort()
        result[key] = {"baseline": options[0][2] if options else None,
                       "selection_split": "dev", "quality_metric": quality_metric,
                       "eligible": [option[2] for option in options], "excluded": excluded,
                       "constraints": limits, "status": "selected" if options else "no_feasible_baseline"}
                                                                             
                                                                             
                                                                                
    compatible = defaultdict(list)
    for key in keys:
        reference_group = next(development_reports[name]["groups"][key] for name in names
                               if key in development_reports[name]["groups"])
        compatible[_fallback_key(reference_group["stratum"])].append(key)
    fallbacks = {}
    for compatibility_key, group_keys in compatible.items():
        options, macro_scores, excluded = [], {}, {}
        for name in names:
            scores, costs, failures = [], [], []
            for key in group_keys:
                selection = result[key]
                if name not in selection["eligible"]:
                    failures.extend(selection["excluded"].get(name, ["ineligible development stratum"]))
                    continue
                group = development_reports[name]["groups"][key]
                scores.append(group["metrics"][quality_metric]["value"])
                cost = group["metrics"].get("cost_usd", {}).get("value")
                costs.append(cost)
            if failures or len(scores) != len(group_keys):
                excluded[name] = failures or ["missing development strata"]
                continue
            macro = sum(scores) / len(scores)
            macro_cost = sum(costs) / len(costs) if all(cost is not None for cost in costs) else math.inf
            macro_scores[name] = macro
            options.append((-macro, macro_cost, name))
        options.sort()
        fallbacks[compatibility_key] = {
            "baseline": options[0][2] if options else None, "selection_split": "dev",
            "quality_metric": quality_metric, "policy": "dev_macro_for_unseen_source",
            "development_groups": group_keys, "development_macro_quality": macro_scores,
            "labels_hash": next(iter(label_hashes)), "constraints": limits,
            "eligible": [option[2] for option in options], "excluded": excluded,
            "status": "selected" if options else "no_feasible_baseline",
        }
    result["__heldout_fallbacks__"] = fallbacks
    return result


def compare_runs(
    queries: Iterable[Mapping] | str | Path,
    runs: Mapping[str, Iterable[Mapping] | str | Path], *,
    baseline_names: Iterable[str], candidate_name: str,
    development_reports: Mapping[str, Mapping], quality_metric: str = "answer_f1",
    bootstrap_samples: int = 2000, seed: int = 0,
    k: int = 10,
    constraints: Mapping[str, float] | None = None,
    run_metadata: Mapping[str, Mapping] | None = None,
) -> dict:
    pass                                                                        

                                                                               
                                                                                
                                                                               
                                                         
       
    labels = _records(queries)
    if not labels or any(row.get("split") not in {"eval", "test", "locked_eval", "evaluation"} for row in labels):
        raise ValueError("Comparisons require explicitly held-out evaluation/test queries")
    names = list(baseline_names)
    if candidate_name in names or candidate_name not in runs:
        raise ValueError("Candidate must be present and distinct from baselines")
    if any(name not in runs for name in names):
        raise ValueError("Every declared baseline requires an evaluation run")
    selected = select_baselines(development_reports, names, quality_metric=quality_metric, constraints=constraints)
                                                                              
                                                                        
    eval_ids = {row.get("id", row.get("query_id")) for row in labels}
    eval_groups = {str(row.get("group_id", row.get("id", row.get("query_id")))) for row in labels}
    for name in names:
        dev_rows = development_reports[name]["per_query"]
        if eval_ids & {row["query_id"] for row in dev_rows} or eval_groups & {row["group_id"] for row in dev_rows}:
            raise ValueError("Development baseline selection overlaps held-out IDs or duplicate groups")
    reports = {name: evaluate_run(labels, runs[name], k=k, run_metadata=(run_metadata or {}).get(name))
               for name in [candidate_name, *names]}
    candidate = reports[candidate_name]
    comparisons = {}
    for key, group in candidate["groups"].items():
        choice = selected.get(key)
        if choice is None:
            choice = selected["__heldout_fallbacks__"].get(_fallback_key(group["stratum"]))
        if choice is None or choice["baseline"] is None:
            comparisons[key] = {"stratum": group["stratum"], "status": "no_dev_selected_baseline"}
            continue
        baseline_name = choice["baseline"]
        baseline = reports[baseline_name]
        candidate_rows = {row["query_id"]: row for row in candidate["per_query"] if _group_key(row["stratum"]) == key}
        baseline_rows = {row["query_id"]: row for row in baseline["per_query"] if _group_key(row["stratum"]) == key}
        ids = sorted(candidate_rows)
        intervals = {}
        for metric in dict.fromkeys((quality_metric, "cost_usd", "latency_ms", "tokens")):
            paired_ids = [qid for qid in ids if candidate_rows[qid]["metrics"].get(metric) is not None
                          and baseline_rows[qid]["metrics"].get(metric) is not None]
            intervals[metric] = paired_bootstrap(
                [candidate_rows[qid]["metrics"][metric] for qid in paired_ids],
                [baseline_rows[qid]["metrics"][metric] for qid in paired_ids],
                samples=bootstrap_samples, seed=seed,
                cluster_ids=[candidate_rows[qid]["group_id"] for qid in paired_ids],
            )
            intervals[metric]["missing_pairs"] = len(ids) - len(paired_ids)
        candidate_feasible, failures = _feasible(candidate, group, dict(constraints or {}))
        comparisons[key] = {
            "stratum": group["stratum"], "status": "measured",
            "baseline": baseline_name, "selection": choice, "paired_deltas": intervals,
            "candidate_feasible_under_supplied_constraints": candidate_feasible,
            "candidate_feasibility_failures": failures,
            "complete_predictions": group["prediction_count"] == group["query_count"]
            and baseline["groups"][key]["prediction_count"] == group["query_count"],
        }
        quality_ci = intervals[quality_metric]
        cost_ci = intervals["cost_usd"]
        comparisons[key]["observations"] = {
            "quality_improvement_ci_above_zero": quality_ci["ci_low"] > 0
            if quality_ci["ci_low"] is not None else None,
            "cost_difference_ci_at_or_below_zero": cost_ci["ci_high"] <= 0
            if cost_ci["ci_high"] is not None else None,
            "cost_reduction_ci_below_zero": cost_ci["ci_high"] < 0
            if cost_ci["ci_high"] is not None else None,
            "quality_difference_ci_at_or_above_zero": quality_ci["ci_low"] >= 0
            if quality_ci["ci_low"] is not None else None,
        }
    return {
        "schema_version": 1, "candidate": candidate_name, "quality_metric": quality_metric,
        "baseline_selection": selected, "groups": comparisons,
        "evaluation_provenance": {name: report["provenance"] for name, report in reports.items()},
        "domains": candidate["domains"], "generality_claim_supported": False,
        "acceptance_status": "not_assessed",
        "limitations": [
            "Intervals are paired percentile bootstrap estimates, not multiple-comparison-corrected hypothesis tests.",
            "Each stratum must be assessed separately; no pooled Hotpot settings or web/synthetic tracks.",
            "A baseline is strongest only among the declared, measured, feasible development runs.",
            "Generality additionally requires five source domains, a domain held out from adaptation, complete runs, resource measurements, and a prespecified acceptance rule.",
        ],
    }


def _binary_judge_scores(labels: list[int], predicted: list[int], probabilities: list[float]) -> dict:
    if not labels:
        return {name: None for name in ("accuracy", "balanced_accuracy", "precision", "recall",
                                        "f1", "cohen_kappa", "auroc", "brier", "ece")}
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    positives, negatives, n = tp + fn, tn + fp, len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else None
    accuracy = (tp + tn) / n
    expected_agreement = (positives * (tp + fp) + negatives * (tn + fn)) / (n * n)
    auroc = None
    if positives and negatives:
                                                                             
        ordered = sorted(zip(probabilities, labels))
        positive_rank_sum, start = 0.0, 0
        while start < n:
            stop = start + 1
            while stop < n and ordered[stop][0] == ordered[start][0]:
                stop += 1
            mean_rank = (start + 1 + stop) / 2
            positive_rank_sum += mean_rank * sum(y for _, y in ordered[start:stop])
            start = stop
        auroc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + tn / negatives) / 2 if positives and negatives else None,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall else 0.0,
        "cohen_kappa": (accuracy - expected_agreement) / (1 - expected_agreement)
        if expected_agreement != 1 else None,
        "auroc": auroc, "brier": brier_score(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _judge_provenance(humans: Mapping, judgments: Mapping) -> dict:
    return {
        "dataset_hash": _hash([humans[qid] for qid in sorted(humans)]),
        "split_hash": _hash([{"query_id": qid, "split": humans[qid]["split"]}
                             for qid in sorted(humans)]),
        "predictions_hash": _hash([judgments[qid] for qid in sorted(judgments)]),
    }


def _judge_inputs(human_rows, predicted_judgments, *, dev_only: bool = False):
    humans, judgments = _records(human_rows), _records(predicted_judgments)
    human_map, judge_map = _index(humans, predictions=True), _index(judgments, predictions=True)
    if set(judge_map) - set(human_map):
        raise ValueError("Judgments include IDs without human audit labels")
    subsets, groups = {"dev": [], "eval": []}, {"dev": set(), "eval": set()}
    for qid, row in human_map.items():
        split = row.get("split")
        if dev_only and split != "dev":
            raise ValueError("Judge fitting is restricted to split='dev'; evaluation rows are forbidden")
        if split not in {"dev", "eval", "test", "locked_eval", "evaluation"}:
            raise ValueError("Human audits require explicit dev/evaluation splits")
        split = "dev" if split == "dev" else "eval"
        label = row.get("label")
        if not isinstance(label, (bool, int)) or label not in (0, 1):
            raise ValueError("Human audit labels must be binary")
        group = row.get("group_id", qid)
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Human audit group IDs must be nonempty strings")
        if dev_only and group in groups[split]:
            raise ValueError("Judge fitting requires unique duplicate/paraphrase group IDs")
        groups[split].add(group)
        pred = judge_map.get(qid)
        if pred is not None:
            if "split" in pred:
                prediction_split = pred["split"]
                if (prediction_split not in {"dev", "eval", "test", "locked_eval", "evaluation"}
                        or ("dev" if prediction_split == "dev" else "eval") != split):
                    raise ValueError("Judge prediction split differs from its human audit split")
            if "group_id" in pred and pred["group_id"] != group:
                raise ValueError("Judge prediction group differs from its human audit group")
            if type(pred.get("abstained", False)) is not bool:
                raise ValueError("Judge abstained must be boolean")
        score = probability(pred.get("groundedness_probability")) \
            if pred is not None and not pred.get("abstained", False) else None
        subsets[split].append((qid, int(label), score))
    if groups["dev"] & groups["eval"]:
        raise ValueError("Human audit duplicate groups cross dev/evaluation boundary")
    return human_map, judge_map, subsets


def _judge_identity(judgments: Mapping) -> dict:
    pass                                                                           
    fields = ("judge_version", "model", "model_id", "model_revision", "model_sha256",
              "rubric_sha256", "runtime_sha256", "runtime", "reasoning_effort",
              "request_runtime_sha256",
              "max_tokens", "temperature")
    rows = list(judgments.values())
    identity = {}
    for field in fields:
                                                                             
                                                                            
        applicable = ([row for row in rows if row.get("request_sent") is not False]
                      if field == "request_runtime_sha256" else rows)
        provided = [row[field] for row in applicable if field in row]
        if not provided:
            continue
        if len(provided) != len(applicable) or any(value != provided[0] for value in provided[1:]):
            raise ValueError(f"Judge identity {field} must be present and consistent in every judgment")
        value = provided[0]
        if field.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"Judge identity {field} must be a lowercase SHA-256 digest")
        identity[field] = value
                                                                             
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def _judge_counts(subsets: Mapping, min_per_class: int, min_coverage: float):
    counts, failures = {}, []
    for split, rows in subsets.items():
        counts[split] = {"total": len(rows), "judged": sum(score is not None for _, _, score in rows),
                         "negative": sum(y == 0 for _, y, score in rows if score is not None),
                         "positive": sum(y == 1 for _, y, score in rows if score is not None)}
        counts[split]["coverage"] = counts[split]["judged"] / len(rows) if rows else None
        if min(counts[split]["negative"], counts[split]["positive"]) < min_per_class:
            failures.append(f"{split}: insufficient judged examples in each human class")
        if counts[split]["coverage"] is None or counts[split]["coverage"] < min_coverage:
            failures.append(f"{split}: insufficient judge coverage")
    return counts, failures


def _judge_requirements(min_per_class: int, min_coverage: float) -> float:
    if isinstance(min_per_class, bool) or not isinstance(min_per_class, int) or min_per_class < 1:
        raise ValueError("min_per_class must be a positive integer")
    return probability(min_coverage)


def _judge_dev_parameters(humans: Mapping, judgments: Mapping, dev_rows):
    pass                                                                  
    from .calibration import fit_calibrator

    dev = sorted((qid, y, score) for qid, y, score in dev_rows if score is not None)
    dev_ids = {qid for qid, _, _ in dev}
    fit_provenance = _judge_provenance(
        {qid: humans[qid] for qid in dev_ids}, {qid: judgments[qid] for qid in dev_ids})
    calibrator = fit_calibrator(
        [{"query_id": qid, "split": "dev", "confidence": score, "correct": y} for qid, y, score in dev],
        provenance=fit_provenance, target_accuracy=0.0, min_accepted=1,
    )
    dev_probabilities = [calibrator.predict(score) for _, _, score in dev]
    dev_labels = [y for _, y, _ in dev]
    options = []
    for threshold in sorted({0.0, 1.000000000001, *dev_probabilities}):
        metrics = _binary_judge_scores(dev_labels, [int(p >= threshold) for p in dev_probabilities], dev_probabilities)
        options.append((metrics["balanced_accuracy"], -abs(threshold - 0.5), -threshold, threshold, metrics))
    _, _, _, threshold, dev_metrics = max(options, key=lambda option: option[:3])
    locked_hash = _hash([{"calibration_hash": calibrator.artifact_hash,
                          "threshold": threshold, "provenance": fit_provenance}])
    return calibrator, {"threshold": threshold, "calibration_artifact": calibrator.to_dict(),
                        "dev_metrics": dev_metrics, "locked_judge_hash": locked_hash}


def _judge_fit_digest(document: Mapping) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def fit_judge_dev(
    human_rows: Iterable[Mapping] | str | Path,
    predicted_judgments: Iterable[Mapping] | str | Path,
    *, min_per_class: int = 5, min_coverage: float = 0.8,
) -> dict:
    pass                                                                    

                                                                          
                                                                                
                                                                              
                                                                          
                                                                  
       
    required_coverage = _judge_requirements(min_per_class, min_coverage)
    humans, judgments, subsets = _judge_inputs(human_rows, predicted_judgments, dev_only=True)
    identity = _judge_identity(judgments)
    counts, failures = _judge_counts({"dev": subsets["dev"]}, min_per_class, required_coverage)
    if failures:
        raise ValueError("Cannot freeze judge DEV fit: " + "; ".join(failures))
    _, parameters = _judge_dev_parameters(humans, judgments, subsets["dev"])
    document = {"schema_version": 1, "status": "dev_frozen", "fit_split": "dev",
                "use_for_acceptance": False, "provenance": _judge_provenance(humans, judgments),
                "judge_identity": identity, "counts": counts["dev"],
                "fit_query_ids": sorted(humans),
                "fit_group_ids": sorted(row.get("group_id", qid) for qid, row in humans.items()),
                "requirements": {"min_per_class": min_per_class, "min_coverage": required_coverage},
                **parameters}
    document["artifact_hash"] = _judge_fit_digest(document)
    return document


def _load_judge_dev_fit(artifact: Mapping | str | Path) -> dict:
    if isinstance(artifact, (str, Path)):
        with Path(artifact).open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    if not isinstance(artifact, Mapping):
        raise ValueError("Frozen judge DEV fit must be a JSON object")
    document = dict(artifact)
    actual = document.pop("artifact_hash", None)
    if not isinstance(actual, str) or actual != _judge_fit_digest(document):
        raise ValueError("Frozen judge DEV fit artifact hash does not match its contents")
    if (document.get("schema_version") != 1 or document.get("status") != "dev_frozen"
            or document.get("fit_split") != "dev" or document.get("use_for_acceptance") is not False):
        raise ValueError("Unsupported or non-development frozen judge fit")
    document["artifact_hash"] = actual
    return document


def evaluate_judge_calibration(
    human_rows: Iterable[Mapping] | str | Path,
    predicted_judgments: Iterable[Mapping] | str | Path,
    *, min_per_class: int = 5, min_balanced_accuracy: float = 0.8,
    min_coverage: float = 0.8,
    frozen_dev_fit: Mapping | str | Path | None = None,
) -> dict:
    pass                                                                       

                                                                              
                                                           
                                                                              
                                                                           

                                                                             
                                                                            
                                                                             
                                                                             
                                                                              
                                                                             
                                                                          
       
    required_coverage = _judge_requirements(min_per_class, min_coverage)
    required_agreement = probability(min_balanced_accuracy)
    frozen = _load_judge_dev_fit(frozen_dev_fit) if frozen_dev_fit is not None else None
    human_map, judge_map, subsets = _judge_inputs(human_rows, predicted_judgments)
    identity = _judge_identity(judge_map)
    if frozen is not None:
        dev_ids = {qid for qid, _, _ in subsets["dev"]}
        recomputed = fit_judge_dev(
            [human_map[qid] for qid in sorted(dev_ids)],
            [judge_map[qid] for qid in sorted(dev_ids) if qid in judge_map],
            min_per_class=min_per_class, min_coverage=required_coverage)
        if frozen != recomputed or frozen["judge_identity"] != identity:
            raise ValueError("Frozen judge DEV fit differs from current DEV inputs, policy, or judge identity")
    counts, failures = _judge_counts(subsets, min_per_class, required_coverage)
    report = {"schema_version": 1, "status": "insufficient_audit" if failures else "evaluated",
              "recommend_abstention": True, "use_for_acceptance": False,
              "provenance": _judge_provenance(human_map, judge_map), "counts": counts, "failures": failures,
              "fit_locked": frozen is not None, "frozen_dev_fit_hash": frozen["artifact_hash"] if frozen else None,
              "judge_identity": identity,
              "threshold": None, "calibration_artifact": None,
              "dev_metrics": None, "evaluation_metrics": None,
              "limitations": ["Human labels and their independence must be established by the audit owner.",
                              "The threshold is fitted on dev; evaluation is a one-time held-out audit.",
                              "A frozen fit verifies input/parameter bindings; its hash does not prove persistence chronology.",
                              "Point estimates with small audit sets do not establish broad judge reliability."]}
    if frozen is not None:
        report.update({field: frozen[field] for field in
                       ("threshold", "calibration_artifact", "dev_metrics", "locked_judge_hash")})
    if failures:
        return report
    calibrator, parameters = _judge_dev_parameters(human_map, judge_map, subsets["dev"])
    threshold = parameters["threshold"]
    evaluation = [(qid, y, score) for qid, y, score in subsets["eval"] if score is not None]
    probabilities = [calibrator.predict(score) for _, _, score in evaluation]
    labels = [y for _, y, _ in evaluation]
    evaluation_metrics = _binary_judge_scores(labels, [int(p >= threshold) for p in probabilities], probabilities)
    if evaluation_metrics["balanced_accuracy"] < required_agreement:
        failures.append("eval: held-out balanced accuracy below the configured requirement")
    report.update(status="audit_failed" if failures else "audit_passed",
                  recommend_abstention=bool(failures), use_for_acceptance=not failures,
                  evaluation_metrics=evaluation_metrics, **parameters)
    return report
