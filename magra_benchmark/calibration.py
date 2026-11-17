                                                                                

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

from .metrics import brier_score, expected_calibration_error, probability


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _validate_provenance(provenance: Mapping) -> dict:
    required = ("dataset_hash", "split_hash", "predictions_hash")
    for field in required:
        value = provenance.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"provenance.{field} must be a lowercase SHA-256 digest")
    return dict(provenance)


@dataclass(frozen=True)
class Calibrator:
    pass                                                                            

    _serialized: str

    @property
    def artifact_hash(self) -> str:
        return self.to_dict()["artifact_hash"]

    @property
    def fit_query_ids(self) -> tuple[str, ...]:
        return tuple(self.to_dict()["fit_query_ids"])

    @property
    def fit_group_ids(self) -> tuple[str, ...]:
        return tuple(self.to_dict().get("fit_group_ids", self.fit_query_ids))

    @property
    def threshold(self) -> float:
        return self.to_dict()["threshold"]

    def to_dict(self) -> dict:
        return json.loads(self._serialized)

    def predict(self, confidence: float) -> float:
        score = probability(confidence)
        blocks = self.to_dict()["blocks"]
        index = min(bisect_left([block[0] for block in blocks], score), len(blocks) - 1)
        return blocks[index][1]

    def should_abstain(self, confidence: float) -> bool:
        return self.predict(confidence) < self.threshold

    def save(self, path: str | Path) -> None:
        pass                                                            
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n")


def fit_calibrator(
    records: Iterable[Mapping], *, provenance: Mapping, output_path: str | Path | None = None,
    target_accuracy: float = 0.9, min_accepted: int = 10,
) -> Calibrator:
    pass                                                                       

                                                                               
                                                                         
                                                                               
                                                                               
                                                                                 
       
    origin = _validate_provenance(provenance)
    target = probability(target_accuracy)
    if isinstance(min_accepted, bool) or not isinstance(min_accepted, int) or min_accepted < 1:
        raise ValueError("min_accepted must be a positive integer")
    rows = list(records)
    if not rows:
        raise ValueError("Cannot calibrate without development examples")
    seen, fit_groups, parsed = set(), set(), []
    for row in rows:
        if row.get("split") != "dev":
            raise ValueError("Calibration fitting is restricted to split='dev'")
        qid = row.get("query_id", row.get("id"))
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Calibration requires nonempty unique string query IDs")
        seen.add(qid)
        fit_groups.add(str(row.get("group_id", qid)))
        correct = row.get("correct")
        if not isinstance(correct, (bool, int)) or correct not in (0, 1):
            raise ValueError("Calibration requires binary labeled correctness")
        parsed.append((probability(row.get("confidence")), int(correct)))
                                                                             
    grouped = {}
    for score, correct in parsed:
        total, count = grouped.get(score, (0, 0))
        grouped[score] = (total + correct, count + 1)
    blocks = []
    for score, (total, count) in sorted(grouped.items()):
        blocks.append([score, total, count])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([right[0], left[1] + right[1], left[2] + right[2]])
    fitted = [[upper, total / count, count] for upper, total, count in blocks]
    ids = sorted(seen)
    document = {"version": 1, "method": "isotonic_pava", "fit_split": "dev",
                "provenance": origin, "fit_query_ids": ids, "fit_group_ids": sorted(fit_groups),
                "query_ids_hash": _digest(ids),
                "fit_count": len(rows), "blocks": fitted, "threshold": 1.000000000001,
                "target_accuracy": target, "min_accepted": min_accepted}
    provisional = Calibrator(json.dumps(document))
    calibrated = [provisional.predict(score) for score, _ in parsed]
    outcomes = [correct for _, correct in parsed]
    threshold = document["threshold"]
    for value in sorted(set(calibrated)):
        accepted = [correct for score, correct in zip(calibrated, outcomes) if score >= value]
        if len(accepted) >= min_accepted and sum(accepted) / len(accepted) >= target:
            threshold = value
            break
    document["threshold"] = threshold
    document["fit_diagnostics"] = {
        "scope": "in_sample_dev_only", "raw_brier": brier_score([score for score, _ in parsed], outcomes),
        "calibrated_brier": brier_score(calibrated, outcomes),
        "raw_ece": expected_calibration_error([score for score, _ in parsed], outcomes),
        "calibrated_ece": expected_calibration_error(calibrated, outcomes),
        "accepted": sum(score >= threshold for score in calibrated),
    }
    document["artifact_hash"] = _digest(document)
    artifact = _from_dict(document)
    if output_path is not None:
        artifact.save(output_path)
    return artifact


def _from_dict(document: Mapping, expected_hash: str | None = None) -> Calibrator:
    doc = dict(document)
    actual = doc.pop("artifact_hash", None)
    if not isinstance(actual, str) or actual != _digest(doc) or (expected_hash is not None and actual != expected_hash):
        raise ValueError("Calibration artifact hash does not match its contents or pinned hash")
    if doc.get("version") != 1 or doc.get("method") != "isotonic_pava" or doc.get("fit_split") != "dev":
        raise ValueError("Unsupported or non-development calibration artifact")
    _validate_provenance(doc.get("provenance", {}))
    ids = doc.get("fit_query_ids", [])
    if (not ids or any(not isinstance(qid, str) or not qid for qid in ids)
            or len(set(ids)) != len(ids) or ids != sorted(ids)
            or doc.get("query_ids_hash") != _digest(ids) or doc.get("fit_count") != len(ids)):
        raise ValueError("Invalid calibration fit query IDs")
    groups = doc.get("fit_group_ids", ids)
    if (not groups or any(not isinstance(group, str) or not group for group in groups)
            or len(set(groups)) != len(groups) or groups != sorted(groups)):
        raise ValueError("Invalid calibration fit group IDs")
    blocks = doc.get("blocks", [])
    if not blocks:
        raise ValueError("Calibration artifact has no fitted blocks")
    previous_upper, previous_probability = -1.0, -1.0
    count = 0
    for block in blocks:
        if not isinstance(block, list) or len(block) != 3:
            raise ValueError("Malformed calibration block")
        upper, value, n = block
        probability(upper)
        probability(value)
        if upper <= previous_upper or value < previous_probability or type(n) is not int or n < 1:
            raise ValueError("Calibration blocks must have increasing bounds and monotone probabilities")
        previous_upper, previous_probability = upper, value
        count += n
    if count != len(ids):
        raise ValueError("Calibration block counts do not match fit count")
    threshold = doc.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1.000000000001:
        raise ValueError("Invalid calibration threshold")
    probability(doc.get("target_accuracy"))
    doc["artifact_hash"] = actual
    return Calibrator(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False))


def load_calibrator(path: str | Path, *, expected_hash: str | None = None) -> Calibrator:
    pass                                                                         
    with Path(path).open(encoding="utf-8") as handle:
        return _from_dict(json.load(handle), expected_hash)
