                                                                            

                                                                               
                                                                                
   
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

HISTORICAL_QA_SCOPE = "historical_ragtruth_qa_source_faithfulness_v1"
HISTORICAL_QA_FRAMING = "question_full_declared_context_candidate_response_v1"
PROFILE_SHA256 = "02ce9db42a265d3ede798a0c0eacc033ef326923ade7b376b3bba5e35f89480b"
REFERENCE_RUNTIME_SHA256 = "fa1bd42797dbcdb484181cc83fece57fedf739fb94923b2b8a3fd0afb7b1f1ee"
REFERENCE_BUILDER = {"source": "train_decode_inputs_v2.py", "function": "request_body",
                     "source_sha256": "cef8f72b83837772b9b67f61a172769eb064ddb9413a727c38609f5c473ee3c4"}
MAX_REQUEST_BYTES = 1_000_000
MAX_PROFILE_BYTES = 128_000
MAX_REFERENCE_BYTES = 2_000_000
MAX_REFERENCE_TOTAL_BYTES = 8_000_000
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CONTROLS = ("model", "provider", "temperature", "max_tokens", "stream", "reasoning", "response_format", "usage")
_REFERENCES = frozenset(("train_fit_run_v1/calibrator.json", "dev_threshold_run_v2/thresholds.json",
    "late_test_assessment_run_v1/assessment.json", "late_test_assessment_run_v1/complete.json",
    "independent_test_arithmetic_audit_v1.json", "root_test64_raw_envelope_audit_gate_v1.json",
    "heldout_diagnostics_v1.png", "train_decode_inputs_v2.py", "train_fit_v1.py"))
_FROZEN_BLOCKS = ((.3, .125, 8, 1), (.6, 3/13, 13, 3), (.62, .5, 2, 1),
                  (.82, 11/14, 14, 11), (.93, 18/19, 19, 18), (.95, 1., 5, 5))
_THRESHOLDS = {"raw": .8, "calibrated": 11/14, "raw_fixed_control": .5}
_LIMITATION = ("Historical QA source-faithfulness research only. Brier and AUROC worsened versus raw scores; "
              "ECE improved and paired balanced-accuracy intervals included zero. No overall calibration gain "
              "or transfer to current answers, relevance, completeness, or new domains is established.")


class JudgeProfileError(ValueError):
    pass                                                                       


def _require(condition, message):
    if not condition:
        raise JudgeProfileError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise JudgeProfileError("Invalid JSON profile/runtime value") from None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate profile/runtime JSON field")
            result[key] = value
        return result
    def constant(_):
        raise JudgeProfileError("Nonfinite profile/runtime JSON value")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise JudgeProfileError("Invalid profile/runtime JSON") from None


def encode_request(body: Mapping) -> bytes:
    pass                                                                      
    _require(isinstance(body, Mapping), "Judge request must be a mapping")
    raw = _canonical(dict(body))
    _require(len(raw) <= MAX_REQUEST_BYTES, "Judge request exceeds byte bound")
    return raw


def _qa_user(question, full_context, candidate_response):
    return {"query": {"question": question}, "prediction": {"answer": candidate_response, "abstained": False, "citations": []},
            "evidence": [{"doc_id": "supplied_context", "sentence_id": 0, "title": "", "text": full_context}], "missing_citations": []}


def build_historical_qa_request(question: str, full_context: str, candidate_response: str) -> dict:
    pass                                                                          

                                                                                
                                                                                
       
    from .judge import SYSTEM
    _require(all(isinstance(value, str) for value in (question, full_context, candidate_response)), "QA inputs must be strings")
    body = {"model": "minimax/minimax-m3.1-flash",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": _canonical(_qa_user(question, full_context, candidate_response)).decode("utf-8")}],
        "temperature": 0, "max_tokens": 4096, "stream": False, "reasoning": {"effort": "low"},
        "response_format": {"type": "json_object"}, "usage": {"include": True},
        "provider": {"max_price": {"prompt": .15, "completion": .60}, "require_parameters": True, "allow_fallbacks": False, "only": ["fireworks"]}}
    encode_request(body)
    return body


def runtime_contract_from_request(body: Mapping, *, task_scope: str, input_framing_id: str, judgment_schema: Mapping | None = None) -> dict:
    pass                                                                      

                                                                          
                                                                            
                                                                            
                                                                               
                                           
       
    raw = encode_request(body)
    request = _strict(raw)
    _require(isinstance(task_scope, str) and bool(task_scope) and isinstance(input_framing_id, str) and bool(input_framing_id), "Explicit task and input framing are required")
    required = (set(_CONTROLS) - {"reasoning"}) | {"messages"}
    _require(required <= set(request) <= required | {"reasoning"}, "Unexpected or missing judge request controls")
    messages = request["messages"]
    _require(isinstance(messages, list) and len(messages) == 2 and all(isinstance(m, dict) and set(m) == {"role", "content"} and isinstance(m["content"], str) for m in messages)
             and messages[0]["role"] == "system" and messages[1]["role"] == "user", "Judge messages require explicit system/user framing")
    _require(isinstance(request["model"], str) and bool(request["model"]) and type(request["max_tokens"]) is int and request["max_tokens"] > 0
             and type(request["temperature"]) in (int, float) and math.isfinite(request["temperature"])
             and type(request["stream"]) is bool and all(isinstance(request[k], dict) for k in ("provider", "response_format", "usage"))
             and ("reasoning" not in request or isinstance(request["reasoning"], dict)), "Invalid judge runtime controls")
    if input_framing_id == HISTORICAL_QA_FRAMING or task_scope == HISTORICAL_QA_SCOPE:
        _require(input_framing_id == HISTORICAL_QA_FRAMING and task_scope == HISTORICAL_QA_SCOPE, "Historical QA task and framing must be selected together")
        user = _strict(messages[1]["content"])
        try:
            question, context, answer = user["query"]["question"], user["evidence"][0]["text"], user["prediction"]["answer"]
        except (KeyError, IndexError, TypeError):
            raise JudgeProfileError("Historical QA input framing mismatch") from None
        _require(all(isinstance(value, str) for value in (question, context, answer)) and user == _qa_user(question, context, answer)
                 and messages[1]["content"] == _canonical(user).decode("utf-8"), "Historical QA input framing mismatch")
    if judgment_schema is None:
        from .judge import JUDGMENT_SCHEMA
        judgment_schema = JUDGMENT_SCHEMA
    _require(isinstance(judgment_schema, Mapping), "Explicit judgment schema must be a mapping")
    return {"contract_version": 1, **{key: request.get(key) for key in _CONTROLS},
            "rubric_utf8_sha256": _digest(messages[0]["content"].encode("utf-8")), "judgment_schema_sha256": _digest(_canonical(dict(judgment_schema))),
            "task_scope": task_scope, "input_framing_id": input_framing_id}


def require_matching_contract(actual: Mapping, expected: Mapping) -> None:
    pass                                                                        
    _require(isinstance(actual, Mapping) and isinstance(expected, Mapping) and _canonical(dict(actual)) == _canonical(dict(expected)), "Judge runtime/task/input framing does not match historical QA profile")


def _owned_file(path, root, limit):
    candidate = Path(path).absolute()
    _require(candidate.is_relative_to(root) and candidate.resolve() == candidate and candidate.is_file()
             and not candidate.is_symlink() and candidate.stat().st_size <= limit, "Profile reference must be a bounded owned local file")
    for parent in candidate.parents:
        _require(not parent.is_symlink(), "Profile reference cannot traverse a symlink")
        if parent == root:
            break
    return candidate


def _reference(root, name, pin):
    _require(isinstance(name, str) and name in _REFERENCES and "\\" not in name and ":" not in name
             and not PurePosixPath(name).is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")), "Invalid owned profile reference path")
    _require(isinstance(pin, dict) and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int and 0 < pin["bytes"] <= MAX_REFERENCE_BYTES
             and isinstance(pin["sha256"], str) and _HEX.fullmatch(pin["sha256"]), "Invalid bounded profile reference pin")
    path = _owned_file(root / name, root, MAX_REFERENCE_BYTES)
    _require(path.stat().st_size == pin["bytes"] and _digest(path.read_bytes()) == pin["sha256"], "Owned profile reference hash/size mismatch")
    return path


@dataclass(frozen=True)
class HistoricalQAProfile:
    pass                                                                                    
    profile_sha256: str
    runtime_json: bytes
    blocks: tuple[tuple[float, float], ...]
    scope: str

    @property
    def runtime_contract(self) -> dict:
        return _strict(self.runtime_json)


def load_historical_qa_profile(path, *, artifact_root, runtime_contract: Mapping, task_scope: str,
                               expected_sha256: str = PROFILE_SHA256) -> HistoricalQAProfile:
    pass                                                                        

                                                                              
                                                                                 
                                                                                  
       
    try:
        return _load_historical_qa_profile(path, artifact_root=artifact_root, runtime_contract=runtime_contract,
            task_scope=task_scope, expected_sha256=expected_sha256)
    except JudgeProfileError:
        raise
    except (OSError, KeyError, ValueError, TypeError, AttributeError, RecursionError):
        raise JudgeProfileError("Invalid or unavailable owned historical QA profile bundle") from None


def _load_historical_qa_profile(path, *, artifact_root, runtime_contract, task_scope, expected_sha256):
    _require(task_scope == HISTORICAL_QA_SCOPE, "Historical QA scope must be explicitly selected")
    root = Path(artifact_root).absolute()
    _require(root.is_dir() and root.resolve() == root and not root.is_symlink(), "Artifact root must be an owned local directory")
    source = _owned_file(path, root, MAX_PROFILE_BYTES)
    raw = source.read_bytes()
    _require(isinstance(expected_sha256, str) and _HEX.fullmatch(expected_sha256) and _digest(raw) == expected_sha256, "Historical QA profile hash mismatch")
    profile = _strict(raw)
    _require(isinstance(profile, dict) and profile.get("format") == "magra.experimental.historical-QA-source-faithfulness-profile.v1"
             and profile.get("status") == "FROZEN_RESEARCH_PROFILE_POINT_GATES_MET" and profile.get("automatic_generic_judge_loading_enabled") is False
             and profile.get("original_human33_AUDIT_labels_used") is False and profile.get("TEST_retuning_performed") is False, "Unsupported historical QA profile scope/status")
    legacy = profile["runtime_contract"]
    _require(isinstance(legacy, dict) and _digest(_canonical(legacy)) == profile["runtime_contract_sha256"] == REFERENCE_RUNTIME_SHA256
             and legacy.get("request_builder") == REFERENCE_BUILDER, "Historical QA reference runtime/framing pin mismatch")
    expected = {"contract_version": 1, **{key: value for key, value in legacy.items() if key != "request_builder"},
                "task_scope": HISTORICAL_QA_SCOPE, "input_framing_id": HISTORICAL_QA_FRAMING}
    require_matching_contract(runtime_contract, expected)
    blocks = profile["transform"]["blocks"]
    _require(isinstance(blocks, list) and len(blocks) == 6 and profile["thresholds"] == _THRESHOLDS
             and profile["classifier"] == "supported iff probability>=threshold", "Frozen historical QA transform/threshold mismatch")
    for block, expected_block in zip(blocks, _FROZEN_BLOCKS):
        _require(isinstance(block, dict) and set(block) == {"upper_score", "probability", "components", "supported"}
                 and type(block["upper_score"]) in (int, float) and type(block["probability"]) in (int, float)
                 and type(block["components"]) is int and type(block["supported"]) is int
                 and (block["upper_score"], block["probability"], block["components"], block["supported"]) == expected_block, "Frozen six-block historical QA step mismatch")
    provenance = profile["provenance"]
    _require(isinstance(provenance, dict) and set(provenance) == _REFERENCES
             and sum(pin.get("bytes", MAX_REFERENCE_TOTAL_BYTES + 1) for pin in provenance.values() if isinstance(pin, dict)) <= MAX_REFERENCE_TOTAL_BYTES, "Bounded portable profile provenance required")
    references = {name: _reference(root, name, pin) for name, pin in provenance.items()}
    calibrator = _strict(references["train_fit_run_v1/calibrator.json"].read_bytes())
    thresholds = _strict(references["dev_threshold_run_v2/thresholds.json"].read_bytes())
    for artifact in (calibrator, thresholds):
        _require(artifact.get("artifact_hash") == _digest(_canonical({key: value for key, value in artifact.items() if key != "artifact_hash"})), "Portable fitted artifact internal hash mismatch")
    _require(calibrator.get("fit_role") == "TRAIN" and calibrator.get("blocks") == blocks and calibrator.get("threshold_selected") is False
             and thresholds.get("selection_role") == "DEV" and thresholds.get("TRAIN_calibrator_refit") is False and thresholds.get("TEST_settings_frozen") is True
             and thresholds["TEST_settings"]["thresholds"] == _THRESHOLDS
             and thresholds["TEST_settings"]["TRAIN_calibrator_sha256"] == provenance["train_fit_run_v1/calibrator.json"]["sha256"], "Frozen TRAIN fit/DEV thresholds provenance mismatch")
    _require(isinstance(profile.get("scope"), str) and bool(profile["scope"]), "Historical research scope is required")
    return HistoricalQAProfile(expected_sha256, _canonical(expected), tuple((block["upper_score"], block["probability"]) for block in blocks), profile["scope"])


def apply_historical_qa_profile(profile: HistoricalQAProfile, score, *, abstained: bool, actual_provider: str,
                                runtime_contract: Mapping, task_scope: str) -> dict:
    pass                                                                            
    _require(isinstance(profile, HistoricalQAProfile) and task_scope == HISTORICAL_QA_SCOPE, "An explicitly loaded historical QA profile is required")
    declared = profile.runtime_contract
    _require(declared.get("contract_version") == 1 and declared.get("task_scope") == HISTORICAL_QA_SCOPE
             and declared.get("input_framing_id") == HISTORICAL_QA_FRAMING, "Historical profile runtime identity is invalid")
    legacy = {key: value for key, value in declared.items() if key not in {"contract_version", "task_scope", "input_framing_id"}}
    legacy["request_builder"] = REFERENCE_BUILDER
    _require(_digest(_canonical(legacy)) == REFERENCE_RUNTIME_SHA256
             and profile.blocks == tuple((upper, probability) for upper, probability, _, _ in _FROZEN_BLOCKS), "Historical profile runtime/step identity is invalid")
    require_matching_contract(runtime_contract, profile.runtime_contract)
    _require(actual_provider == "Fireworks", "Actual judge provider does not match historical QA profile")
    _require(type(abstained) is bool, "Judge abstained must be boolean")
    if abstained:
        _require(score is None, "Abstained judge score must be unavailable")
        calibrated = None
    else:
        _require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, "Invalid or unavailable raw judge probability")
        calibrated = next((p for upper, p in profile.blocks if score <= upper), profile.blocks[-1][1])
    return {"raw_groundedness_probability": score, "calibrated_groundedness_probability": calibrated,
        "raw_supported_at_profile_threshold": None if score is None else score >= _THRESHOLDS["raw"],
        "calibrated_supported_at_profile_threshold": None if calibrated is None else calibrated >= _THRESHOLDS["calibrated"],
        "raw_supported_at_fixed_0_5": None if score is None else score >= .5,
        "thresholds": dict(_THRESHOLDS), "calibration_applied": calibrated is not None, "abstained": abstained,
        "profile_sha256": profile.profile_sha256, "runtime_contract_sha256": _digest(profile.runtime_json),
        "reference_runtime_contract_sha256": REFERENCE_RUNTIME_SHA256,
        "task_scope": HISTORICAL_QA_SCOPE, "input_framing_id": HISTORICAL_QA_FRAMING,
        "scope": profile.scope, "limitations": _LIMITATION}
