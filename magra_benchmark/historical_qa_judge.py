                                                                            

                                                                          
                                                                               
   
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math

from magra_rag.generation import OpenRouterM3, PaidHTTPError, RateLimitError, IncompleteResponseError
from .judge import JudgeError, M3GroundednessJudge, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, _hash, _json, _text
from . import judge_profiles as profiles

_MODEL = "minimax/minimax-m3.1-flash"
_STEP = ((.3, .125), (.6, 3/13), (.62, .5), (.82, 11/14), (.93, 18/19), (.95, 1.))


def _check(condition, message):
    if not condition:
        raise JudgeError(message)


class HistoricalQASourceFaithfulnessJudge:
    pass                                                                       

                                                                         
                                                                            
                                                                             
                                                             
       

    def __init__(self, client: OpenRouterM3, profile: profiles.HistoricalQAProfile, *,
                 max_input_bytes: int = MAX_INPUT_BYTES, max_output_bytes: int = MAX_OUTPUT_BYTES):
        self._validate_client(client)
        _check(isinstance(profile, profiles.HistoricalQAProfile), "An explicitly loaded historical QA profile is required")
        self._sender = M3GroundednessJudge(client, max_input_bytes=max_input_bytes, max_output_bytes=max_output_bytes)
        self.client, self.profile = client, profile
        self.max_input_bytes, self.max_output_bytes = max_input_bytes, max_output_bytes
        self._validate_profile()

    @staticmethod
    def _validate_client(client):
        _check(isinstance(client, OpenRouterM3), "Historical QA judging requires an explicit OpenRouterM3 client")
        _check(getattr(client, "model_id", None) == _MODEL and type(client.max_tokens) is int and client.max_tokens == 4096
               and client.reasoning_effort == "low" and getattr(client, "paid_profile", False) is True,
               "Historical QA requires the exact paid M3 model, 4096 output tokens, and low reasoning")
        _check(all(type(value) in (int, float) and math.isfinite(value) for value in (client.prompt_price, client.output_price))
               and client.prompt_price == .15 and client.output_price == .60,
               "Historical QA requires the frozen prompt/completion price ceilings")

    @staticmethod
    def _runtime(body):
        return profiles.runtime_contract_from_request(body, task_scope=profiles.HISTORICAL_QA_SCOPE,
            input_framing_id=profiles.HISTORICAL_QA_FRAMING)

    def _validate_profile(self):
        try:
            _text(self.profile.scope, 16_000, nonempty=True)
            expected = self._runtime(profiles.build_historical_qa_request("", "", ""))
            profiles.require_matching_contract(self.profile.runtime_contract, expected)
            _check(self.profile.blocks == _STEP, "Historical QA requires the exact frozen six-block step")
        except profiles.JudgeProfileError:
            raise JudgeError("Historical QA profile runtime or framing mismatch") from None

    def _result(self, result):
        encoded = _json(result)
        _check(len(encoded.encode("utf-8")) <= self.max_output_bytes + 64_000, "Historical QA result exceeds output byte bound")
        _check(not self.client.key or self.client.key not in encoded, "Credential material detected in historical QA result")
        return result

    @classmethod
    def from_profile(cls, client, path, *, artifact_root, expected_sha256=profiles.PROFILE_SHA256,
                     max_input_bytes=MAX_INPUT_BYTES, max_output_bytes=MAX_OUTPUT_BYTES):
        pass                                                                   
        cls._validate_client(client)
        runtime = cls._runtime(profiles.build_historical_qa_request("", "", ""))
        try:
            profile = profiles.load_historical_qa_profile(path, artifact_root=artifact_root, runtime_contract=runtime,
                task_scope=profiles.HISTORICAL_QA_SCOPE, expected_sha256=expected_sha256)
        except profiles.JudgeProfileError:
            raise JudgeError("Historical QA owned profile could not be loaded") from None
        return cls(client, profile, max_input_bytes=max_input_bytes, max_output_bytes=max_output_bytes)

    def judge(self, *, question: str, full_context: str, candidate_response: str, observation_id: str) -> dict:
        try:
            return self._judge(question, full_context, candidate_response, observation_id)
        except (PaidHTTPError, RateLimitError, IncompleteResponseError) as error:
            raise error from None
        except JudgeError as error:
            raise JudgeError(str(error)) from None
        except Exception:
            raise JudgeError("Historical QA judging failed; no automatic retry") from None

    def _judge(self, question, full_context, candidate_response, observation_id):
        self._validate_client(self.client)
        self._validate_profile()
        observation_id = _text(observation_id, 2048, nonempty=True)
        question = _text(question, 32_000, nonempty=True)
        full_context = _text(full_context, self.max_input_bytes)
        candidate_response = _text(candidate_response, 128_000)
        _check(not self.client.key or all(self.client.key not in value for value in (observation_id, question, full_context, candidate_response)),
               "Credential material is forbidden in historical QA input")
        body = profiles.build_historical_qa_request(question, full_context, candidate_response)
        encoded = profiles.encode_request(body)
        _check(len(encoded) <= self.max_input_bytes, "Historical QA request exceeds input byte bound")
        runtime = self._runtime(body)
        profiles.require_matching_contract(runtime, self.profile.runtime_contract)
        text_hash = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        base = {"schema_version": 1, "judge_version": "historical-qa-source-faithfulness-v1", "query_id": observation_id,
            "observation_id": observation_id, "model": _MODEL, "reasoning_effort": "low", "task_scope": profiles.HISTORICAL_QA_SCOPE,
            "input_framing_id": profiles.HISTORICAL_QA_FRAMING, "confidence_kind": "uncalibrated_model_self_report",
            "calibration_status": "explicit_historical_QA_research_profile", "profile_sha256": self.profile.profile_sha256,
            "question_sha256": text_hash(question), "context_sha256": text_hash(full_context), "candidate_response_sha256": text_hash(candidate_response),
            "input_sha256": _hash({"question": question, "full_context": full_context, "candidate_response": candidate_response}),
            "prompt_sha256": _hash({"model": body["model"], "messages": body["messages"]}), "body_sha256": hashlib.sha256(encoded).hexdigest(),
            "model_sha256": _hash(_MODEL), "request_runtime": runtime, "request_runtime_sha256": _hash(runtime),
            "rubric_utf8_sha256": runtime["rubric_utf8_sha256"], "judgment_schema_sha256": runtime["judgment_schema_sha256"],
            "evidence_count": int(bool(full_context.strip())), "missing_citation_count": 0, "judged_at": datetime.now(timezone.utc).isoformat()}
        if not full_context.strip() or not candidate_response.strip():
            reason = "No supplied context" if not full_context.strip() else "No factual candidate answer text supplied"
            return self._result({**base, "groundedness_probability": None, "abstained": True, "explanation": reason, "unsupported_claims": [],
                "provider": None, "request_id": None, "request_sent": False, "tokens": 0, "usage": {}, "cost_usd": 0., "reserved_usd": 0., "latency_ms": 0., "catalog": None,
                "profile_outputs": {"raw_groundedness_probability": None, "calibrated_groundedness_probability": None, "calibration_applied": False,
                    "abstained": True, "unavailable_reason": reason, "profile_sha256": self.profile.profile_sha256,
                    "task_scope": profiles.HISTORICAL_QA_SCOPE, "scope": self.profile.scope,
                    "limitations": "Local unavailable input; no actual provider or calibrated judgment is inferred. Historical QA research only."}})
        try:
            preflight = self.client.preflight()
            _check(isinstance(preflight, dict) and preflight.get("id") == _MODEL, "Historical QA preflight model mismatch")
            supported = preflight.get("supported_parameters")
            _check(isinstance(supported, list) and all(isinstance(value, str) for value in supported)
                   and ("response_format" in supported or "structured_outputs" in supported)
                   and ("reasoning" in supported or "reasoning_effort" in supported), "Historical QA requires advertised JSON and reasoning support")
            output_format = self.client.response_format(preflight, schema=None, name="groundedness_judgment")
            reasoning = self.client.reasoning_options(preflight)
            _check(output_format == {"type": "json_object"} and reasoning == {"reasoning": {"effort": "low"}}, "Historical QA preflight format/reasoning mismatch")
        except (PaidHTTPError, RateLimitError, IncompleteResponseError) as error:
            raise error from None
        except Exception:
            raise JudgeError("Historical QA preflight failed; verify exact model and advertised JSON/reasoning support") from None
        self._validate_client(self.client)
        result = self._sender._send(body, base, preflight)
        try:
            outputs = profiles.apply_historical_qa_profile(self.profile, result["groundedness_probability"], abstained=result["abstained"],
                actual_provider=result["provider"], runtime_contract=runtime, task_scope=profiles.HISTORICAL_QA_SCOPE)
        except profiles.JudgeProfileError:
            raise JudgeError("Historical QA actual provider/runtime/score does not permit profile application") from None
        return self._result({**result, "profile_outputs": outputs})
