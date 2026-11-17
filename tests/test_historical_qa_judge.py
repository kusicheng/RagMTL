                                                                                  
from copy import deepcopy
import json
import traceback
import unittest
from unittest.mock import patch

from magra_benchmark.historical_qa_judge import HistoricalQASourceFaithfulnessJudge
from magra_benchmark import judge, judge_profiles as profiles
from magra_rag.generation import OpenRouterM3, PaidHTTPError, RateLimitError, IncompleteResponseError

KEY="invented-test-credential-never-real"
MODEL="minimax/minimax-m3.1-flash"


class Ledger:
    def __init__(self):self.reserved=[];self.settled=[]
    def reserve(self,value):self.reserved.append(value);return "invented-reservation"
    def settle(self,identity,cost):self.settled.append((identity,cost))


class Client(OpenRouterM3):
    def __init__(self):
                                                                                     
        self.key=KEY;self.model_id=MODEL;self.max_tokens=4096;self.reasoning_effort="low"
        self.prompt_price=.15;self.output_price=.60;self.paid_profile=True;self.timeout=5;self.ledger=Ledger()
        self.preflights=0;self.requests=[];self.failure=None;self.transport_failure=None
        self.catalog={"id":MODEL,"catalog_sha256":"a"*64,"checked_at":"invented-time","pricing":{"prompt":.00000015,"completion":.0000006},
            "context_length":1048576,"supported_parameters":["response_format","structured_outputs","reasoning"],"reasoning":{"supported_efforts":["low","medium","high"]}}
        self.content=json.dumps({"groundedness_probability":.7,"abstained":False,"explanation":"Invented judgment","unsupported_claims":[]})
        self.envelope={"model":MODEL,"provider":"Fireworks","id":"invented-generation","usage":{"cost":.0001,"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},
            "choices":[{"finish_reason":"stop","message":{"content":self.content}}]}
    def preflight(self):
        self.preflights+=1
        if self.failure is not None:raise self.failure
        self._paid_prompt_capacity=1048576
        return self.catalog
    def paid_reservation(self,encoded):
        assert self._paid_prompt_capacity==1048576
        return (self._paid_prompt_capacity*self.prompt_price+self.max_tokens*self.output_price)/1_000_000
    def _json_request(self,request,timeout):
        self.requests.append(request)
        if self.transport_failure is not None:raise self.transport_failure
        return self.envelope


def profile():
    body=profiles.build_historical_qa_request("", "", "")
    runtime=profiles.runtime_contract_from_request(body,task_scope=profiles.HISTORICAL_QA_SCOPE,input_framing_id=profiles.HISTORICAL_QA_FRAMING)
    blocks=((.3,.125),(.6,3/13),(.62,.5),(.82,11/14),(.93,18/19),(.95,1.))
    return profiles.HistoricalQAProfile(profiles.PROFILE_SHA256,json.dumps(runtime,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode(),blocks,"Invented historical research scope")


class HistoricalJudgeTests(unittest.TestCase):
    def setUp(self):
        self.block=patch("urllib.request.urlopen",side_effect=AssertionError("Real network is forbidden in offline tests"));self.block.start();self.addCleanup(self.block.stop)
        self.client=Client();self.judge=HistoricalQASourceFaithfulnessJudge(self.client,profile())
    def invoke(self,**changes):
        values={"question":"Invented question?","full_context":"Full invented context\n雪 \"quoted\"","candidate_response":"Invented answer","observation_id":"invented-case"};values.update(changes)
        return self.judge.judge(**values)

    def test_exact_request_shared_send_raw_and_separate_step_identity(self):
        with patch.object(profiles,"load_historical_qa_profile",return_value=profile()) as loader:
            adapter=HistoricalQASourceFaithfulnessJudge.from_profile(self.client,"invented-profile.json",artifact_root="invented-owned-root")
        self.assertEqual(loader.call_args.kwargs["task_scope"],profiles.HISTORICAL_QA_SCOPE)
        self.assertEqual(adapter.profile.runtime_contract,self.judge.profile.runtime_contract)
        with patch.object(judge.M3GroundednessJudge,"_send",wraps=self.judge._sender._send) as shared:
            result=self.invoke()
        self.assertEqual(shared.call_count,1);self.assertEqual(self.client.preflights,1);self.assertEqual(len(self.client.requests),1)
        expected=profiles.build_historical_qa_request("Invented question?","Full invented context\n雪 \"quoted\"","Invented answer")
        self.assertEqual(self.client.requests[0].data,profiles.encode_request(expected))
        self.assertEqual(result["groundedness_probability"],.7);self.assertEqual(result["profile_outputs"]["calibrated_groundedness_probability"],11/14)
        self.assertEqual(result["request_runtime"],self.judge.profile.runtime_contract)
        self.assertEqual(result["task_scope"],profiles.HISTORICAL_QA_SCOPE);self.assertEqual(result["confidence_kind"],"uncalibrated_model_self_report")
        self.assertEqual(self.client.ledger.settled,[("invented-reservation",.0001)])
        self.assertNotIn(KEY,json.dumps(result));self.assertNotIn("Full invented context",json.dumps(result))

    def test_config_and_profile_mismatch_rejected_before_preflight(self):
        for field,value in (("model_id","other/model"),("max_tokens",4095),("reasoning_effort","high"),("prompt_price",.16),("output_price",.61),("paid_profile",False)):
            with self.subTest(field=field):
                client=Client();setattr(client,field,value)
                with self.assertRaises(judge.JudgeError):HistoricalQASourceFaithfulnessJudge(client,profile())
                self.assertEqual(client.preflights,0);self.assertEqual(client.requests,[])
        wrong=profile();runtime=wrong.runtime_contract;runtime["model"]="other/model"
        wrong=profiles.HistoricalQAProfile(wrong.profile_sha256,json.dumps(runtime).encode(),wrong.blocks,wrong.scope)
        with self.assertRaises(judge.JudgeError):HistoricalQASourceFaithfulnessJudge(Client(),wrong)
        self.client.max_tokens=1000
        with self.assertRaises(judge.JudgeError):self.invoke()
        self.assertEqual(self.client.preflights,0)

    def test_empty_inputs_are_local_unavailable_without_provider_inference(self):
        for changes in ({"full_context":"  "},{"candidate_response":""}):
            result=self.invoke(**changes)
            self.assertFalse(result["request_sent"]);self.assertIsNone(result["provider"]);self.assertIsNone(result["groundedness_probability"])
            self.assertFalse(result["profile_outputs"]["calibration_applied"]);self.assertIsNone(result["profile_outputs"]["calibrated_groundedness_probability"])
        self.assertEqual(self.client.preflights,0);self.assertEqual(self.client.ledger.reserved,[])

    def test_preflight_capabilities_and_format_drift_stop_before_reserve(self):
        for supported in (["response_format"],["reasoning"],[]):
            self.client.catalog["supported_parameters"]=supported
            with self.assertRaises(judge.JudgeError):self.invoke()
        self.client.catalog["supported_parameters"]=["response_format","reasoning"]
        with patch.object(self.client,"response_format",return_value={"type":"json_schema"}):
            with self.assertRaises(judge.JudgeError):self.invoke()
        self.assertEqual(self.client.requests,[]);self.assertEqual(self.client.ledger.reserved,[])

    def test_actual_provider_rejected_after_known_charge_and_strict_parsing(self):
        self.client.envelope["provider"]="OtherProvider"
        with self.assertRaises(judge.JudgeError):self.invoke()
        self.assertEqual(len(self.client.requests),1);self.assertEqual(self.client.ledger.settled,[("invented-reservation",.0001)])
        self.client.envelope["provider"]="Fireworks";self.client.envelope["choices"][0]["message"]["content"]='{"groundedness_probability":0.8}'
        with self.assertRaises(judge.JudgeError):self.invoke()
        self.assertEqual(len(self.client.requests),2)

    def test_unknown_cost_abstention_and_incomplete_remain_distinct(self):
        self.client.envelope["usage"]["cost"]=None
        with self.assertRaises(judge.JudgeError):self.invoke()
        self.assertEqual(self.client.ledger.settled[-1],("invented-reservation",None))
        self.client.envelope["usage"]["cost"]=.0001
        self.client.envelope["choices"][0]["message"]["content"]=json.dumps({"groundedness_probability":None,"abstained":True,"explanation":"Invented refusal","unsupported_claims":[]})
        result=self.invoke();self.assertIsNone(result["groundedness_probability"]);self.assertFalse(result["profile_outputs"]["calibration_applied"])
        self.client.envelope["choices"][0]["finish_reason"]="length"
        with self.assertRaises(IncompleteResponseError):self.invoke()

    def test_typed_exhaustion_and_rate_limit_propagate_no_retry_and_key_redacts(self):
        for error in (PaidHTTPError(402),RateLimitError(retry_after_seconds=40),IncompleteResponseError("length")):
            self.client.failure=error
            with self.assertRaises(type(error)) as caught:self.invoke()
            self.assertIs(caught.exception,error)
        self.assertEqual(self.client.requests,[])
        self.client.failure=RuntimeError(KEY+" raw provider details")
        try:self.invoke()
        except judge.JudgeError as error:self.assertNotIn(KEY,"".join(traceback.format_exception(error)))
        self.client.failure=None
        with self.assertRaises(judge.JudgeError):self.invoke(full_context="contains "+KEY)
        self.assertEqual(self.client.requests,[])
        for error in (PaidHTTPError(402),RateLimitError(retry_after_seconds=40)):
            self.client.transport_failure=error;before=len(self.client.requests)
            with self.assertRaises(type(error)) as caught:self.invoke()
            self.assertIs(caught.exception,error);self.assertEqual(len(self.client.requests),before+1)


if __name__=="__main__":unittest.main()
