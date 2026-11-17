                                                                                   
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from magra_benchmark import judge
from magra_benchmark import judge_profiles as p


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
BLOCKS=[{"upper_score":u,"probability":v,"components":n,"supported":y} for u,v,n,y in
        ((.3,.125,8,1),(.6,3/13,13,3),(.62,.5,2,1),(.82,11/14,14,11),(.93,18/19,19,18),(.95,1.,5,5))]
THRESHOLDS={"raw":.8,"calibrated":11/14,"raw_fixed_control":.5}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.body=p.build_historical_qa_request('Q "quoted"?','Full context\n雪 \\ "text"','Candidate\nanswer')
        self.runtime=p.runtime_contract_from_request(self.body,task_scope=p.HISTORICAL_QA_SCOPE,input_framing_id=p.HISTORICAL_QA_FRAMING)
        legacy={key:value for key,value in self.runtime.items() if key not in {"contract_version","task_scope","input_framing_id"}}
        legacy["request_builder"]=dict(p.REFERENCE_BUILDER)
        self.assertEqual(sha(canonical(legacy)),p.REFERENCE_RUNTIME_SHA256)
        calibrator={"fit_role":"TRAIN","threshold_selected":False,"blocks":BLOCKS}
        calibrator["artifact_hash"]=sha(canonical(calibrator));cal_raw=canonical(calibrator)
        thresholds={"selection_role":"DEV","TRAIN_calibrator_refit":False,"TEST_settings_frozen":True,
            "TEST_settings":{"thresholds":THRESHOLDS,"TRAIN_calibrator_sha256":sha(cal_raw)}}
        thresholds["artifact_hash"]=sha(canonical(thresholds))
        required=("train_fit_run_v1/calibrator.json","dev_threshold_run_v2/thresholds.json","late_test_assessment_run_v1/assessment.json",
            "late_test_assessment_run_v1/complete.json","independent_test_arithmetic_audit_v1.json","root_test64_raw_envelope_audit_gate_v1.json",
            "heldout_diagnostics_v1.png","train_decode_inputs_v2.py","train_fit_v1.py")
        refs={}
        for name in required:
            raw=cal_raw if name.endswith("calibrator.json") else canonical(thresholds) if name.endswith("thresholds.json") else b"invented aggregate reference"
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            refs[name]={"bytes":len(raw),"sha256":sha(raw)}
        self.manifest={"format":"magra.experimental.historical-QA-source-faithfulness-profile.v1","status":"FROZEN_RESEARCH_PROFILE_POINT_GATES_MET",
            "automatic_generic_judge_loading_enabled":False,"original_human33_AUDIT_labels_used":False,"TEST_retuning_performed":False,
            "runtime_contract":legacy,"runtime_contract_sha256":sha(canonical(legacy)),"transform":{"blocks":BLOCKS},
            "thresholds":THRESHOLDS,"classifier":"supported iff probability>=threshold","provenance":refs,"scope":"Invented historical QA research fixture"}
        self.path=self.root/"profile.json";self.freeze()

    def freeze(self):
        self.path.write_bytes(canonical(self.manifest));self.pin=sha(self.path.read_bytes())

    def load(self,**changes):
        args={"artifact_root":self.root,"runtime_contract":self.runtime,"task_scope":p.HISTORICAL_QA_SCOPE,"expected_sha256":self.pin};args.update(changes)
        return p.load_historical_qa_profile(self.path,**args)

    def apply(self,profile,score,**changes):
        args={"abstained":False,"actual_provider":"Fireworks","runtime_contract":self.runtime,"task_scope":p.HISTORICAL_QA_SCOPE};args.update(changes)
        return p.apply_historical_qa_profile(profile,score,**args)

    def test_exact_QA_body_and_double_escaping_without_input_projection(self):
        user=json.loads(self.body["messages"][1]["content"])
        self.assertEqual(user,{"query":{"question":'Q "quoted"?'},"prediction":{"answer":"Candidate\nanswer","abstained":False,"citations":[]},
            "evidence":[{"doc_id":"supplied_context","sentence_id":0,"title":"","text":'Full context\n雪 \\ "text"'}],"missing_citations":[]})
        wire=p.encode_request(self.body);self.assertEqual(json.loads(wire),self.body)
        self.assertEqual(sha(judge.SYSTEM.encode()),self.runtime["rubric_utf8_sha256"])
        self.assertEqual(sha(canonical(judge.JUDGMENT_SCHEMA)),self.runtime["judgment_schema_sha256"])
        with self.assertRaises(p.JudgeProfileError):p.build_historical_qa_request("question",{"labels":[]},"candidate")

    def test_provider_format_model_controls_schema_and_framing_are_identity(self):
        changes=({"model":"different/model"},{"provider":{**self.body["provider"],"only":["different"]}},
            {"response_format":{"type":"json_schema","json_schema":{"strict":True}}},{"max_tokens":4097},
            {"temperature":.1},{"reasoning":{"effort":"high"}},{"provider":{**self.body["provider"],"allow_fallbacks":True}})
        for change in changes:
            with self.subTest(change=change):
                runtime=p.runtime_contract_from_request({**self.body,**change},task_scope=p.HISTORICAL_QA_SCOPE,input_framing_id=p.HISTORICAL_QA_FRAMING)
                with self.assertRaises(p.JudgeProfileError):self.load(runtime_contract=runtime)
        runtime=p.runtime_contract_from_request(self.body,task_scope=p.HISTORICAL_QA_SCOPE,input_framing_id=p.HISTORICAL_QA_FRAMING,judgment_schema={"type":"object"})
        with self.assertRaises(p.JudgeProfileError):self.load(runtime_contract=runtime)
        rubric=deepcopy(self.body);rubric["messages"][0]["content"]+=" changed"
        runtime=p.runtime_contract_from_request(rubric,task_scope=p.HISTORICAL_QA_SCOPE,input_framing_id=p.HISTORICAL_QA_FRAMING)
        with self.assertRaises(p.JudgeProfileError):self.load(runtime_contract=runtime)
        with self.assertRaises(p.JudgeProfileError):self.load(runtime_contract={**self.runtime,"ignored_extra":True})
        generic=p.runtime_contract_from_request(self.body,task_scope="generic",input_framing_id="generic_query_prediction_evidence")
        with self.assertRaises(p.JudgeProfileError):self.load(runtime_contract=generic)
        absent=deepcopy(self.body);del absent["reasoning"]
        generic=p.runtime_contract_from_request(absent,task_scope="generic",input_framing_id="generic_query_prediction_evidence")
        self.assertIsNone(generic["reasoning"])
        with self.assertRaises(p.JudgeProfileError):self.load(task_scope="generic")
        altered=deepcopy(self.body);user=json.loads(altered["messages"][1]["content"]);user["query"]["id"]="generic-query"
        altered["messages"][1]["content"]=canonical(user).decode()
        with self.assertRaises(p.JudgeProfileError):p.runtime_contract_from_request(altered,task_scope=p.HISTORICAL_QA_SCOPE,input_framing_id=p.HISTORICAL_QA_FRAMING)

    def test_owned_bounded_references_fail_closed_without_private_annotations(self):
        profile=self.load();self.assertEqual(len(profile.blocks),6)
        self.assertFalse(any(self.root.rglob("*labels*")));self.assertFalse((self.root/"private_test").exists())
        with self.assertRaises(p.JudgeProfileError):self.load(expected_sha256="0"*64)
        (self.root/"train_fit_v1.py").write_bytes(b"corrupt reference")
        with self.assertRaises(p.JudgeProfileError):self.load()
        self.manifest["provenance"]["../private_test/000.labels_quality.json"]={"bytes":1,"sha256":"0"*64};self.freeze()
        with self.assertRaises(p.JudgeProfileError):self.load()

    def test_frozen_six_block_clamped_step_boundaries_preserves_raw(self):
        profile=self.load()
        for score,expected in ((0.,.125),(.3,.125),(.3000001,3/13),(.6,3/13),(.62,.5),(.6200001,11/14),(.82,11/14),(.93,18/19),(.95,1.),(1.,1.)):
            result=self.apply(profile,score)
            self.assertEqual(result["raw_groundedness_probability"],score);self.assertEqual(result["calibrated_groundedness_probability"],expected)
        result=self.apply(profile,.7)
        self.assertFalse(result["raw_supported_at_profile_threshold"]);self.assertTrue(result["calibrated_supported_at_profile_threshold"])
        self.assertEqual(result["thresholds"],THRESHOLDS);self.assertIn("Brier and AUROC worsened",result["limitations"])
        leaked=profile.runtime_contract;leaked["model"]="different"
        with self.assertRaises(p.JudgeProfileError):self.apply(profile,.7,runtime_contract=leaked)
        self.assertEqual(profile.runtime_contract,self.runtime)

    def test_abstained_unavailable_and_invalid_scores_never_become_zero(self):
        profile=self.load();result=self.apply(profile,None,abstained=True)
        self.assertIsNone(result["raw_groundedness_probability"]);self.assertIsNone(result["calibrated_groundedness_probability"])
        self.assertFalse(result["calibration_applied"]);self.assertIsNone(result["calibrated_supported_at_profile_threshold"])
        for value in (None,True,".8",-.01,1.01,float("nan"),float("inf")):
            with self.subTest(score=value):
                with self.assertRaises(p.JudgeProfileError):self.apply(profile,value)
        with self.assertRaises(p.JudgeProfileError):self.apply(profile,.8,abstained=True)
        with self.assertRaises(p.JudgeProfileError):self.apply(profile,.8,actual_provider="GMICloud")
        with self.assertRaises(p.JudgeProfileError):self.apply(profile,.8,task_scope="current_MAGRA_answers")

    def test_frozen_public_63_request_bodies_have_exact_wire_parity(self):
                                                                           
                                                                               
        root=Path(__file__).resolve().parents[1]/"data/magra-research-v2/public_calibration_v1"
        paths=[root/"paid_dev_run_v1"/("request_%03d.body.bin"%i) for i in range(35)]
        paths += [root/"paid_dev_continuation_run_v1"/("request_%03d.body.bin"%i) for i in range(36,64)]
        if not all(path.is_file() for path in paths):self.skipTest("Public research request bodies are not installed")
        for path in paths:
            raw=path.read_bytes();body=json.loads(raw);user=json.loads(body["messages"][1]["content"])
            rebuilt=p.build_historical_qa_request(user["query"]["question"],user["evidence"][0]["text"],user["prediction"]["answer"])
            self.assertTrue(p.encode_request(rebuilt)==raw,path.name+": exact wire-byte mismatch")


if __name__=="__main__":unittest.main()
