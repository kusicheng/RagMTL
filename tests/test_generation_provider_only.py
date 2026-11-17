                                                                               
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from magra_rag import generation as g


KEY="OFFLINE-PROVIDER-KEY"
QUESTION="When did Aur\u00f6ra open?"
EVIDENCE=[{"doc_id":"owned:aurora","sentence_id":0,"title":"Aur\u00f6ra",
           "text":"Aur\u00f6ra opened in 1998.\nThis is fictional evidence."}]
                                                                       
                                                                                
DEFAULT_WIRE_SHA="d86ad142938007597c0e94f3c5aeeb057c8b21d0747f40f0436ae0abca708563"


class Ledger:
    def __init__(self):self.rows=[];self.events=[]
    def validate_admission(self):
        self.events.append(("admission",))
        if any(row["charged"] is None for row in self.rows):
            raise g.GenerationError("Fictional uncertain paid billing prevents another request")
    def reserve(self,amount):
        self.validate_admission();self.events.append(("reserve",amount))
        self.rows.append({"reserved":amount,"charged":None,"state":"reserved"})
        return str(len(self.rows)-1)
    def settle(self,key,cost):
        self.events.append(("settle",key,cost))
        self.rows[int(key)].update(charged=cost,state="unconfirmed" if cost is None else "settled")


class ProviderOnlyTests(unittest.TestCase):
    def setUp(self):
        self.ledger=Ledger()
        self.budget=SimpleNamespace(root=Path("fictional-provider-root"))
        self.path=Mock(spec=Path);self.path.is_file.return_value=True
        self.settings=SimpleNamespace(api_key=KEY,model_id=g.PAID_MODEL,profile="paid")
        self.settings_loader=Mock(return_value=self.settings)
        def ledger_factory(*args,**kwargs):
            if kwargs.get("halt_on_unconfirmed"):self.ledger.validate_admission()
            return self.ledger
        self.ledger_factory=Mock(side_effect=ledger_factory)
        for item in (
            patch.object(g,"load_openrouter_settings",self.settings_loader),
            patch.object(g,"paid_controls",return_value={"max_total_usd":100,"shared_ledger_path":self.path,
                "prompt_price":.15,"completion_price":.60}),
            patch.object(g,"ResourceBudget",return_value=self.budget),
            patch.object(g,"SpendLedger",self.ledger_factory),
            patch.object(g.sqlite3,"connect",side_effect=AssertionError("No SQL")),
            patch.object(socket.socket,"connect",side_effect=AssertionError("No network")),
            patch.object(g.urllib.request,"urlopen",side_effect=AssertionError("No API")),
            patch.object(g.urllib.request,"build_opener",side_effect=AssertionError("No API")),
        ):
            item.start();self.addCleanup(item.stop)

    def client(self,**options):
        client=g.OpenRouterM3(data_root=self.budget.root,budget=self.budget,api_key=KEY,
                              env_file=None,profile="paid",**options)
        def preflight():
            client.ledger.validate_admission()
            client._paid_prompt_capacity=1048576;client._paid_remaining=100
            return {"id":g.PAID_MODEL,"supported_parameters":["response_format","reasoning"],
                    "reasoning":{"supported_efforts":["low"]}}
        client.preflight=Mock(side_effect=preflight)
        client._json_request=Mock(return_value=self.envelope())
        return client

    @staticmethod
    def envelope(**updates):
        value={"id":"fictional-provider-response","provider":"Fireworks","model":g.PAID_MODEL,
            "usage":{"cost":.000123,"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},
            "choices":[{"finish_reason":"stop","message":{"content":json.dumps({
                "answer":"1998","citations":[{"doc_id":"owned:aurora","sentence_id":0}],
                "confidence":.8,"abstained":False,"reason":"Fictional support."})}}]}
        value.update(updates);return value

    @staticmethod
    def body(**overrides):
        options={"model_id":g.PAID_MODEL,"max_tokens":4096,"prompt_price":.15,"output_price":.60,
                 "response_format":{"type":"json_object"},"reasoning_options":{"reasoning":{"effort":"low"}},
                 "as_of":"2026-01-01"}
        options.update(overrides)
        return g.build_generation_body(QUESTION,EVIDENCE,**options)

    def test_pure_default_body_preserves_original_wire_bytes(self):
        body=self.body();wire=json.dumps(body).encode("utf-8")
        self.assertEqual(hashlib.sha256(wire).hexdigest(),DEFAULT_WIRE_SHA)
        self.assertEqual(len(wire),1342)
        self.assertEqual(wire,json.dumps(self.body(provider_only=None)).encode("utf-8"))
        self.assertNotIn("only",body["provider"])
        self.settings_loader.assert_not_called();self.ledger_factory.assert_not_called()

    def test_generate_default_and_preparer_share_exact_old_body_and_messages(self):
        client=self.client()
        result=client.generate(QUESTION,EVIDENCE,as_of="2026-01-01")
        wire=client._json_request.call_args.args[0].data
        self.assertEqual(hashlib.sha256(wire).hexdigest(),DEFAULT_WIRE_SHA)
        self.assertEqual(wire,json.dumps(self.body()).encode("utf-8"))
        self.assertEqual(result["prompt_sha256"],hashlib.sha256(json.dumps(self.body()["messages"],sort_keys=True).encode()).hexdigest())
        self.assertNotIn("provider_only",result)
        self.assertAlmostEqual(result["reserved_usd"],.159744)
        self.assertEqual(self.ledger.rows[-1]["charged"],.000123)

    def test_explicit_provider_is_immutable_normalized_and_wire_matches_preparer(self):
        providers=["Fireworks"];client=self.client(provider_only=providers)
        providers[0]="another-provider"
        self.assertEqual(client.provider_only,("fireworks",))
        result=client.generate(QUESTION,EVIDENCE,as_of="2026-01-01")
        request=client._json_request.call_args.args[0]
        self.assertEqual(request.data,json.dumps(self.body(provider_only=["fireworks"])).encode("utf-8"))
        self.assertEqual(json.loads(request.data)["provider"],{"max_price":{"prompt":.15,"completion":.60},
            "require_parameters":True,"allow_fallbacks":False,"only":["fireworks"]})
        self.assertEqual(result["provider"],"Fireworks")
        self.assertAlmostEqual(self.ledger.rows[-1]["reserved"],.159744)
        client._json_request.assert_called_once()

    def test_invalid_provider_policies_fail_before_config_ledger_or_transport(self):
        invalid=("fireworks",[],(),{},set(["fireworks"]),[""],[None],[True],[" fireworks"],
                 ["fireworks "],["fireworks\n"],["fireworks","Fireworks"],["fireworks"]*17,
                 ["x"*129],["Fireworks AI"],["fireworks/../other"],["\u00f6ther"])
        for value in invalid:
            with self.subTest(value=type(value).__name__),self.assertRaises(ValueError):self.client(provider_only=value)
        self.settings_loader.assert_not_called();self.ledger_factory.assert_not_called()

    def test_missing_wrong_or_malformed_provider_keeps_known_charge_before_error(self):
        for provider in (None,"",123,"another-provider","Fireworks AI",KEY):
            self.ledger=Ledger();client=self.client(provider_only=("fireworks",))
            client._json_request.return_value=self.envelope(provider=provider)
            with patch.object(g,"validate_answer",side_effect=AssertionError("Provider mismatch must precede answer parsing")):
                with self.subTest(provider_type=type(provider).__name__),self.assertRaisesRegex(g.GenerationError,"provider_only") as caught:
                    client.generate(QUESTION,EVIDENCE)
            self.assertNotIn(KEY,str(caught.exception))
            self.assertEqual(self.ledger.rows[-1],{"reserved":.159744,"charged":.000123,"state":"settled"})
            client._json_request.assert_called_once()

    def test_unknown_billing_remains_primary_and_halts_another_request(self):
        client=self.client(provider_only=("fireworks",))
        client._json_request.return_value=self.envelope(provider="another-provider",usage={"total_tokens":15})
        with self.assertRaisesRegex(g.GenerationError,"billing is unknown"):client.generate(QUESTION,EVIDENCE)
        self.assertEqual(self.ledger.rows[-1],{"reserved":.159744,"charged":None,"state":"unconfirmed"})
        with self.assertRaisesRegex(g.GenerationError,"uncertain"):client.generate(QUESTION,EVIDENCE)
        client._json_request.assert_called_once()

    def test_paid402_and429_keep_types_and_pending_reservation_without_retry(self):
        for error in (g.PaidHTTPError(402),g.RateLimitError(retry_after_seconds=60)):
            self.ledger=Ledger();client=self.client(provider_only=("fireworks",));client._json_request.side_effect=error
            with self.subTest(status=error.status_code),self.assertRaises(type(error)) as caught:
                client.generate(QUESTION,EVIDENCE)
            self.assertIs(caught.exception,error)
            self.assertEqual(self.ledger.rows[-1],{"reserved":.159744,"charged":None,"state":"reserved"})
            with self.assertRaisesRegex(g.GenerationError,"uncertain"):client.generate(QUESTION,EVIDENCE)
            client._json_request.assert_called_once()

    def test_incomplete_costly_response_retains_type_and_known_charge(self):
        client=self.client(provider_only=("fireworks",));payload=self.envelope()
        payload["choices"][0]["finish_reason"]="length";client._json_request.return_value=payload
        with self.assertRaises(g.IncompleteResponseError):client.generate(QUESTION,EVIDENCE)
        self.assertEqual(self.ledger.rows[-1]["charged"],.000123)
        self.assertEqual(self.ledger.rows[-1]["state"],"settled")
        client._json_request.assert_called_once()

    def test_provider_error_cannot_mask_over_reservation_or_wrong_model(self):
        for update,message in (({"usage":{"cost":1}},"exceeded reservation"),({"model":"different-model"},"Response model")):
            self.ledger=Ledger();client=self.client(provider_only=("fireworks",))
            client._json_request.return_value=self.envelope(provider="another-provider",**update)
            with self.subTest(message=message),self.assertRaisesRegex(g.GenerationError,message):client.generate(QUESTION,EVIDENCE)
            self.assertEqual(self.ledger.rows[-1]["state"],"settled")
            self.assertEqual(self.ledger.rows[-1]["charged"],update.get("usage",{}).get("cost",.000123))

    def test_no_evidence_preserves_zero_request_abstention(self):
        client=self.client(provider_only=["fireworks"])
        result=client.generate(QUESTION,[])
        self.assertTrue(result["abstained"]);self.assertEqual(result["cost_usd"],0)
        client.preflight.assert_not_called();client._json_request.assert_not_called();self.assertEqual(self.ledger.rows,[])

    def test_multiple_explicit_slugs_are_exact_and_unrequested_provider_stays_legacy(self):
        client=self.client(provider_only=["fireworks","other-provider/zone_1"])
        client._json_request.return_value=self.envelope(provider="OTHER-PROVIDER/ZONE_1")
        client.generate(QUESTION,EVIDENCE)
        self.assertEqual(json.loads(client._json_request.call_args.args[0].data)["provider"]["only"],["fireworks","other-provider/zone_1"])
        self.ledger=Ledger();unrequested=self.client();payload=self.envelope();payload.pop("provider")
        unrequested._json_request.return_value=payload
        self.assertIsNone(unrequested.generate(QUESTION,EVIDENCE)["provider"])

    def test_billing_coverage_is_passed_unchanged_and_reasoning_cannot_override_route(self):
        coverage=object();self.client(provider_only=["fireworks"],billing_coverage=coverage)
        self.assertIs(self.ledger_factory.call_args.kwargs["billing_coverage"],coverage)
        with self.assertRaises(ValueError):self.body(reasoning_options={"provider":{"only":["another-provider"]}})
        self.assertEqual(self.body(provider_only=["fireworks"])["reasoning"],{"effort":"low"})


if __name__=="__main__":unittest.main()
