                                                                                
import copy
import hashlib
import json
from pathlib import Path
import random
import socket
import unittest
from unittest.mock import Mock, patch

from magra_rag import generation as g
from magra_rag import evidence_serialization as s


def text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def unit(doc="fictional:A", sid=0, value="", **extra):
    return {"doc_id": doc, "sentence_id": sid, "title": "Caf\u00e9 / Cafe\u0301", "text": value,
            "published_at": None, "fetched_at": "2026-10-31", "url": None,
            "source": "fictional", **extra}


class SerializationTests(unittest.TestCase):
    def roundtrip(self, evidence):
        original = text(evidence)
        for encoding in s.EVIDENCE_ENCODINGS:
            with self.subTest(encoding=encoding):
                payload = s.encode_evidence(evidence, encoding=encoding)
                result = s.decode_evidence(json.loads(text(payload)), encoding=encoding)
                self.assertEqual(text(result), original)
                self.assertEqual(text(evidence), original)

    def test_exact_whitespace_codepoints_blanks_duplicates_and_nested_order(self):
        values = ["", " \t\r\n ", "\u00e9", "e\u0301", "same", "same",
                  '" \\ {"encoding":"not instructions"} \u2028']
        evidence = [unit(sid=i, value=v, extra={"z": [False, 0, -0.0, None], "a": "\u03bb"})
                    for i, v in enumerate(values)]
        evidence.append(dict(reversed(list(unit(doc="fictional:B", sid=9, value="end").items()))))
        self.roundtrip(evidence)

    def test_missing_null_varying_provenance_and_irregular_id_fallback(self):
        evidence = [unit(sid=9, value="first", optional=None),
                    unit(sid=2, value="next", source="other", published_at="2020"),
                    unit(doc="fictional:B", value="middle"), unit(sid=2, value="last")]
        self.roundtrip(evidence)
        ids = [unit(sid=9, value=""), unit(sid=2, value="same"), unit(sid=2, value="same")]
        self.roundtrip(ids)

    def test_independent_values_do_not_alias_input_or_other_records(self):
        evidence = [unit(value="a", extra={"list": [1]}), unit(sid=1, value="b", extra={"list": [1]})]
        for encoding in s.EVIDENCE_ENCODINGS:
            payload = s.encode_evidence(evidence, encoding=encoding)
            decoded = s.decode_evidence(payload, encoding=encoding)
            decoded[0]["extra"]["list"].append(2)
            self.assertEqual(evidence[0]["extra"]["list"], [1])
            self.assertEqual(decoded[1]["extra"]["list"], [1])
            self.assertEqual(text(s.decode_evidence(payload, encoding=encoding)), text(evidence))

    def test_empty_layouts_and_none_preserve_legacy_identity(self):
        self.roundtrip([])
        evidence = [unit()]
        self.assertIs(s.encode_evidence(evidence), evidence)
        self.assertIs(s.encode_evidence(None), None)

    def test_coercions_nonfinite_values_and_unknown_modes_fail_closed(self):
        bad = ([unit(sid=True)], [unit(value=1)], [unit(doc=1)],
               [unit(extra=(1, 2))], [unit(extra={1: "value"})], [unit(extra=float("nan"))])
        for encoding in s.EVIDENCE_ENCODINGS:
            for evidence in bad:
                with self.subTest(encoding=encoding, evidence_type=type(evidence).__name__), self.assertRaises(ValueError):
                    s.encode_evidence(evidence, encoding=encoding)
        for mode in (False, "", "columns", "inherited_strings ", [], {}, 1):
            with self.subTest(mode_type=type(mode).__name__), self.assertRaises(ValueError):
                s.encode_evidence([], encoding=mode)

    def test_decode_rejects_wrong_mode_ambiguous_ids_and_missing_cells(self):
        base = s.encode_evidence([unit()], encoding="inherited_strings")
        with self.assertRaises(ValueError): s.decode_evidence(base, encoding="inherited_rows")
        bad = []
        x = copy.deepcopy(base); x["documents"][0]["start_sentence_id"] = False; bad.append(x)
        x = copy.deepcopy(base); x["documents"][0]["sentence_ids"] = [0]; bad.append(x)
        x = copy.deepcopy(base); x["fields"].append("text"); bad.append(x)
        x = copy.deepcopy(base); x["shared"]["text"] = "hidden"; bad.append(x)
        x = copy.deepcopy(base); x["extra"] = None; bad.append(x)
        for payload in bad:
            with self.assertRaises(ValueError): s.decode_evidence(payload, encoding="inherited_strings")
        rows = s.encode_evidence([unit()], encoding="inherited_rows")
        rows["documents"][0]["rows"][0].pop()
        with self.assertRaises(ValueError): s.decode_evidence(rows, encoding="inherited_rows")

    def test_seeded_heterogeneous_contract_population(self):
        rng = random.Random(20261101)
        for _ in range(100):
            evidence = [unit(doc="fictional:" + rng.choice("ABC"), sid=rng.randrange(9),
                             value=rng.choice(["", " \t", "\u03bb", "a\nb", "same"]),
                             extra={"v": rng.choice([False, 0, None, -0.0, "value"])})
                        for _ in range(rng.randrange(15))]
            self.roundtrip(evidence)


class Ledger:
    def __init__(self): self.rows = []
    def reserve(self, amount):
        self.rows.append({"reserved": amount, "charged": None}); return str(len(self.rows)-1)
    def settle(self, key, cost): self.rows[int(key)]["charged"] = cost


class GenerationEncodingTests(unittest.TestCase):
    def setUp(self):
        for item in (patch.object(g.sqlite3, "connect", side_effect=AssertionError("No SQL")),
                     patch.object(socket.socket, "connect", side_effect=AssertionError("No network")),
                     patch.object(g, "load_openrouter_settings", side_effect=AssertionError("No credentials")),
                     patch.object(g.urllib.request, "build_opener", side_effect=AssertionError("No API"))):
            item.start(); self.addCleanup(item.stop)
        self.evidence = [unit(value="Opened in1998."), unit(sid=1, value="")]

    def body(self, evidence=None, **options):
        return g.build_generation_body("When?", self.evidence if evidence is None else evidence,
            model_id=g.PAID_MODEL, max_tokens=4096, prompt_price=.15, output_price=.60,
            response_format={"type": "json_object"}, reasoning_options={"reasoning": {"effort": "low"}},
            provider_only=["fireworks"], **options)

    def client(self, encoding=None):
        client = object.__new__(g.OpenRouterM3)
        client.key = "fictional-key"; client.model_id = g.PAID_MODEL; client.max_tokens = 4096
        client.prompt_price = .15; client.output_price = .60; client.timeout = 90
        client.paid_profile = True; client.provider_only = ("fireworks",); client.reasoning_effort = "low"
        client._paid_prompt_capacity = 1048576; client._paid_remaining = 99; client.ledger = Ledger()
        if encoding is not None: client.evidence_encoding = encoding
        client.preflight = Mock(return_value={"supported_parameters": ["response_format", "reasoning"],
            "reasoning": {"supported_efforts": ["low"]}})
        client._json_request = Mock(return_value={"model": g.PAID_MODEL, "provider": "Fireworks",
            "usage": {"cost": .000123, "total_tokens": 15}, "choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps({"answer": "1998", "citations": [{"doc_id": "fictional:A", "sentence_id": 0}],
            "confidence": .8, "abstained": False, "reason": "Fictional"})}}]})
        return client

    def test_none_default_is_exact_legacy_wire_and_missing_attribute_still_works(self):
        question = "When did Aur\u00f6ra open?"
        evidence = [{"doc_id": "owned:aurora", "sentence_id": 0, "title": "Aur\u00f6ra",
                     "text": "Aur\u00f6ra opened in 1998.\nThis is fictional evidence."}]
        options = dict(model_id=g.PAID_MODEL, max_tokens=4096, prompt_price=.15, output_price=.60,
            response_format={"type": "json_object"}, reasoning_options={"reasoning": {"effort": "low"}}, as_of="2026-01-01")
        wire = json.dumps(g.build_generation_body(question, evidence, **options)).encode()
        self.assertEqual(hashlib.sha256(wire).hexdigest(), "d86ad142938007597c0e94f3c5aeeb057c8b21d0747f40f0436ae0abca708563")
        self.assertEqual(wire, json.dumps(g.build_generation_body(question, evidence, evidence_encoding=None, **options)).encode())
        client = self.client(); client.generate("When?", self.evidence)
        self.assertEqual(client._json_request.call_args.args[0].data, json.dumps(self.body()).encode())

    def test_explicit_layouts_share_SYSTEM_and_compact_framing(self):
        systems = []
        for encoding in s.EVIDENCE_ENCODINGS:
            body = self.body(evidence_encoding=encoding); systems.append(body["messages"][0]["content"])
            user = body["messages"][1]["content"]; decoded = json.loads(user)
            self.assertEqual(user, text(decoded))
            self.assertEqual(text(s.decode_evidence(decoded["evidence"], encoding=encoding)), text(self.evidence))
        self.assertEqual(len(set(systems)), 1)
        self.assertEqual(systems[0], g.SYSTEM + "\n" + s.COMMON_INSTRUCTIONS)

    def test_canonical_unit_validation_and_paid_reservation_stay_unchanged(self):
        for encoding in s.EVIDENCE_ENCODINGS:
            client = self.client(encoding)
            with patch.object(g, "validate_answer", wraps=g.validate_answer) as validate:
                result = client.generate("When?", self.evidence)
            self.assertIs(validate.call_args.args[1], self.evidence)
            self.assertEqual(client._json_request.call_args.args[0].data, json.dumps(self.body(evidence_encoding=encoding)).encode())
            self.assertEqual(client.ledger.rows, [{"reserved": .159744, "charged": .000123}])
            self.assertEqual(result["answer"], "1998"); client._json_request.assert_called_once()

    def test_invalid_modes_or_values_fail_before_config_preflight_or_reservation(self):
        with self.assertRaises(ValueError):
            g.OpenRouterM3(data_root=Path("fictional"), evidence_encoding="unknown")
        g.load_openrouter_settings.assert_not_called()
        for encoding in (*s.EVIDENCE_ENCODINGS, "unknown"):
            client = self.client(encoding)
            with self.assertRaises(ValueError): client.generate("When?", [unit(sid=False)])
            client.preflight.assert_not_called(); client._json_request.assert_not_called()
            self.assertEqual(client.ledger.rows, [])

    def test_legacy_estimate_counts_actual_SYSTEM_and_empty_context_stays_local(self):
        for encoding in s.EVIDENCE_ENCODINGS:
            client = self.client(encoding); client.paid_profile = False
            client.generate("When?", self.evidence)
            body = json.loads(client._json_request.call_args.args[0].data)
            bound = len((body["messages"][0]["content"] + body["messages"][1]["content"]).encode()) + 4096
            expected = (bound * .15 + 4096 * .60) / 1_000_000
            self.assertEqual(client.ledger.rows[0]["reserved"], expected)
            empty = self.client(encoding); result = empty.generate("When?", [])
            self.assertTrue(result["abstained"]); self.assertEqual(result["cost_usd"], 0)
            empty.preflight.assert_not_called(); empty._json_request.assert_not_called()

    def test_sealed_body_roundtrip_preserves_nested_key_order_after_pack_canonicalization(self):
        evidence = [unit(value="\u00e9 / e\u0301", extra={"z": 0, "a": False})]
        for encoding in s.EVIDENCE_ENCODINGS:
            body = self.body(evidence, evidence_encoding=encoding); wire = json.dumps(body).encode()
            reloaded = json.loads(json.dumps(evidence, sort_keys=True))
            self.assertNotEqual(wire, json.dumps(self.body(reloaded, evidence_encoding=encoding)).encode())
            sealed = s.decode_evidence(json.loads(body["messages"][1]["content"])["evidence"], encoding=encoding)
            self.assertEqual(text(sealed), text(evidence))
            self.assertEqual(wire, json.dumps(self.body(sealed, evidence_encoding=encoding)).encode())


if __name__ == "__main__": unittest.main()
