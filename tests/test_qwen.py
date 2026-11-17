                                                                            

from contextlib import contextmanager
import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from magra_rag.qwen import QWEN_EMBEDDING_MODEL, QWEN_RERANKER_MODEL, QwenEncoder, QwenReranker, qwen_model_identity
from magra_rag.resources import BudgetExceeded


REVISION = "a" * 40


class Vector(list):
    def __truediv__(self, scalar):
        return Vector(value / scalar for value in self)


class Matrix(list):
    def __init__(self, rows, width=1024):
        super().__init__(rows)
        self.shape = (len(self), width)
        self.dtype = "<f4"


def fake_numpy():
    def asarray(values, *, dtype):
        if dtype != "<f4":
            raise AssertionError("Expected portable float32 vectors")
        return Vector(values)

    return SimpleNamespace(asarray=asarray, stack=lambda values: Matrix(values),
                           empty=lambda shape, dtype: Matrix([], shape[1]),
                           linalg=SimpleNamespace(norm=lambda values: math.sqrt(sum(value * value for value in values))),
                           isfinite=lambda values: SimpleNamespace(all=lambda: all(math.isfinite(value) for value in values)))


class Tensor:
    def __init__(self, data):
        self.data = data
        shape = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            if not current:
                break
            current = current[0]
        self.shape = tuple(shape)

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, key):
        value = self.data
        for index in key if isinstance(key, tuple) else (key,):
            value = value[index]
        return Tensor(value) if isinstance(value, list) else value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return Vector(self.data)


class Logits:
    def __init__(self, differences, malformed=False):
        self.differences = differences
        self.shape = (len(differences), 2 if malformed else 1, 151669)

    def __getitem__(self, key):
        row, _, token = key
        return self.differences[row] + 1.0 if token == 9693 else 1.0


class Torch:
    float32 = "float32"

    def __init__(self):
        self.set_num_threads = Mock()
        self.active = 0
        self.entries = 0

    @contextmanager
    def inference_mode(self):
        self.active += 1
        self.entries += 1
        try:
            yield
        finally:
            self.active -= 1


class Tokenizer:
    eos_token_id = 151645                                                      

    def __init__(self):
        self.calls = []
        self.encode_calls = []
        self.pad_calls = []
        self.right_padding = False
        self.wrong_eos = False
        self.empty_mask = False
        self.overflow = False
        self.drop_suffix = False
        self.wrong_yes = False

    def convert_tokens_to_ids(self, token):
        return (0 if self.wrong_yes else 9693) if token == "yes" else 2152

    def encode(self, text, *, add_special_tokens):
        self.encode_calls.append((text, add_special_tokens))
        return [201, 202] if text.endswith("</think>\n\n") else [101, 102, 103]

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        maximum = kwargs["max_length"]
        if kwargs.get("return_tensors") == "pt":
            rows = []
            for text in texts:
                count = min(maximum, max(2, len(text.split())))
                if self.overflow:
                    count = maximum + 1
                rows.append([17] * (count - 1) + [151645 if self.wrong_eos else 151643])
            return self._pad(rows)
        count = maximum + 1 if self.overflow else min(20, maximum)
        return {"input_ids": [[11 if "better" in text else 10]
                              * (min(3, count) if "shortbody" in text else count) for text in texts]}

    def _pad(self, rows):
        width = max(map(len, rows))
        padded, masks = [], []
        for row in rows:
            extra = width - len(row)
            padded.append(row + [0] * extra if self.right_padding else [0] * extra + row)
            masks.append([1] * len(row) + [0] * extra if self.right_padding else [0] * extra + [1] * len(row))
        if self.empty_mask:
            masks[0] = [0] * width
        return {"input_ids": Tensor(padded), "attention_mask": Tensor(masks)}

    def pad(self, encoded, **kwargs):
        self.pad_calls.append((copy.deepcopy(encoded), kwargs))
        rows = encoded["input_ids"]
        if self.drop_suffix:
            rows = [row[:-1] for row in rows]
        return self._pad(rows)


class Model:
    def __init__(self, torch, *, reranker=False):
        self.torch = torch
        self.reranker = reranker
        self.calls = []
        self.to = Mock(return_value=self)
        self.eval = Mock(return_value=self)
        self.generate = Mock(side_effect=AssertionError("Generation is forbidden"))
        self.config = SimpleNamespace(hidden_size=1024, eos_token_id=151643)
        self.width = 1024
        self.nonfinite = False
        self.zero = False
        self.malformed = False
        self.kv_cache = False

    def __call__(self, **kwargs):
        if self.torch.active < 1:
            raise AssertionError("Model forward requires inference_mode")
        self.calls.append(kwargs)
        rows = kwargs["input_ids"].data
        if self.reranker:
            differences = [3.0 if 11 in row else 1.0 for row in rows]
            if self.nonfinite:
                differences[0] = float("nan")
            return SimpleNamespace(logits=Logits(differences, self.malformed),
                                   past_key_values=object() if self.kv_cache else None)
        hidden = []
        for row in rows:
            units = []
            for token in row:
                head = [3.0, 4.0] if token == 151643 else [100.0, 0.0]
                if self.zero:
                    head = [0.0, 0.0]
                if self.nonfinite:
                    head[0] = float("nan")
                units.append(head + [0.0] * (self.width - 2))
            hidden.append(units)
        if self.malformed:
            hidden = hidden[:-1]
        return SimpleNamespace(last_hidden_state=Tensor(hidden))


class QwenTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        budget = SimpleNamespace(root=self.root, check=Mock(return_value={}),
                                 contains=lambda path: Path(path).resolve())
        self.store = SimpleNamespace(budget=budget)
        self.torch, self.tokenizer = Torch(), Tokenizer()
        self.embedding = Model(self.torch)
        self.reranking = Model(self.torch, reranker=True)
        self.tokenizer_factory = SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer))
        self.embedding_factory = SimpleNamespace(from_pretrained=Mock(return_value=self.embedding))
        self.reranking_factory = SimpleNamespace(from_pretrained=Mock(return_value=self.reranking))
        modules = {"numpy": fake_numpy(), "torch": self.torch,
                   "transformers": SimpleNamespace(AutoTokenizer=self.tokenizer_factory,
                                                   AutoModel=self.embedding_factory,
                                                   AutoModelForCausalLM=self.reranking_factory)}
        for patcher in (patch.dict(sys.modules, modules), patch("magra_rag.qwen.os.cpu_count", return_value=16)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def encoder(self, **kwargs):
        return QwenEncoder(self.store, QWEN_EMBEDDING_MODEL, REVISION, **kwargs)

    def reranker(self, **kwargs):
        return QwenReranker(self.store, QWEN_RERANKER_MODEL, REVISION, **kwargs)

    def test_identity_is_lightweight_and_tracks_revision_role_and_truncation(self):
        with patch.dict(sys.modules, {"torch": None, "numpy": None, "transformers": None}):
            identity = qwen_model_identity(QWEN_EMBEDDING_MODEL, REVISION)
        self.assertEqual(self.encoder().model_id, identity)
        self.assertEqual(self.encoder(batch_size=8).model_id, identity)
        self.assertNotEqual(identity, qwen_model_identity(QWEN_EMBEDDING_MODEL, "b" * 40))
        self.assertNotEqual(identity, qwen_model_identity(QWEN_EMBEDDING_MODEL, REVISION, max_length=256))
        self.assertNotEqual(identity, qwen_model_identity(QWEN_RERANKER_MODEL, REVISION))

    def test_identity_rejects_aliases_unpinned_revisions_and_bad_bounds(self):
        for model, revision, length in [("alias", REVISION, 512), (QWEN_EMBEDDING_MODEL, "main", 512),
                                        (QWEN_EMBEDDING_MODEL, None, 512), (QWEN_EMBEDDING_MODEL, REVISION, 513),
                                        (QWEN_EMBEDDING_MODEL, REVISION, True)]:
            with self.subTest(model=model, revision=revision, length=length), self.assertRaises(ValueError):
                qwen_model_identity(model, revision, max_length=length)

    def test_constructors_load_only_local_pinned_cpu_weights_and_check_budget(self):
        encoder, reranker = self.encoder(), self.reranker()
        for wrapper, factory in ((encoder, self.embedding_factory), (reranker, self.reranking_factory)):
            options = factory.from_pretrained.call_args.kwargs
            self.assertEqual(options["revision"], REVISION)
            self.assertEqual(options["cache_dir"], str(self.root / "models"))
            self.assertTrue(options["local_files_only"])
            self.assertFalse(options["trust_remote_code"])
            self.assertEqual(options["dtype"], self.torch.float32)
            wrapper.model.to.assert_called_once_with("cpu")
            wrapper.model.eval.assert_called_once_with()
            self.assertEqual(wrapper.cpu_threads, 4)
        self.assertEqual(self.store.budget.check.call_count, 4)
        self.assertEqual(self.torch.set_num_threads.call_count, 2)
        self.assertEqual(self.tokenizer_factory.from_pretrained.call_args.kwargs["padding_side"], "left")

    def test_budget_and_argument_failures_precede_weight_loading(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": 9}, {"batch_size": True}, {"max_length": 0}, {"max_length": 513}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.encoder(**kwargs)
        with self.assertRaises(ValueError):
            QwenEncoder(self.store, QWEN_RERANKER_MODEL, REVISION)
        self.store.budget.check.side_effect = BudgetExceeded("fixture cap")
        with self.assertRaises(BudgetExceeded):
            self.encoder()
        self.embedding_factory.from_pretrained.assert_not_called()

    def test_document_and_query_formats_differ_only_by_explicit_query_instruction(self):
        encoder = self.encoder()
        encoder.encode(["plain document"], show_progress_bar=False)
        encoder.encode(["capital?"], is_query=True)
        self.assertEqual(self.tokenizer.calls[0][0], ["plain document"])
        self.assertEqual(self.tokenizer.calls[1][0], ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:capital?"])
        for _, options in self.tokenizer.calls:
            self.assertEqual(options["max_length"], 512)
            self.assertTrue(options["truncation"])
        self.assertTrue(all(call["use_cache"] is False for call in self.embedding.calls))

    def test_pooling_honors_attention_mask_and_model_eos_not_chat_eos(self):
        self.tokenizer.right_padding = True
        result = self.encoder(batch_size=2).encode(["short", "a substantially longer document"])
        self.assertEqual(result.shape, (2, 1024))
        self.assertEqual(result.dtype, "<f4")
        for vector in result:
            self.assertAlmostEqual(vector[0], .6)
            self.assertAlmostEqual(vector[1], .8)
            self.assertAlmostEqual(sum(value * value for value in vector), 1.0)
        self.assertNotEqual(self.tokenizer.eos_token_id, 151643)

    def test_batching_override_empty_input_and_normalization_contract(self):
        encoder = self.encoder(batch_size=2)
        result = encoder.encode(["a", "b", "c"], batch_size=1)
        self.assertEqual(result.shape, (3, 1024))
        self.assertEqual(len(self.embedding.calls), 3)
        self.assertEqual(encoder.encode([]).shape, (0, 1024))
        for kwargs in ({"normalize_embeddings": False}, {"batch_size": 9}, {"is_query": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                encoder.encode(["document"], **kwargs)

    def test_encoder_rejects_bad_numerics_dimensions_and_tokenizer_outputs(self):
        encoder = self.encoder()
        for attribute, value in (("nonfinite", True), ("zero", True), ("width", 1023), ("malformed", True)):
            original = getattr(self.embedding, attribute)
            setattr(self.embedding, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                encoder.encode(["document"])
            setattr(self.embedding, attribute, original)
        for attribute in ("wrong_eos", "empty_mask", "overflow"):
            setattr(self.tokenizer, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                encoder.encode(["document"])
            setattr(self.tokenizer, attribute, False)

    def test_encoder_rejects_incompatible_checkpoint_dimensions_or_model_eos(self):
        self.embedding.config.hidden_size = 1023
        with self.assertRaisesRegex(ValueError, "dimensions or model EOS"):
            self.encoder()
        self.embedding.config.hidden_size = 1024
        self.embedding.config.eos_token_id = 151645
        with self.assertRaisesRegex(ValueError, "dimensions or model EOS"):
            self.encoder()

    def test_encoder_validates_text_collection_and_call_size(self):
        encoder = self.encoder()
        for texts in ("characters are not a batch", None, [42], ("x" for _ in range(1001))):
            with self.subTest(texts=type(texts)), self.assertRaises(ValueError):
                encoder.encode(texts)

    def test_call_budgets_are_checked_before_and_after_inference(self):
        encoder = self.encoder()
        before = self.store.budget.check.call_count
        encoder.encode(["document"])
        self.assertEqual(self.store.budget.check.call_count, before + 2)
        before = len(self.embedding.calls)
        self.store.budget.check.side_effect = BudgetExceeded("fixture cap")
        with self.assertRaises(BudgetExceeded):
            encoder.encode(["document"])
        self.assertEqual(len(self.embedding.calls), before)

    def test_call_payload_byte_bound_fails_before_forward(self):
        encoder = self.encoder()
        with patch("magra_rag.qwen._MAX_INPUT_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "64 MB"):
                encoder.encode(["ninebytes"])
        self.assertEqual(self.embedding.calls, [])

    def test_reranker_uses_forward_yes_no_difference_and_preserves_candidate_data(self):
        reranker = self.reranker(batch_size=2)
        candidates = [{"id": "z", "title": "Weak", "text": "ordinary"},
                      {"id": "b", "title": "Best", "text": "better"},
                      {"id": "a", "title": "Tie", "text": "ordinary"}]
        original = copy.deepcopy(candidates)
        ranked = reranker.rerank("question", candidates)
        self.assertEqual([row["id"] for row in ranked], ["b", "a", "z"])
        self.assertEqual([row["rerank_score"] for row in ranked], [3.0, 1.0, 1.0])
        self.assertEqual(candidates, original)
        self.assertEqual(len(self.reranking.calls), 2)
        for call in self.reranking.calls:
            self.assertFalse(call["use_cache"])
            self.assertEqual(call["logits_to_keep"], 1)
        self.reranking.generate.assert_not_called()

    def test_reranker_body_truncation_retains_official_fixed_prefix_and_suffix(self):
        reranker = self.reranker(max_length=12)
        reranker.rerank("question", [{"id": "a", "title": "Title", "text": "long text " * 100}])
        bodies, options = self.tokenizer.calls[-1]
        self.assertIn("<Query>: question\n<Document>: Title\n", bodies[0])
        self.assertEqual(options["max_length"], 7)
        self.assertEqual(options["truncation"], "longest_first")
        padded, _ = self.tokenizer.pad_calls[-1]
        self.assertEqual(padded["input_ids"][0][:3], [101, 102, 103])
        self.assertEqual(padded["input_ids"][0][-2:], [201, 202])
        self.assertEqual(len(padded["input_ids"][0]), 12)
        self.assertTrue(self.tokenizer.encode_calls[0][0].startswith("<|im_start|>system\nJudge whether"))
        self.assertTrue(self.tokenizer.encode_calls[1][0].endswith("<think>\n\n</think>\n\n"))
        self.assertTrue(all(special is False for _, special in self.tokenizer.encode_calls))

    def test_reranker_rejects_unusable_prefix_budget_or_wrong_yes_no_before_weights(self):
        with self.assertRaisesRegex(ValueError, "prefix/suffix"):
            self.reranker(max_length=5)
        self.tokenizer.wrong_yes = True
        with self.assertRaisesRegex(ValueError, "yes/no"):
            self.reranker()
        self.reranking_factory.from_pretrained.assert_not_called()

    def test_reranker_rejects_right_padding_before_final_position_forward(self):
        reranker = self.reranker(batch_size=2)
        self.tokenizer.right_padding = True
        with self.assertRaisesRegex(ValueError, "left padding"):
            reranker.rerank("question", [{"id": "a", "title": "A", "text": "shortbody"},
                                         {"id": "b", "title": "B", "text": "longer body"}])
        self.assertEqual(self.reranking.calls, [])

    def test_reranker_rejects_nonfinite_or_malformed_logits_cache_and_suffix_loss(self):
        reranker = self.reranker()
        candidate = [{"id": "a", "title": "Title", "text": "text"}]
        for attribute in ("nonfinite", "malformed", "kv_cache"):
            setattr(self.reranking, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                reranker.rerank("question", candidate)
            setattr(self.reranking, attribute, False)
        for attribute in ("overflow", "drop_suffix"):
            setattr(self.tokenizer, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                reranker.rerank("question", candidate)
            setattr(self.tokenizer, attribute, False)

    def test_empty_rerank_does_not_forward_and_bad_inputs_fail_clearly(self):
        reranker = self.reranker()
        self.assertEqual(reranker.rerank("question", []), [])
        self.reranking.generate.assert_not_called()
        self.assertEqual(self.reranking.calls, [])
        for question, candidates in (("", []), ("q", None), ("q", [{"id": "a", "text": "no title"}])):
            with self.subTest(question=question, candidates=candidates), self.assertRaises(ValueError):
                reranker.rerank(question, candidates)

    def test_missing_optional_dependencies_fail_without_model_or_strategy_fallback(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "optional dense dependencies"):
                self.encoder()
        self.embedding_factory.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
