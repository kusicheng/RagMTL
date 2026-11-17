                                                                              
from contextlib import contextmanager
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import threading
import weakref
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REVISION = "a" * 40
GPU_UUID = "GPU-12345678-1234-5678-1234-567812345678"


def original_doubles():
    names = {"Vector", "Matrix", "fake_numpy", "Tensor", "Logits", "Torch", "Tokenizer", "Model"}
    tree = ast.parse((ROOT/"tests/test_qwen.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = {"Mock": Mock, "SimpleNamespace": SimpleNamespace, "math": math,
                 "contextmanager": contextmanager, "copy": copy}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "original_pure_test_doubles", "exec"), namespace)
    return namespace


class CudaRerankerTests(unittest.TestCase):
    def setUp(self):
                                                                              
                                                                                 
        package = ModuleType("_magra_cuda_adapter_tests")
        package.__path__ = [str(ROOT/"magra_rag")]
        scope = patch.dict(sys.modules, {package.__name__: package})
        scope.start()
        self.addCleanup(scope.stop)
        def load(name, path=None):
            full = package.__name__ + "." + name
            spec = importlib.util.spec_from_file_location(full, path or ROOT/"magra_rag"/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
            return module
        self.q, self.r = load("qwen"), load("cuda_resources")
        load("resources", ROOT/"magra_rag/resources.py")
        doubles = original_doubles()
        Tensor = doubles["Tensor"]
        def to(tensor, device):
            tensor.last_device = device
            return tensor
        Tensor.to = to
        self.torch = doubles["Torch"]()
        self.torch.float16 = "float16"
        self.thread_count = 6
        self.torch.get_num_threads = lambda: self.thread_count
        self.torch.set_num_threads.side_effect = lambda count: setattr(self, "thread_count", count)
        self.fraction = 1.
        self.reserved = self.allocated = 0
        self.total, self.free, self.used, self.host = 6_000_000_000, 5_000_000_000, 800_000_000, 20_000_000_000
        self.bad_sample_seen = threading.Event()
        def physical(_):
            if self.used > 4_000_000_000:
                self.bad_sample_seen.set()
            return SimpleNamespace(total=self.total, free=self.free, used=self.used)
        def set_fraction(value, device):
            self.assertEqual(device, "cuda:0")
            self.fraction = value
        self.cuda = SimpleNamespace(is_available=Mock(return_value=True), init=Mock(),
            get_device_properties=Mock(return_value=SimpleNamespace(uuid=GPU_UUID, total_memory=self.total)),
            memory_allocated=lambda _: self.allocated, memory_reserved=lambda _: self.reserved,
            get_per_process_memory_fraction=lambda _: self.fraction,
            set_per_process_memory_fraction=Mock(side_effect=set_fraction), synchronize=Mock(),
            empty_cache=Mock(side_effect=lambda: setattr(self, "reserved", self.allocated)))
        self.torch.cuda = self.cuda
        self.nvml = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value="invented-handle"),
            nvmlDeviceGetUUID=Mock(return_value=GPU_UUID), nvmlDeviceGetMemoryInfo=Mock(side_effect=physical))
        self.psutil = SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100_000_000)),
            virtual_memory=lambda: SimpleNamespace(used=self.host))
        self.tokenizer = doubles["Tokenizer"]()
        self.tokenizer.encode = Mock(side_effect=lambda text, **_: list(range(100,139)) if text == self.q._RERANK_PREFIX else list(range(200,209)))
        self.model = doubles["Model"](self.torch, reranker=True)
        self.factory = SimpleNamespace(from_pretrained=Mock(return_value=self.model))
        modules = {"torch": self.torch, "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)),
            AutoModelForCausalLM=self.factory), "pynvml": self.nvml, "psutil": self.psutil}
        optional = patch.dict(sys.modules, modules)
        optional.start()
        self.addCleanup(optional.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        budget = SimpleNamespace(root=Path(temp.name), contains=lambda p: Path(p), check=Mock(return_value={}))
        self.store = SimpleNamespace(budget=budget)

    def reranker(self, **kwargs):
        value = self.q.QwenReranker(self.store, self.q.QWEN_RERANKER_MODEL, REVISION, **kwargs)
        self.addCleanup(value.close)
        return value

    def guard(self):
        value = self.r.CudaResourceGuard(self.torch)
        self.addCleanup(value.close)
        return value

    def test_identity_is_lazy_and_CPU_compatible(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None, "pynvml": None, "psutil": None}):
            cpu = self.q.qwen_reranker_identity(self.q.QWEN_RERANKER_MODEL, REVISION)
            gpu = self.q.qwen_reranker_identity(self.q.QWEN_RERANKER_MODEL, REVISION, device="cuda:0")
        self.assertEqual(cpu, self.q.qwen_model_identity(self.q.QWEN_RERANKER_MODEL, REVISION))
        self.assertNotEqual(cpu, gpu)
        self.assertIn("qwen-native-cuda-v1", gpu)
        self.factory.from_pretrained.assert_not_called()

    def test_cpu_default_preserves_loading_and_does_not_open_driver(self):
        reranker = self.reranker()
        self.assertEqual(reranker.device, "cpu")
        self.assertEqual(self.factory.from_pretrained.call_args.kwargs["dtype"], self.torch.float32)
        self.model.to.assert_called_once_with("cpu")
        self.nvml.nvmlInit.assert_not_called()
        self.assertIsNone(reranker.resource_observations())


    def test_invalid_optional_policies_fail_before_model_or_driver(self):
        for kwargs in ({"device":"cuda"}, {"device":"cuda:1"}, {"device":"cuda:0","batch_size":2},
                       {"device":"cuda:0","max_length":256}, {"device":"cuda:0","batch_size":True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.reranker(**kwargs)
        self.factory.from_pretrained.assert_not_called()
        self.nvml.nvmlInit.assert_not_called()

    def test_cuda_uses_native_framing_fp16_placement_scores_and_lifecycle(self):
        with self.reranker(device="cuda:0") as reranker:
            options = self.factory.from_pretrained.call_args.kwargs
            self.assertEqual(options["dtype"], self.torch.float16)
            self.assertTrue(options["local_files_only"])
            self.assertFalse(options["trust_remote_code"])
            self.assertEqual(options["attn_implementation"], "sdpa")
            self.model.to.assert_called_once_with("cuda:0")
            self.model.eval.assert_called_once_with()
            original = [{"id":"z","title":"ordinary","text":"text","extra":7},
                        {"id":"a","title":"better","text":"text","extra":9}]
            saved = copy.deepcopy(original)
            ranked = reranker.rerank("question", original)
            self.assertEqual(original, saved)
            self.assertEqual([row["id"] for row in ranked], ["a","z"])
            self.assertEqual([row["rerank_score"] for row in ranked], [3.,1.])
            self.assertEqual(len(self.model.calls), 2)
            for call in self.model.calls:
                self.assertEqual(call["input_ids"].last_device, "cuda:0")
                self.assertEqual(call["attention_mask"].last_device, "cuda:0")
                self.assertFalse(call["use_cache"])
                self.assertEqual(call["logits_to_keep"], 1)
            self.assertEqual(self.tokenizer.calls[0][1]["max_length"], 464)
            self.assertEqual(self.tokenizer.pad_calls[0][0]["input_ids"][0][:39], list(range(100,139)))
            self.assertEqual(self.tokenizer.pad_calls[0][0]["input_ids"][0][-9:], list(range(200,209)))
            observation = reranker.resource_observations()
            self.assertIsNone(observation["last_sample"]["own_PID_VRAM_bytes"])
            self.assertEqual(observation["sampled_peaks"]["GPU_total_unavailable_including_reserved_bytes"], 1_000_000_000)
            self.assertLessEqual(observation["allocator_bytes"], 2_000_000_000)
            self.assertFalse(observation["policy"]["subsample_peak_guarantee"])
        self.nvml.nvmlShutdown.assert_called_once_with()
        self.assertEqual(self.fraction, 1.)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reranker.rerank("q", [])

    def test_admission_accounts_reserved_physical_memory_and_host_before_weights(self):
        for free, used, host in ((3_200_000_000,800_000_000,20_000_000_000),
                                (5_000_000_000,800_000_000,32_000_000_000)):
            self.free, self.used, self.host = free, used, host
            with self.subTest(free=free, host=host), self.assertRaises(self.r.CudaResourceError):
                self.reranker(device="cuda:0")
        self.factory.from_pretrained.assert_not_called()
        self.assertEqual(self.nvml.nvmlShutdown.call_count, 2)

    def test_missing_driver_inconsistent_counters_and_unavailable_cuda_do_not_fallback(self):
        with patch.dict(sys.modules, {"pynvml":None}), self.assertRaises(ImportError):
            self.reranker(device="cuda:0")
        self.cuda.is_available.return_value = False
        with self.assertRaises(self.r.CudaResourceError):
            self.reranker(device="cuda:0")
        self.cuda.is_available.return_value = True
        self.free = self.total + 1
        with self.assertRaises(self.r.CudaResourceError):
            self.reranker(device="cuda:0")
        self.factory.from_pretrained.assert_not_called()

    def test_uuid_allocator_and_prior_fraction_guards_fail_before_weights(self):
        self.nvml.nvmlDeviceGetUUID.return_value = "GPU-other"
        with self.assertRaises(self.r.CudaResourceError):
            self.reranker(device="cuda:0")
        self.nvml.nvmlDeviceGetUUID.return_value = GPU_UUID
        self.reserved = 2_100_000_000
        with self.assertRaises(self.r.CudaResourceError):
            self.reranker(device="cuda:0")
        self.reserved, self.fraction = 0, .1
        with self.assertRaises(self.r.CudaResourceError):
            self.reranker(device="cuda:0")
        self.assertEqual(self.fraction, .1)
        self.factory.from_pretrained.assert_not_called()

    def test_background_failure_is_latched_at_boundary_without_killing_process(self):
        guard = self.guard()
        with self.assertRaises(self.r.CudaResourceError):
            with guard.monitoring():
                self.used, self.free = 4_100_000_000, 1_000_000_000
                self.assertTrue(self.bad_sample_seen.wait(1.))
        self.used, self.free = 800_000_000, 5_000_000_000
        with self.assertRaises(self.r.CudaResourceError):
            guard.check()
        self.assertTrue(guard.observations()["monitor_failed"])
        self.assertEqual(guard.observations()["sampled_peaks"]["GPU_total_unavailable_including_reserved_bytes"], 5_000_000_000)
        self.assertIsNone(guard._thread)

    def test_exclusive_guard_restores_allocator_and_releases_after_init_or_model_failure(self):
        guard = self.guard()
        with self.assertRaises(self.r.CudaResourceError):
            self.guard()
        guard.close()
        self.assertEqual(self.fraction, 1.)
        self.model.to.side_effect = RuntimeError("invented model load failure")
        with self.assertRaisesRegex(RuntimeError, "invented model load failure"):
            self.reranker(device="cuda:0")
        self.assertEqual(self.fraction, 1.)
        second = self.guard()
        second.close()

    def test_real_style_UUID_objects_bare_strings_and_NVML_bytes_match(self):
        class CUuuidLike:
            def __str__(self):
                return GPU_UUID[4:].upper()
        self.nvml.nvmlDeviceGetUUID.return_value = GPU_UUID.encode("ascii")
        for raw in (CUuuidLike(), GPU_UUID[4:], GPU_UUID):
            self.cuda.get_device_properties.return_value.uuid = raw
            with self.subTest(raw=type(raw).__name__):
                guard = self.guard()
                self.assertEqual(guard.device_uuid, GPU_UUID)
                self.nvml.nvmlDeviceGetHandleByUUID.assert_called_with(GPU_UUID)
                guard.close()
        for raw in (None, object(), "GPU-invalid", "MIG-"+GPU_UUID[4:], GPU_UUID+" "):
            self.cuda.get_device_properties.return_value.uuid = raw
            with self.subTest(raw=type(raw).__name__), self.assertRaises(self.r.CudaResourceError):
                self.guard()

    def test_missing_allocator_reader_fails_clearly_before_weight_loading(self):
        self.cuda.get_per_process_memory_fraction = None
        with self.assertRaisesRegex(self.r.CudaResourceError, "get_per_process_memory_fraction"):
            self.reranker(device="cuda:0")
        self.factory.from_pretrained.assert_not_called()
        self.assertEqual(self.thread_count, 6)

    def test_active_close_rejection_preserves_model_state_and_thread_policy(self):
        reranker = self.reranker(device="cuda:0")
        model = reranker.model
        with reranker.cuda_guard.monitoring():
            with self.assertRaisesRegex(self.r.CudaResourceError, "active"):
                reranker.close()
            self.assertIs(reranker.model, model)
            self.assertFalse(reranker._closed)
            self.assertFalse(reranker.cuda_guard.observations()["closed"])
            self.assertEqual(self.thread_count, 1)
            self.cuda.empty_cache.assert_not_called()
        self.assertEqual(len(reranker.rerank("q", [{"id":"a","title":"title","text":"text"}])), 1)
        reranker.close()
        self.assertEqual(self.thread_count, 6)

    def test_close_releases_unoccupied_cache_before_restore_and_reopen(self):
        models = weakref.WeakSet()
        external_live_bytes = 100_000_000
        external_tensor = object()
        self.allocated = self.reserved = external_live_bytes
        def physical_from_allocator():
            self.used = 1_000_000_000 + self.reserved
            self.free = self.total - self.used
        physical_from_allocator()
        def release_weights():
            self.allocated -= 1_200_000_000
        def load_weights(*_, **__):
            model = type(self.model)(self.torch, reranker=True)
            models.add(model)
            self.allocated += 1_200_000_000
            self.reserved = 2_000_000_000
            physical_from_allocator()
            weakref.finalize(model, release_weights)
            return model
        def empty_cache():
            self.assertTrue(self.r._OWNER.locked())
            self.assertFalse(models)
            self.assertEqual(self.allocated, external_live_bytes)
            self.assertAlmostEqual(self.fraction, 2_000_000_000/self.total)
            self.reserved = self.allocated
            physical_from_allocator()
        self.factory.from_pretrained.side_effect = load_weights
        self.cuda.empty_cache.side_effect = empty_cache
        first = self.reranker(device="cuda:0")
        self.assertEqual(self.reserved, 2_000_000_000)
        first.close()
        self.assertEqual(self.reserved, external_live_bytes)
        self.assertIsNotNone(external_tensor)
        self.assertEqual(self.fraction, 1.)
        second = self.reranker(device="cuda:0")
        self.assertEqual(second.cuda_guard.allocator_bytes, 2_000_000_000)
        second.close()
        self.assertEqual(self.cuda.empty_cache.call_count, 2)

    def test_Torch_threads_restore_on_failure_close_and_policy_changes_fail(self):
        reranker = self.reranker(device="cuda:0")
        self.assertEqual(self.thread_count, 1)
        self.thread_count = 4
        with self.assertRaisesRegex(RuntimeError, "one Torch CPU worker"):
            reranker.rerank("q", [{"id":"a","title":"title","text":"text"}])
        self.assertEqual(self.thread_count, 4)
        self.assertEqual(self.model.calls, [])
        self.thread_count = 1
        original_forward = type(self.model).__call__
        def changed_forward(model, **kwargs):
            output = original_forward(model, **kwargs)
            self.thread_count = 4
            return output
        with patch.object(type(self.model), "__call__", changed_forward), self.assertRaisesRegex(RuntimeError, "during forward"):
            reranker.rerank("q", [{"id":"a","title":"title","text":"text"}])
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.thread_count, 4)
        reranker.close()
        self.assertEqual(self.thread_count, 6)
        self.model.to.side_effect = RuntimeError("invented model load failure")
        with self.assertRaisesRegex(RuntimeError, "invented model load failure"):
            self.reranker(device="cuda:0")
        self.assertEqual(self.thread_count, 6)

    def test_bad_logits_preserve_no_partial_result_and_closed_allocator(self):
        reranker = self.reranker(device="cuda:0")
        self.model.nonfinite = True
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            reranker.rerank("q", [{"id":"a","title":"title","text":"text"}])
        self.assertIsNone(reranker.cuda_guard._thread)
        reranker.close()
        self.assertEqual(self.fraction, 1.)


if __name__ == "__main__":
    unittest.main()
