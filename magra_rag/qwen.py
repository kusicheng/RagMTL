                                                                          

                                                                              
                                                                            
                                                                             
   

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re


QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_QUERY_PREFIX = f"Instruct: {_INSTRUCTION}\nQuery:"
_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_EMBEDDING_EOS = 151643                                                                    
_YES_TOKEN, _NO_TOKEN = 9693, 2152
_DIMENSIONS = 1024
_MAX_INPUTS = 1000
_MAX_INPUT_BYTES = 64_000_000


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _protocol(model: str, revision: str, max_length: int) -> dict:
    if not isinstance(model, str) or model not in {QWEN_EMBEDDING_MODEL, QWEN_RERANKER_MODEL}:
        raise ValueError("Unsupported Qwen role model; select the exact Qwen3 0.6B embedding or reranker ID")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Qwen requires a pinned lowercase 40-hex revision")
    _integer(max_length, "max_length", 1, 512)
    result = {"format": "qwen-native-v1", "dtype": "float32", "max_length": max_length,
              "padding_side": "left", "instruction": _INSTRUCTION}
    if model == QWEN_EMBEDDING_MODEL:
        result.update(role="embedding", query_prefix=_QUERY_PREFIX, document_prefix="",
                      pooling="last_nonpadding_model_eos", eos_token_id=_EMBEDDING_EOS,
                      dimensions=_DIMENSIONS, normalization="l2", output_dtype="little-endian-float32")
    else:
        result.update(role="reranker", body="<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
                      document="title + newline + text", prefix=_RERANK_PREFIX, suffix=_RERANK_SUFFIX,
                      yes_token_id=_YES_TOKEN, no_token_id=_NO_TOKEN, score="yes_minus_no_logit",
                      use_cache=False, logits_to_keep=1)
    return result


def qwen_model_identity(model: str, revision: str, *, max_length: int = 512) -> str:
    pass                                                                          

                                                                                
                                                                               
       
    protocol = _protocol(model, revision, max_length)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{model}@{revision}#qwen-native-v1:{fingerprint[:16]}"


def qwen_reranker_execution(model: str, revision: str, *, device="cpu", max_length=512,
                           batch_size=1) -> dict:
    pass                                                                            
    if model != QWEN_RERANKER_MODEL:
        raise ValueError("Execution policy requires the native Qwen reranker")
    protocol = _protocol(model, revision, max_length)
    if device == "cpu":
        _integer(batch_size, "batch_size", 1, 8)
        return protocol
    if device != "cuda:0" or max_length != 512 or type(batch_size) is not int or batch_size != 1:
        raise ValueError("Optional CUDA reranking requires explicit cuda:0, native512 and batch_size=1")
    from .cuda_resources import CUDA_POLICY
    return {**protocol, "format": "qwen-native-cuda-v1", "dtype": "float16", "device_policy": device,
            "input_policy": "native512", "batch_size": 1, "attention": "sdpa", "eval": True,
            "torch_cpu_threads": 1, "resource_policy": dict(CUDA_POLICY)}


def qwen_reranker_identity(model: str, revision: str, *, device="cpu", max_length=512,
                          batch_size=1) -> str:
    policy = qwen_reranker_execution(model, revision, device=device,
                                    max_length=max_length, batch_size=batch_size)
    if device == "cpu":
        return qwen_model_identity(model, revision, max_length=max_length)
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"{model}@{revision}#qwen-native-cuda-v1:{fingerprint[:16]}"


def _texts(values, *, max_items=_MAX_INPUTS) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("texts must be a collection of strings")
    try:
        rows = list(itertools.islice(iter(values), max_items + 1))
    except TypeError as error:
        raise ValueError("texts must be a collection of strings") from error
    if len(rows) > max_items:
        raise ValueError(f"At most {max_items} texts or candidates are accepted per call")
    if any(not isinstance(text, str) for text in rows):
        raise ValueError("Every input text must be a string")
    if sum(len(text.encode("utf-8")) for text in rows) > _MAX_INPUT_BYTES:
        raise ValueError("Input text exceeds the 64 MB call bound")
    return rows


class _QwenCPU:
    def _prepare(self, store, model, revision, max_length, batch_size, expected):
        if model != expected:
            raise ValueError("Qwen adapter model does not match its embedding/reranker role")
        self.protocol = _protocol(model, revision, max_length)
        self.model_id = qwen_model_identity(model, revision, max_length=max_length)
        self.format_fingerprint = self.model_id.rsplit(":", 1)[1]
        self.revision, self.max_length = revision, max_length
        self.batch_size = _integer(batch_size, "batch_size", 1, 8)
        self.store = store
        self.cache = store.budget.contains(Path(store.budget.root) / "models")
        store.budget.check()
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install optional dense dependencies with native Qwen3 Transformers support") from error
        self.torch = torch
        self.cpu_threads = min(4, os.cpu_count() or 1)
        torch.set_num_threads(self.cpu_threads)
        self.load_options = {"revision": revision, "cache_dir": str(self.cache),
                             "local_files_only": True, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", **self.load_options)

    def _load_model(self, model, factory):
        self.model = factory.from_pretrained(model, dtype=self.torch.float32,
                                             low_cpu_mem_usage=True, attn_implementation="sdpa",
                                             **self.load_options).to("cpu").eval()

    def _validate_tokens(self, tokens, batch_size):
        ids, mask = tokens["input_ids"], tokens["attention_mask"]
        if (len(ids.shape) != 2 or tuple(mask.shape) != tuple(ids.shape)
                or ids.shape[0] != batch_size or not 1 <= ids.shape[1] <= self.max_length):
            raise ValueError("Tokenizer returned a malformed or out-of-bound token batch")
        rows, masks = ids.tolist(), mask.tolist()
        if any(not any(row) or any(value not in (0, 1) for value in row) for row in masks):
            raise ValueError("Tokenizer returned an empty or invalid attention mask")
        return rows, masks


class QwenEncoder(_QwenCPU):
    def __init__(self, store, model, revision, max_length=512, batch_size=1):
        self._prepare(store, model, revision, max_length, batch_size, QWEN_EMBEDDING_MODEL)
        try:
            import numpy as np
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("Install optional dense dependencies with native Qwen3 Transformers support") from error
        self.np = np
        self._load_model(model, AutoModel)
        if (getattr(self.model.config, "hidden_size", None) != _DIMENSIONS
                or getattr(self.model.config, "eos_token_id", None) != _EMBEDDING_EOS):
            raise ValueError("Qwen embedding checkpoint has incompatible dimensions or model EOS")
        store.budget.check()

    def encode(self, texts, normalize_embeddings=True, batch_size=None,
               show_progress_bar=False, *, is_query=False):
        pass                                                                            
        if normalize_embeddings is not True:
            raise ValueError("Qwen vector protocol requires normalize_embeddings=True")
        if not isinstance(is_query, bool):
            raise ValueError("is_query must be boolean")
        size = self.batch_size if batch_size is None else _integer(batch_size, "batch_size", 1, 8)
        rows = _texts(texts)
        self.store.budget.check()
        vectors = []
        for start in range(0, len(rows), size):
            batch = rows[start:start + size]
            if is_query:
                batch = [_QUERY_PREFIX + text for text in batch]
            tokens = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
            ids, masks = self._validate_tokens(tokens, len(batch))
            with self.torch.inference_mode():
                hidden = self.model(**tokens, use_cache=False).last_hidden_state
                if tuple(hidden.shape) != (len(batch), tokens["input_ids"].shape[1], _DIMENSIONS):
                    raise ValueError("Qwen encoder returned malformed hidden states")
                for index, mask in enumerate(masks):
                    last = max(position for position, active in enumerate(mask) if active)
                    if ids[index][last] != _EMBEDDING_EOS:
                        raise ValueError("Qwen tokenizer must append the publisher model EOS automatically")
                    vector = self.np.asarray(hidden[index, last].detach().cpu().numpy(), dtype="<f4")
                    norm = float(self.np.linalg.norm(vector))
                    if not self.np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
                        raise ValueError("Qwen encoder returned a nonfinite or zero vector")
                    vector = self.np.asarray(vector / norm, dtype="<f4")
                    if (not self.np.isfinite(vector).all()
                            or not math.isclose(float(self.np.linalg.norm(vector)), 1.0, rel_tol=1e-5, abs_tol=1e-6)):
                        raise ValueError("Qwen normalization did not return a finite unit vector")
                    vectors.append(vector)
        result = self.np.stack(vectors) if vectors else self.np.empty((0, _DIMENSIONS), dtype="<f4")
        self.store.budget.check()
        return result


class QwenReranker(_QwenCPU):
    def __init__(self, store, model, revision, max_length=512, batch_size=1, *, device="cpu"):
        qwen_reranker_execution(model, revision, device=device,
                               max_length=max_length, batch_size=batch_size)
        self._cuda_previous_threads = None
        if device == "cuda:0":
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("Install optional dense dependencies with native Qwen3 Transformers support") from error
            self._cuda_previous_threads = torch.get_num_threads()
            self.torch = torch
        try:
            self._initialize(store, model, revision, max_length, batch_size, device=device)
        except BaseException:
            self._restore_cuda_threads()
            raise

    def _restore_cuda_threads(self):
        if self._cuda_previous_threads is not None:
            self.torch.set_num_threads(self._cuda_previous_threads)
            self._cuda_previous_threads = None

    def _initialize(self, store, model, revision, max_length, batch_size, *, device):
        execution = qwen_reranker_execution(model, revision, device=device,
                                           max_length=max_length, batch_size=batch_size)
        identity = qwen_reranker_identity(model, revision, device=device,
                                         max_length=max_length, batch_size=batch_size)
        self.device, self.cuda_guard, self._closed = device, None, False
        self._prepare(store, model, revision, max_length, batch_size, QWEN_RERANKER_MODEL)
        self.protocol, self.model_id = execution, identity
        self.format_fingerprint = identity.rsplit(":", 1)[1]
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("Install optional dense dependencies with native Qwen3 Transformers support") from error
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")
        if (self.yes_token_id, self.no_token_id) != (_YES_TOKEN, _NO_TOKEN):
            raise ValueError("Qwen tokenizer has incompatible yes/no token IDs")
        self.prefix_tokens = self.tokenizer.encode(_RERANK_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(_RERANK_SUFFIX, add_special_tokens=False)
        self.body_length = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self.body_length < 1:
            raise ValueError("max_length cannot fit the fixed Qwen reranker prefix/suffix and body")
        if device == "cpu":
            self._load_model(model, AutoModelForCausalLM)
        else:
            if len(self.prefix_tokens) != 39 or len(self.suffix_tokens) != 9:
                raise ValueError("CUDA native512 requires the exact 39+9-token framing")
            from .cuda_resources import CudaResourceGuard
            self.cpu_threads = 1
            self.torch.set_num_threads(1)
            self.cuda_guard = CudaResourceGuard(self.torch, device)
            try:
                with self.cuda_guard.monitoring():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model, dtype=self.torch.float16, low_cpu_mem_usage=True,
                        attn_implementation="sdpa", **self.load_options).to(device).eval()
                    self.torch.cuda.synchronize(device)
                store.budget.check()
            except BaseException:
                self.close()
                raise
            return
        store.budget.check()

    def resource_observations(self):
        return None if self.cuda_guard is None else self.cuda_guard.observations()

    def close(self):
        pass                                                                   
        def release_model():
            self.model = None
            self._closed = True
            if self.cuda_guard is not None:
                gc.collect()
        if self.cuda_guard is None:
            release_model()
            self._restore_cuda_threads()
        else:
                                                                            
                                                                            
            try:
                self.cuda_guard.close(release_model=release_model)
            finally:
                if self._closed:
                    self._restore_cuda_threads()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Qwen reranker is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def rerank(self, question, candidates):
        if self._closed:
            raise RuntimeError("Qwen reranker is closed")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Reranking requires a nonempty question")
        if isinstance(candidates, (str, bytes, Mapping)):
            raise ValueError("candidates must be a collection of document mappings")
        try:
            documents = list(itertools.islice(iter(candidates), _MAX_INPUTS + 1))
        except TypeError as error:
            raise ValueError("candidates must be a collection of document mappings") from error
        if len(documents) > _MAX_INPUTS:
            raise ValueError(f"At most {_MAX_INPUTS} texts or candidates are accepted per call")
        for document in documents:
            if (not isinstance(document, Mapping) or any(not isinstance(document.get(field), str)
                                                       for field in ("id", "title", "text"))
                    or not document["id"]):
                raise ValueError("Reranking candidates need string id, title and text")
        _texts([question] + [document["title"] + "\n" + document["text"] for document in documents],
               max_items=_MAX_INPUTS + 1)
        self.store.budget.check()
        if self.cuda_guard is not None:
            self.cuda_guard.check()
            if self.torch.get_num_threads() != 1:
                raise RuntimeError("CUDA reranker requires one Torch CPU worker; close before changing it")
        ranked = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            bodies = [f"<Instruct>: {_INSTRUCTION}\n<Query>: {question}\n<Document>: "
                      + document["title"] + "\n" + document["text"] for document in batch]
            encoded = self.tokenizer(bodies, padding=False, truncation="longest_first",
                                     return_attention_mask=False, max_length=self.body_length)
            if len(encoded["input_ids"]) != len(batch):
                raise ValueError("Qwen tokenizer returned the wrong number of reranker bodies")
            if any(len(row) > self.body_length for row in encoded["input_ids"]):
                raise ValueError("Qwen tokenizer exceeded the reranker body token bound")
            encoded["input_ids"] = [self.prefix_tokens + row + self.suffix_tokens for row in encoded["input_ids"]]
            tokens = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            ids, masks = self._validate_tokens(tokens, len(batch))
            for row, mask in zip(ids, masks):
                if mask[-1] != 1 or mask != sorted(mask):
                    raise ValueError("Qwen final-position reranker logits require left padding")
                visible = [token for token, active in zip(row, mask) if active]
                if (visible[:len(self.prefix_tokens)] != self.prefix_tokens
                        or visible[-len(self.suffix_tokens):] != self.suffix_tokens):
                    raise ValueError("Qwen tokenizer did not preserve the fixed reranker prefix/suffix")
            boundary = self.cuda_guard.monitoring() if self.cuda_guard is not None else nullcontext()
            with boundary, self.torch.inference_mode():
                if self.cuda_guard is not None:
                    if self.torch.get_num_threads() != 1:
                        raise RuntimeError("CUDA reranker process thread policy changed before forward")
                    tokens = {name: value.to(self.device) for name, value in tokens.items()}
                output = self.model(**tokens, use_cache=False, logits_to_keep=1)
                logits = output.logits
                if (len(logits.shape) != 3 or logits.shape[0] != len(batch) or logits.shape[1] != 1
                        or logits.shape[2] <= max(self.yes_token_id, self.no_token_id)
                        or getattr(output, "past_key_values", None) is not None):
                    raise ValueError("Qwen reranker returned malformed final logits or a KV cache")
                for index, document in enumerate(batch):
                    yes = float(logits[index, -1, self.yes_token_id])
                    no = float(logits[index, -1, self.no_token_id])
                    score = yes - no
                    if not all(math.isfinite(value) for value in (yes, no, score)):
                        raise ValueError("Qwen reranker returned a nonfinite score")
                    ranked.append(dict(document, rerank_score=score))
                if self.cuda_guard is not None:
                    self.torch.cuda.synchronize(self.device)
                    if self.torch.get_num_threads() != 1:
                        raise RuntimeError("CUDA reranker process thread policy changed during forward")
        self.store.budget.check()
        return sorted(ranked, key=lambda document: (-document["rerank_score"], document["id"]))
