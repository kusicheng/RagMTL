                                                                         

                                                                           
                                                                         
   
from __future__ import annotations

from contextlib import contextmanager
import importlib
import math
import re
import threading
import time
from types import MappingProxyType
import uuid


CUDA_POLICY = MappingProxyType({
    "physical_GPU_total_unavailable_bytes": 4_000_000_000,
    "host_used_bytes": 32_000_000_000,
    "process_RSS_bytes": 8_000_000_000,
    "allocator_max_bytes": 2_000_000_000,
    "minimum_allocator_bytes": 1_320_000_000,
    "nonallocator_headroom_bytes": 256_000_000,
    "sample_interval_seconds": .02,
    "measurement": "max(driver_used,driver_total-driver_free), includes other processes/reserved",
    "own_PID_VRAM_measured": False,
    "subsample_peak_guarantee": False,
    "respect_existing_allocator_limit": True,
})
_OWNER = threading.Lock()


class CudaResourceError(RuntimeError):
    pass


def _need(condition, message):
    if not condition:
        raise CudaResourceError(message)


def _physical_uuid(value):
    pass                                                                 
    try:
        text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    except Exception as error:
        raise CudaResourceError("A valid physical CUDA UUID is required") from error
    if text.startswith("GPU-"):
        text = text[4:]
    _need(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", text) is not None,
          "A valid physical CUDA UUID is required")
    return "GPU-" + str(uuid.UUID(text))


class CudaResourceGuard:
    pass                                                                         
    def __init__(self, torch, device="cuda:0"):
        if device != "cuda:0":
            raise ValueError("Only explicit cuda:0 is supported")
        _need(_OWNER.acquire(blocking=False), "Another managed CUDA reranker is already open")
        self.torch, self.device = torch, device
        self._nvml = None
        self._initialized = self._fraction_changed = self._closed = False
        self._operation = threading.RLock()
        self._thread = None
        self._error = None
        self._stats_lock = threading.Lock()
        self._samples, self._previous, self._maximum_gap = 0, None, 0.
        self._peaks, self._last = {}, None
        self.allocator_bytes = None
        try:
            self._psutil = importlib.import_module("psutil")
            self._nvml = importlib.import_module("pynvml")
            self._process = self._psutil.Process()
            _need(torch.cuda.is_available(), "Requested CUDA device is unavailable; no CPU fallback")
            torch.cuda.init()
            properties = torch.cuda.get_device_properties(device)
            self.device_uuid = _physical_uuid(getattr(properties, "uuid", None))
            self._nvml.nvmlInit()
            self._initialized = True
            self._handle = self._nvml.nvmlDeviceGetHandleByUUID(self.device_uuid)
            actual_uuid = _physical_uuid(self._nvml.nvmlDeviceGetUUID(self._handle))
            _need(actual_uuid == self.device_uuid, "CUDA and NVML selected different devices")
            initial = self._observe()
            self.visible_device_bytes = properties.total_memory
            _need(type(self.visible_device_bytes) is int and 0 < self.visible_device_bytes <= initial["GPU_total_bytes"],
                  "Unavailable CUDA-visible memory counter")
            fraction_reader = getattr(torch.cuda, "get_per_process_memory_fraction", None)
            _need(callable(fraction_reader), "CUDA guard requires Torch get_per_process_memory_fraction support")
            self._previous_fraction = fraction_reader(device)
            _need(type(self._previous_fraction) in (int, float)
                  and math.isfinite(self._previous_fraction) and 0 < self._previous_fraction <= 1,
                  "Cannot preserve prior Torch allocator policy")
            self.allocator_bytes = min(CUDA_POLICY["allocator_max_bytes"],
                CUDA_POLICY["physical_GPU_total_unavailable_bytes"]
                - initial["GPU_total_unavailable_including_reserved_bytes"]
                - CUDA_POLICY["nonallocator_headroom_bytes"],
                int(self._previous_fraction * self.visible_device_bytes))
            _need(self.allocator_bytes >= CUDA_POLICY["minimum_allocator_bytes"],
                  "Insufficient physical GPU headroom before weight loading")
            torch.cuda.set_per_process_memory_fraction(self.allocator_bytes / self.visible_device_bytes, device)
            self._fraction_changed = True
            self.check()
        except BaseException:
            self.close()
            raise

    def _observe(self):
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        total, free, used = info.total, info.free, info.used
        _need(all(type(value) is int and 0 <= value <= 2_000_000_000_000 for value in (total, free, used))
              and total > 0 and free <= total and used <= total and free + used <= total,
              "Unavailable or inconsistent physical GPU counters")
        row = {"GPU_total_bytes": total, "GPU_free_bytes": free, "GPU_reported_used_bytes": used,
            "GPU_total_unavailable_including_reserved_bytes": max(used, total-free),
            "host_used_bytes": self._psutil.virtual_memory().used,
            "process_RSS_bytes": self._process.memory_info().rss,
            "torch_allocated_bytes": self.torch.cuda.memory_allocated(self.device),
            "torch_reserved_bytes": self.torch.cuda.memory_reserved(self.device),
            "own_PID_VRAM_bytes": None}
        for key in ("host_used_bytes", "process_RSS_bytes", "torch_allocated_bytes", "torch_reserved_bytes"):
            _need(type(row[key]) is int and row[key] >= 0, "Unavailable resource counters")
        now = time.monotonic()
        with self._stats_lock:
            if self._previous is not None:
                self._maximum_gap = max(self._maximum_gap, now-self._previous)
            self._previous, self._last = now, row
            self._samples += 1
            for key, value in row.items():
                if type(value) is int:
                    self._peaks[key] = max(value, self._peaks.get(key, 0))
                                                                               
                                                                          
        _need(row["GPU_total_unavailable_including_reserved_bytes"] <= CUDA_POLICY["physical_GPU_total_unavailable_bytes"]
              and row["host_used_bytes"] < CUDA_POLICY["host_used_bytes"]
              and row["process_RSS_bytes"] < CUDA_POLICY["process_RSS_bytes"], "CUDA serving resource cap exceeded")
        if self.allocator_bytes is not None:
            _need(max(row["torch_allocated_bytes"], row["torch_reserved_bytes"]) <= self.allocator_bytes,
                  "Torch allocator observation exceeds the admitted limit")
        return row

    def check(self):
        _need(not self._closed, "CUDA resource guard is closed")
        if self._error is not None:
            raise CudaResourceError("CUDA resource monitor failed; close this reranker") from self._error
        try:
            return self._observe()
        except Exception as error:
            self._error = error
            raise

    @contextmanager
    def monitoring(self):
        with self._operation:
            _need(self._thread is None, "Nested CUDA monitoring is unsupported")
            self.check()
            stop = threading.Event()
            def watch():
                while not stop.wait(CUDA_POLICY["sample_interval_seconds"]):
                    try:
                        self.check()
                    except Exception:
                        return
            self._thread = threading.Thread(target=watch, name="magra-cuda-resources", daemon=True)
            self._thread.start()
            try:
                yield self
            finally:
                stop.set()
                self._thread.join()
                self._thread = None
            self.check()

    def observations(self):
        with self._stats_lock:
            return {"policy": dict(CUDA_POLICY), "GPU_UUID": self.device_uuid,
                "CUDA_visible_device_bytes": self.visible_device_bytes,
                "allocator_bytes": self.allocator_bytes, "samples": self._samples,
                "maximum_sample_gap_seconds": self._maximum_gap, "sampled_peaks": dict(self._peaks),
                "last_sample": None if self._last is None else dict(self._last),
                "monitor_failed": self._error is not None, "closed": self._closed,
                "measurement_scope": "sampled during admission, model loading and forward boundaries; idle gaps included"}

    def close(self, *, release_model=None):
        pass                                                                     
        with self._operation:
            if self._closed:
                return
            _need(self._thread is None, "Cannot close an active CUDA operation")
            _need(release_model is None or callable(release_model), "Invalid CUDA model release callback")
            try:
                if release_model is not None:
                    release_model()
                if self._fraction_changed:
                                                                          
                                                                            
                                                                          
                    self.torch.cuda.synchronize(self.device)
                    self.torch.cuda.empty_cache()
            finally:
                self._closed = True
                try:
                    if self._fraction_changed:
                        self.torch.cuda.set_per_process_memory_fraction(self._previous_fraction, self.device)
                finally:
                    try:
                        if self._initialized:
                            self._nvml.nvmlShutdown()
                    finally:
                        _OWNER.release()
