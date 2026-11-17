                                                                                    

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Mapping


DEFAULT_MODEL = "minimax/minimax-m3.1-flash:free"
PAID_MODEL = "minimax/minimax-m3.1-flash"
PAID_LEDGER_PATH = (Path(__file__).resolve().parents[1] /
                    "data/train-unit-ranker-paid-staging-v1/paid-data/runs/spend.sqlite")
_PAID_KEYS = ("ai-model-key-increased-paid", "ai-model-paid-key")
_FIELDS = {"OPENROUTER_API_KEY", "ai-model-key", "OPENROUTER_MODEL", *_PAID_KEYS}
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z")


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str | None = field(repr=False)
    model_id: str
    profile: str = "legacy"

    @property
    def key_configured(self) -> bool:
        return bool(self.api_key)


def _dotenv(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    target = Path(path)
    try:
        with target.open("rb") as source:
            raw = source.read(65_537)
    except FileNotFoundError:
        return {}
    except OSError:
        raise ValueError("Cannot read project environment file") from None
    if len(raw) > 65_536:
        raise ValueError("Project environment file exceeds 64 KiB")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeError:
        raise ValueError("Project environment file must use UTF-8") from None
    result = {}
    for original in content.splitlines():
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in _FIELDS:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            closing = value.find(quote, 1)
            trailer = value[closing + 1:].strip()
            if closing < 0 or (trailer and not trailer.startswith("#")):
                raise ValueError("Invalid quoted OpenRouter environment setting")
            value = value[1:closing]
        else:
                                                                                 
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if "\x00" in value or "\r" in value or "\n" in value:
            raise ValueError("Invalid OpenRouter environment setting")
        if key in _PAID_KEYS and key in result:
            raise ValueError("Exactly one entry per paid credential setting is required")
        result[key] = value
    return result


def load_openrouter_settings(*, env_file: str | Path | None = ".env",
                             model_id: str | None = None, api_key: str | None = None,
                             environ: Mapping[str, str] | None = None,
                             profile: str = "legacy") -> OpenRouterSettings:
    pass                                                                              

                                                                                                
                                                                               
                                                                 
       
    if profile not in {"auto", "paid", "legacy"}:
        raise ValueError("OpenRouter profile must be auto, paid or legacy")
    environment = os.environ if environ is None else environ
                                                                             
    local = {} if api_key is not None else _dotenv(env_file)
    if profile == "auto":
        if api_key is not None and model_id != PAID_MODEL:
            profile = "legacy"
        else:
            declared = any(name in source for source in (environment, local) for name in _PAID_KEYS)
            profile = "paid" if model_id == PAID_MODEL or model_id is None and declared else "legacy"
    if profile == "paid":
        if model_id is not None and model_id != PAID_MODEL:
            raise ValueError("Paid profile requires the exact main MiniMax model")
        selected_model, selected_key = PAID_MODEL, api_key
        if selected_key is None:
            for name in _PAID_KEYS:
                for source in (environment, local):
                    if name in source:
                        selected_key = source[name]
                        break
                if selected_key is not None:
                    break
    else:
        selected_model = model_id if model_id is not None else environment.get(
            "OPENROUTER_MODEL", local.get("OPENROUTER_MODEL", DEFAULT_MODEL))
        selected_key = api_key
        if selected_key is None:
            for source in (environment, local):
                for name in ("OPENROUTER_API_KEY", "ai-model-key"):
                    if name in source:
                        selected_key = source[name]
                        break
                if selected_key is not None:
                    break
    if not isinstance(selected_model, str) or not _MODEL.fullmatch(selected_model):
        raise ValueError("Invalid OpenRouter model identifier")
    if selected_key is not None and (not isinstance(selected_key, str)
            or len(selected_key) > 8192 or any(c.isspace() or ord(c) < 32 for c in selected_key)):
        raise ValueError("Invalid OpenRouter credential setting")
    if selected_model == PAID_MODEL and profile != "paid":
        raise ValueError("The main paid model requires the paid profile and credential")
    return OpenRouterSettings(api_key=selected_key or None, model_id=selected_model, profile=profile)


def paid_controls(*, max_total_usd=None, shared_ledger_path=None,
                  prompt_price=None, completion_price=None) -> dict:
    pass                                                                              
    limit = 100.0 if max_total_usd is None else max_total_usd
    if (isinstance(limit, bool) or not isinstance(limit, (int, float))
            or not math.isfinite(limit) or not 0 <= limit <= 100):
        raise ValueError("Paid UTC monthly spending cap must be between 0 and 100 USD")
    expected = PAID_LEDGER_PATH.absolute()
    selected = expected if shared_ledger_path is None else Path(shared_ledger_path).absolute()
    for candidate in (expected, selected):
        for part in (candidate, *candidate.parents):
            if part.exists() and (part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
                raise ValueError("Shared paid ledger links and reparse points are forbidden")
    if selected.resolve() != expected.resolve():
        raise ValueError("Paid answers and judges must reuse the existing project ledger")
    prices = []
    for value, exact in ((prompt_price, .15), (completion_price, .60)):
        value = exact if value is None else value
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value != exact):
            raise ValueError("Paid prompt/output ceilings must be 0.15/0.60 USD per million")
        prices.append(float(value))
    return {"max_total_usd": float(limit), "shared_ledger_path": expected.resolve(),
            "prompt_price": prices[0], "completion_price": prices[1]}
