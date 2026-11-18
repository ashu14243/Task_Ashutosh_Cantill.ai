"""Load and validate a company config (YAML or JSON).

A config describes *what* the data looks like; the pipeline code never refers to a
specific company. See configs/company_A.yaml for an annotated example and
configs/TEMPLATE.yaml for every supported key.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

# Canonical fields a source can map. Every one is optional except `id`.
CANONICAL_FIELDS = [
    "id", "amount", "tax_rate", "tax_amount", "currency", "fx_rate",
    "date", "fx_date", "supplier_name", "supplier_id", "description",
    "item_code", "quantity", "unit_price", "po_number", "po_line", "invoice_number",
    "invoice_line", "status", "gl_account", "gl_description", "cost_centre",
    "document_type", "extra_text",
]

DEFAULTS: dict[str, Any] = {
    "reporting_currency": "USD",
    "candidate_name": "ashutosh",
    "output": {"dir": "outputs", "file": None, "diagnostics": True},
    "taxonomy": {"file": None, "level_columns": None, "max_levels": 4},
    "labels": None,
    "fx": None,
    "status": {
        "map": {},              # exact (case-insensitive) value -> include | CANCELLED | NOT_APPROVED
        "use_default_patterns": True,
        "blank": "include",     # what a blank status means
        "unknown": "NOT_APPROVED",
    },
    "non_procurement": {
        "use_default_rules": True,
        "rules": [],            # [{field: gl_description, pattern: '^IC\\d+', reason: NON_PROCUREMENT}]
        "learn_from_labels": True,
        "supplier_propagation": {"enabled": True, "min_lines": 3, "min_share": 0.8},
        "zero_amount": "include",  # include | NON_PROCUREMENT
    },
    "matching": None,           # cross-system matching (company A style)
    "duplicates": {"within_source": []},
    "suppliers": {"fuzzy_threshold": 92, "link_by_supplier_id": True, "min_len_for_fuzzy": 5},
    "items": {"fuzzy_threshold": 95, "generic_code_max_share": 0.05, "generic_code_min_distinct_texts": 25,
              "match_unlabelled_to_catalogue": True, "catalogue_threshold": 90},
    "classification": {
        "text_fields": {"description": 3.0, "gl_description": 1.0, "supplier_name": 1.0, "extra_text": 1.0},
        "use_labels": True,
        "self_training": True,
        "llm": {"enabled": "auto", "provider": "anthropic", "model": None, "base_url": None,
                 "api_key_env": None, "batch_size": 40, "max_unique": 4000, "timeout": 60,
                 "cache_file": ".cache/llm_cache.json", "weight": 0.6},
        "min_confidence_for_leaf": 0.0,
    },
    "dates": {"dayfirst": True, "min_year": 2000, "max_year": 2035, "formats": []},
    "amounts": {"decimal": "auto"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(ValueError):
    pass


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    cfg = _deep_merge(DEFAULTS, raw)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path.resolve())
    cfg["_base_dir"] = str(_resolve_base_dir(path, cfg))
    validate(cfg)
    return cfg


def _resolve_base_dir(path: Path, cfg: dict) -> Path:
    """Relative file paths in a config resolve against `data_dir` (if given), else the
    repo root (parent of configs/), else the config's own folder."""
    if cfg.get("data_dir"):
        d = Path(os.path.expandvars(os.path.expanduser(cfg["data_dir"])))
        return d if d.is_absolute() else (path.parent.parent / d).resolve()
    return path.parent.parent.resolve()


def resolve_path(cfg: dict, p: str | None) -> Path | None:
    if not p:
        return None
    pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
    if pp.is_absolute():
        return pp
    return Path(cfg["_base_dir"]) / pp


def validate(cfg: dict) -> None:
    errs = []
    if not cfg.get("company"):
        errs.append("`company` is required")
    srcs = cfg.get("sources") or []
    if not srcs:
        errs.append("at least one entry in `sources` is required")
    names = set()
    for i, s in enumerate(srcs):
        if not s.get("name"):
            errs.append(f"sources[{i}].name is required")
        if s.get("name") in names:
            errs.append(f"duplicate source name {s.get('name')}")
        names.add(s.get("name"))
        if not s.get("file"):
            errs.append(f"sources[{i}].file is required")
        f = s.get("fields") or {}
        if "id" not in f:
            errs.append(f"sources[{i}].fields.id is required")
        if "amount" not in f:
            errs.append(f"sources[{i}].fields.amount is required")
        for k in f:
            if k not in CANONICAL_FIELDS:
                errs.append(f"sources[{i}].fields.{k} is not a known field ({', '.join(CANONICAL_FIELDS)})")
        at = s.get("amount_type", "net")
        if at not in ("net", "gross"):
            errs.append(f"sources[{i}].amount_type must be net or gross")
    if not (cfg.get("taxonomy") or {}).get("file"):
        errs.append("taxonomy.file is required")
    m = cfg.get("matching")
    if m:
        if m.get("primary") not in names or m.get("secondary") not in names:
            errs.append("matching.primary / matching.secondary must name sources")
    if errs:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errs))
