from __future__ import annotations

from copy import deepcopy
import os
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

DEFAULT_CONFIG_FILE_NAME = "no-watermar.toml"
CONFIG_PATH_ENV_VAR = "NO_WATERMAR_CONFIG"
WATERMARK_KEYWORDS_ENV_VAR = "NO_WATERMAR_WATERMARK_KEYWORDS"
WATERMARK_KEYWORD_PRESETS_ENV_VAR = "NO_WATERMAR_WATERMARK_KEYWORD_PRESETS"
DEFAULT_CONFIG_TEMPLATE_NAME = "default"

DEFAULT_WATERMARK_TOKENS = (
    "copyright",
    "allrights",
    "reserved",
    "watermark",
    ".com",
    ".net",
    ".org",
    ".cn",
)

PROJECT_CONFIG_TEMPLATES = {
    "default": """[watermark_keywords]
active_presets = ["brand"]

[watermark_keywords.presets]
brand = [
  "brandname",
  "brand name",
  "@brand",
]
stock_sites = [
  "example.com",
  "rights reserved",
]
""",
    "brand-social": """[watermark_keywords]
active_presets = ["brand_social"]

[watermark_keywords.presets]
brand_social = [
  "brandname",
  "brand studio",
  "@brand",
  "@brandofficial",
  "brand media",
]
""",
    "stock-marketplaces": """[watermark_keywords]
active_presets = ["stock_marketplaces"]

[watermark_keywords.presets]
stock_marketplaces = [
  "example.com",
  "example.net",
  "rights reserved",
  "licensed preview",
  "all rights reserved",
]
""",
    "mixed-corner-text": """[watermark_keywords]
active_presets = ["brand_social", "generic_corner_text"]

[watermark_keywords.presets]
brand_social = [
  "brandname",
  "@brand",
  "brand official",
]

generic_corner_text = [
  "copyright",
  "rights reserved",
  "watermark",
  "do not repost",
]
""",
    "stable-public": """[watermark_keywords]
active_presets = ["stock_sites"]

[watermark_keywords.presets]
stock_sites = [
  "example.com",
  "rights reserved",
]

[profiles.datasets.local_smoke]
input = "./inputs"
recursive = false
limit = 2
benchmark_dataset = "regular_corner_text"

[profiles.providers.seed_telea]
mask_provider = "seed_manifest"
restore_provider = "telea"
ocr_session_mode = "auto"

[profiles.providers.ocr_telea]
mask_provider = "paddleocr"
restore_provider = "telea"
ocr_session_mode = "persistent"

[profiles.providers.lama_eval]
mask_provider = "seed_manifest"
restore_provider = "lama"
ocr_session_mode = "auto"

[profiles.providers.ocr_corner_crop]
mask_provider = "paddleocr"
restore_provider = "corner_crop"
ocr_session_mode = "persistent"

[profiles.providers.ocr_corner_crop.restore_options]
edge_tolerance = 24
""",
}


@dataclass(frozen=True, slots=True)
class WatermarkKeywordSettings:
    tokens: tuple[str, ...]
    active_presets: tuple[str, ...]
    config_path: Path | None


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    name: str
    input_root: Path | None
    recursive: bool | None
    limit: int | None
    benchmark_dataset: str | None
    config_path: Path | None


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    name: str
    mask_provider: str | None
    restore_provider: str | None
    ocr_session_mode: str | None
    restore_prompt: str | None
    restore_negative_prompt: str | None
    restore_options: dict[str, Any]
    config_path: Path | None


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    active_presets: tuple[str, ...]
    presets: dict[str, tuple[str, ...]]
    dataset_profiles: dict[str, DatasetProfile]
    provider_profiles: dict[str, ProviderProfile]
    path: Path | None


def get_project_config(
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> ProjectConfig:
    resolved_config_path = resolve_config_path(config_path=config_path, start_dir=start_dir)
    return _load_project_config(resolved_config_path)


def get_watermark_keyword_settings(
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> WatermarkKeywordSettings:
    config = get_project_config(start_dir=start_dir, config_path=config_path)
    preset_names = _merge_active_preset_names(
        config.active_presets,
        _split_csv(os.getenv(WATERMARK_KEYWORD_PRESETS_ENV_VAR, "")),
    )
    tokens = list(_normalize_tokens(DEFAULT_WATERMARK_TOKENS))

    missing_presets: list[str] = []
    for preset_name in preset_names:
        preset_tokens = config.presets.get(preset_name)
        if preset_tokens is None:
            missing_presets.append(preset_name)
            continue
        tokens.extend(_normalize_tokens(preset_tokens))

    if missing_presets:
        joined_names = ", ".join(missing_presets)
        raise ValueError(
            f"Unknown watermark keyword preset(s): {joined_names}. "
            f"Define them in {config.path or DEFAULT_CONFIG_FILE_NAME}."
        )

    tokens.extend(_normalize_tokens(_split_csv(os.getenv(WATERMARK_KEYWORDS_ENV_VAR, ""))))
    return WatermarkKeywordSettings(
        tokens=tuple(_dedupe_preserve_order(tokens)),
        active_presets=tuple(preset_names),
        config_path=config.path,
    )


def resolve_dataset_profile(
    profile_name: str,
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> DatasetProfile:
    config = get_project_config(start_dir=start_dir, config_path=config_path)
    normalized_name = _normalize_profile_name(profile_name)
    profile = config.dataset_profiles.get(normalized_name)
    if profile is None:
        raise ValueError(
            f"Unknown dataset profile: {profile_name}. "
            f"Available dataset profiles: {', '.join(sorted(config.dataset_profiles)) or 'none'}."
        )
    return profile


def resolve_provider_profile(
    profile_name: str,
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> ProviderProfile:
    config = get_project_config(start_dir=start_dir, config_path=config_path)
    normalized_name = _normalize_profile_name(profile_name)
    profile = config.provider_profiles.get(normalized_name)
    if profile is None:
        raise ValueError(
            f"Unknown provider profile: {profile_name}. "
            f"Available provider profiles: {', '.join(sorted(config.provider_profiles)) or 'none'}."
        )
    return profile


def dataset_profile_to_dict(profile: DatasetProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "input_root": str(profile.input_root) if profile.input_root else None,
        "recursive": profile.recursive,
        "limit": profile.limit,
        "benchmark_dataset": profile.benchmark_dataset,
        "config_path": str(profile.config_path) if profile.config_path else None,
    }


def provider_profile_to_dict(profile: ProviderProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "mask_provider": profile.mask_provider,
        "restore_provider": profile.restore_provider,
        "ocr_session_mode": profile.ocr_session_mode,
        "restore_prompt": profile.restore_prompt,
        "restore_negative_prompt": profile.restore_negative_prompt,
        "restore_options": deepcopy(profile.restore_options),
        "config_path": str(profile.config_path) if profile.config_path else None,
    }


def show_project_config(
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    summary = validate_project_config(start_dir=start_dir, config_path=config_path)
    summary["command"] = "config show"
    return summary


def validate_project_config(
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, object]:
    search_root = (start_dir or Path.cwd()).resolve()
    resolved_config_path = resolve_config_path(config_path=config_path, start_dir=search_root)
    config = _load_project_config(resolved_config_path)
    settings = get_watermark_keyword_settings(start_dir=search_root, config_path=config_path)

    env_preset_names = [
        _normalize_preset_name(name)
        for name in _split_csv(os.getenv(WATERMARK_KEYWORD_PRESETS_ENV_VAR, ""))
        if _normalize_preset_name(name)
    ]
    env_extra_keywords = _normalize_tokens(_split_csv(os.getenv(WATERMARK_KEYWORDS_ENV_VAR, "")))
    warnings: list[str] = []
    if resolved_config_path is None:
        warnings.append(
            "No local no-watermar.toml file was found. Validation used built-in defaults and environment overrides only."
        )
    if not config.presets:
        warnings.append("No named watermark keyword presets are defined in the project config.")
    if not settings.active_presets:
        warnings.append("No named watermark keyword presets are active.")

    return {
        "command": "config validate",
        "status": "ok",
        "config": {
            "path": str(resolved_config_path) if resolved_config_path else None,
            "requested_path": str(config_path.expanduser().resolve()) if config_path else None,
            "search_root": str(search_root),
            "resolution_mode": _resolve_config_mode(config_path, resolved_config_path),
            "available_presets": sorted(config.presets.keys()),
            "configured_active_presets": list(config.active_presets),
        },
        "environment": {
            "config_override": os.getenv(CONFIG_PATH_ENV_VAR) or None,
            "keyword_preset_overrides": env_preset_names,
            "extra_keyword_overrides": env_extra_keywords,
        },
        "watermark_keywords": {
            "default_tokens": list(DEFAULT_WATERMARK_TOKENS),
            "effective_tokens": list(settings.tokens),
            "effective_count": len(settings.tokens),
            "active_presets": list(settings.active_presets),
        },
        "profiles": {
            "dataset_profiles": {
                name: dataset_profile_to_dict(profile)
                for name, profile in sorted(config.dataset_profiles.items())
            },
            "provider_profiles": {
                name: provider_profile_to_dict(profile)
                for name, profile in sorted(config.provider_profiles.items())
            },
        },
        "warnings": warnings,
    }


def init_project_config(
    start_dir: Path | None = None,
    *,
    config_path: Path | None = None,
    template_name: str = DEFAULT_CONFIG_TEMPLATE_NAME,
    force: bool = False,
) -> dict[str, object]:
    resolved_start_dir = (start_dir or Path.cwd()).resolve()
    target_path = resolve_init_config_path(config_path=config_path, start_dir=resolved_start_dir)
    normalized_template_name = _normalize_template_name(template_name)
    template = PROJECT_CONFIG_TEMPLATES.get(normalized_template_name)
    if template is None:
        available_templates = ", ".join(list_config_template_names())
        raise ValueError(
            f"Unknown config template: {template_name}. "
            f"Available templates: {available_templates}."
        )

    existed = target_path.exists()
    if existed and not force:
        raise FileExistsError(f"Config file already exists: {target_path}. Use --force to overwrite it.")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(template, encoding="utf-8")

    return {
        "command": "config init",
        "status": "ok",
        "config_path": str(target_path),
        "template": normalized_template_name,
        "force": force,
        "overwritten": existed,
        "available_templates": list(list_config_template_names()),
        "size_bytes": target_path.stat().st_size,
    }


def list_config_template_names() -> tuple[str, ...]:
    return tuple(PROJECT_CONFIG_TEMPLATES.keys())


def resolve_init_config_path(
    config_path: Path | None = None,
    *,
    start_dir: Path | None = None,
) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    configured_path = os.getenv(CONFIG_PATH_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return ((start_dir or Path.cwd()).resolve() / DEFAULT_CONFIG_FILE_NAME)


def resolve_config_path(
    config_path: Path | None = None,
    *,
    start_dir: Path | None = None,
) -> Path | None:
    if config_path is not None:
        return config_path.expanduser().resolve()
    return discover_config_path(start_dir=start_dir)


def discover_config_path(start_dir: Path | None = None) -> Path | None:
    configured_path = os.getenv(CONFIG_PATH_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser().resolve()

    search_root = (start_dir or Path.cwd()).resolve()
    for directory in (search_root, *search_root.parents):
        candidate = directory / DEFAULT_CONFIG_FILE_NAME
        if candidate.exists():
            return candidate
    return None


def _load_project_config(config_path: Path | None) -> ProjectConfig:
    if config_path is None:
        return ProjectConfig(
            active_presets=(),
            presets={},
            dataset_profiles={},
            provider_profiles={},
            path=None,
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Configured no-watermar config does not exist: {config_path}")

    raw_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    keyword_table = _read_table(raw_data.get("watermark_keywords", {}), config_path, "watermark_keywords")

    active_presets = _read_string_list(
        keyword_table.get("active_presets", []),
        config_path,
        "watermark_keywords.active_presets",
    )
    raw_presets = _read_table(keyword_table.get("presets", {}), config_path, "watermark_keywords.presets")

    presets: dict[str, tuple[str, ...]] = {}
    for name, tokens in raw_presets.items():
        normalized_name = _normalize_preset_name(name)
        if not normalized_name:
            continue
        presets[normalized_name] = _read_string_list(
            tokens,
            config_path,
            f"watermark_keywords.presets.{name}",
        )

    profiles_table = _read_table(raw_data.get("profiles", {}), config_path, "profiles")
    raw_dataset_profiles = _read_table(profiles_table.get("datasets", {}), config_path, "profiles.datasets")
    raw_provider_profiles = _read_table(profiles_table.get("providers", {}), config_path, "profiles.providers")

    dataset_profiles: dict[str, DatasetProfile] = {}
    for name, raw_profile in raw_dataset_profiles.items():
        field_name = f"profiles.datasets.{name}"
        profile_table = _read_table(raw_profile, config_path, field_name)
        normalized_name = _normalize_profile_name(name)
        if not normalized_name:
            continue
        dataset_profiles[normalized_name] = DatasetProfile(
            name=normalized_name,
            input_root=_read_optional_path(profile_table.get("input"), config_path, f"{field_name}.input"),
            recursive=_read_optional_bool(profile_table.get("recursive"), config_path, f"{field_name}.recursive"),
            limit=_read_optional_int(profile_table.get("limit"), config_path, f"{field_name}.limit"),
            benchmark_dataset=_read_optional_string(
                profile_table.get("benchmark_dataset"),
                config_path,
                f"{field_name}.benchmark_dataset",
            ),
            config_path=config_path,
        )

    provider_profiles: dict[str, ProviderProfile] = {}
    for name, raw_profile in raw_provider_profiles.items():
        field_name = f"profiles.providers.{name}"
        profile_table = _read_table(raw_profile, config_path, field_name)
        normalized_name = _normalize_profile_name(name)
        if not normalized_name:
            continue
        provider_profiles[normalized_name] = ProviderProfile(
            name=normalized_name,
            mask_provider=_read_optional_string(
                profile_table.get("mask_provider"),
                config_path,
                f"{field_name}.mask_provider",
            ),
            restore_provider=_read_optional_string(
                profile_table.get("restore_provider"),
                config_path,
                f"{field_name}.restore_provider",
            ),
            ocr_session_mode=_read_optional_string(
                profile_table.get("ocr_session_mode"),
                config_path,
                f"{field_name}.ocr_session_mode",
            ),
            restore_prompt=_read_optional_string(
                profile_table.get("restore_prompt"),
                config_path,
                f"{field_name}.restore_prompt",
            ),
            restore_negative_prompt=_read_optional_string(
                profile_table.get("restore_negative_prompt"),
                config_path,
                f"{field_name}.restore_negative_prompt",
            ),
            restore_options=_read_optional_table(
                profile_table.get("restore_options"),
                config_path,
                f"{field_name}.restore_options",
            ),
            config_path=config_path,
        )

    return ProjectConfig(
        active_presets=tuple(
            normalized_name
            for normalized_name in (_normalize_preset_name(name) for name in active_presets)
            if normalized_name
        ),
        presets=presets,
        dataset_profiles=dataset_profiles,
        provider_profiles=provider_profiles,
        path=config_path,
    )


def _resolve_config_mode(config_path: Path | None, resolved_config_path: Path | None) -> str:
    if config_path is not None:
        return "explicit-path"
    if os.getenv(CONFIG_PATH_ENV_VAR):
        return "env-var"
    if resolved_config_path is not None:
        return "auto-discovery"
    return "defaults-only"


def _read_table(value: object, config_path: Path, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(_invalid_config_message(config_path, f"{field_name} must be a TOML table"))
    return value


def _read_optional_table(value: object, config_path: Path, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(_read_table(value, config_path, field_name))


def _read_string_list(value: object, config_path: Path, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(_invalid_config_message(config_path, f"{field_name} must be an array of strings"))

    values: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError(_invalid_config_message(config_path, f"{field_name} must contain only strings"))
        stripped = entry.strip()
        if stripped:
            values.append(stripped)
    return tuple(values)


def _read_optional_string(value: object, config_path: Path, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(_invalid_config_message(config_path, f"{field_name} must be a string"))
    stripped = value.strip()
    return stripped or None


def _read_optional_bool(value: object, config_path: Path, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(_invalid_config_message(config_path, f"{field_name} must be a boolean"))
    return value


def _read_optional_int(value: object, config_path: Path, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(_invalid_config_message(config_path, f"{field_name} must be an integer"))
    return value


def _read_optional_path(value: object, config_path: Path, field_name: str) -> Path | None:
    raw_path = _read_optional_string(value, config_path, field_name)
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _invalid_config_message(config_path: Path, message: str) -> str:
    return f"Invalid no-watermar config at {config_path}: {message}"


def _merge_active_preset_names(*preset_groups: tuple[str, ...]) -> list[str]:
    merged: list[str] = []
    for preset_group in preset_groups:
        for preset_name in preset_group:
            normalized_name = _normalize_preset_name(preset_name)
            if normalized_name:
                merged.append(normalized_name)
    return _dedupe_preserve_order(merged)


def _normalize_preset_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_profile_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_template_name(name: str) -> str:
    return str(name).strip().lower()


def _normalize_tokens(tokens: tuple[str, ...] | list[str]) -> list[str]:
    normalized_tokens: list[str] = []
    for token in tokens:
        normalized = "".join(token.strip().lower().split())
        if normalized:
            normalized_tokens.append(normalized)
    return normalized_tokens


def _split_csv(raw_value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _dedupe_preserve_order(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
