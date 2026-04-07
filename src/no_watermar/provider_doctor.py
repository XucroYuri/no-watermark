from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .benchmark_providers import list_provider_descriptors
from .env_loader import read_local_env_file
from .provider_runtime import probe_python_info


SIDECAR_SLOTS = (
    {
        "provider_name": "paddleocr",
        "env_var": "NO_WATERMAR_PADDLEOCR_PYTHON",
        "script_relative_path": Path("tools/sidecars/paddleocr_mask.py"),
        "setup_doc": "docs/setup/provider-sidecars.md",
        "kind": "mask",
    },
    {
        "provider_name": "lama",
        "env_var": "NO_WATERMAR_LAMA_PYTHON",
        "script_relative_path": Path("tools/sidecars/lama_restore.py"),
        "setup_doc": "docs/setup/provider-sidecars.md",
        "kind": "restore",
    },
    {
        "provider_name": "diffusers_inpaint",
        "env_var": "NO_WATERMAR_DIFFUSERS_PYTHON",
        "script_relative_path": Path("tools/sidecars/diffusers_restore.py"),
        "setup_doc": "docs/setup/provider-sidecars.md",
        "kind": "restore",
    },
    {
        "provider_name": "powerpaint_v2_1",
        "env_var": "NO_WATERMAR_POWERPAINT_PYTHON",
        "script_relative_path": Path("tools/sidecars/powerpaint_restore.py"),
        "setup_doc": "docs/setup/provider-sidecars.md",
        "kind": "restore",
    },
    {
        "provider_name": "brushnet",
        "env_var": "NO_WATERMAR_BRUSHNET_PYTHON",
        "script_relative_path": Path("tools/sidecars/brushnet_restore.py"),
        "setup_doc": "docs/setup/provider-sidecars.md",
        "kind": "restore",
    },
)

SIDECAR_COMPATIBILITY = {
    "paddleocr": {
        "validated_python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
        "validated_packages": ["paddleocr", "paddlepaddle or the matching runtime for the target machine"],
        "status": "broadly-supported",
        "notes": [
            "Prefer a dedicated sidecar environment instead of the repository default Python.",
            "Validate direct importability first because Paddle runtime compatibility varies by machine.",
        ],
    },
    "lama": {
        "validated_python_versions": ["3.12"],
        "validated_packages": ["simple-lama-inpainting"],
        "status": "validated-local-path",
        "notes": [
            "The current validated local path uses Python 3.12 with simple-lama-inpainting.",
            "Treat newer interpreter versions as unvalidated until they are re-tested in this repository.",
        ],
    },
    "diffusers_inpaint": {
        "validated_python_versions": ["3.12"],
        "validated_packages": ["torch", "diffusers", "transformers", "accelerate", "safetensors"],
        "status": "validated-local-path",
        "notes": [
            "The current validated local path uses Python 3.12 with torch 2.11.0+cu128 and diffusers 0.37.1.",
            "The current validated smoke path uses single-file Stable Diffusion inpainting weights plus an explicit original_config URL.",
            "This provider still needs restore_options.model_id or NO_WATERMAR_DIFFUSERS_MODEL at execution time.",
        ],
    },
    "powerpaint_v2_1": {
        "validated_python_versions": ["3.12"],
        "validated_packages": ["torch", "diffusers", "transformers", "safetensors", "powerpaint"],
        "status": "validated-local-path",
        "notes": [
            "The current validated local path uses Python 3.12 with torch 2.11.0+cu128, diffusers 0.27.0, transformers 4.41.2, huggingface-hub 0.25.2, and mmengine 0.10.7.",
            "The current validated smoke path uses the JunhaoZhuang/PowerPaint-v2-1 checkpoint plus the bundled realisticVisionV60B1_v51VAE base model folder in local-only mode.",
            "It still needs restore_options.checkpoint_dir or NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR at execution time.",
        ],
    },
    "brushnet": {
        "validated_python_versions": ["3.12"],
        "validated_packages": [
            "torch",
            "transformers",
            "accelerate",
            "editable install of the BrushNet upstream diffusers fork or NO_WATERMAR_BRUSHNET_SOURCE_DIR",
        ],
        "status": "validated-local-path",
        "notes": [
            "The current validated local path reuses the repository-local PowerPaint Python 3.12 environment plus NO_WATERMAR_BRUSHNET_SOURCE_DIR pointing at the upstream BrushNet clone.",
            "The current validated smoke path uses the official segmentation_mask_brushnet_ckpt weights with the bundled realisticVisionV60B1_v51VAE base model folder from the local PowerPaint assets.",
            "It still needs restore_options.brushnet_model_path or NO_WATERMAR_BRUSHNET_MODEL at execution time.",
        ],
    },
}


def build_provider_doctor_report(project_root: Path | None = None) -> dict[str, Any]:
    resolved_project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    env_path = resolved_project_root / ".env"
    env_file_values = read_local_env_file(env_path)
    descriptors = list_provider_descriptors()
    descriptor_index = _index_provider_descriptors(descriptors)

    sidecars = [
        _describe_sidecar_slot(
            descriptor_index=descriptor_index,
            project_root=resolved_project_root,
            env_file_values=env_file_values,
            slot=slot,
        )
        for slot in SIDECAR_SLOTS
    ]
    warnings = _collect_warnings(sidecars=sidecars, env_path=env_path, env_file_values=env_file_values)
    recommendations = _collect_recommendations(sidecars)
    return {
        "status": "ok",
        "project_root": str(resolved_project_root),
        "current_python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "env_file": {
            "path": str(env_path),
            "exists": env_path.exists(),
            "resolved_values": {
                key: value for key, value in env_file_values.items() if key in {slot["env_var"] for slot in SIDECAR_SLOTS}
            },
        },
        "summary": _summarize_provider_descriptors(descriptors),
        "mask_providers": descriptors["mask_providers"],
        "restore_providers": descriptors["restore_providers"],
        "compatibility_matrix": _build_compatibility_matrix(resolved_project_root),
        "sidecars": sidecars,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def _describe_sidecar_slot(
    *,
    descriptor_index: dict[str, dict[str, Any]],
    project_root: Path,
    env_file_values: dict[str, str],
    slot: dict[str, Any],
) -> dict[str, Any]:
    provider_name = str(slot["provider_name"])
    env_var = str(slot["env_var"])
    script_path = project_root / Path(str(slot["script_relative_path"]))
    descriptor = descriptor_index.get(provider_name)
    process_env_value = os.getenv(env_var)
    env_file_value = env_file_values.get(env_var)
    configured_python = process_env_value or env_file_value
    if process_env_value:
        configured_source = "process_env"
    elif env_file_value:
        configured_source = "env_file"
    else:
        configured_source = None
    python_info = probe_python_info(configured_python) if configured_python else None
    compatibility = _describe_compatibility(provider_name, python_info)

    return {
        "provider_name": provider_name,
        "kind": slot["kind"],
        "env_var": env_var,
        "configured_python": configured_python,
        "configured_source": configured_source,
        "process_env_value": process_env_value,
        "env_file_value": env_file_value,
        "python_path_exists": Path(configured_python).exists() if configured_python else None,
        "configured_python_version": python_info.get("python_version") if python_info else None,
        "configured_python_probe_error": python_info.get("error") if python_info and not python_info.get("ok") else None,
        "script_path": str(script_path),
        "script_exists": script_path.exists(),
        "setup_doc": str(project_root / str(slot["setup_doc"])),
        "runtime_available": bool(descriptor and descriptor.get("runtime_available")),
        "runtime_note": descriptor.get("runtime_note") if descriptor else None,
        "runtime_probe": descriptor.get("runtime_probe") if descriptor else None,
        "default_mode": descriptor.get("default_mode") if descriptor else None,
        "execution_modes": list(descriptor.get("execution_modes") or []) if descriptor else [],
        "compatibility": compatibility,
    }


def _collect_warnings(
    *,
    sidecars: list[dict[str, Any]],
    env_path: Path,
    env_file_values: dict[str, str],
) -> list[str]:
    warnings: list[str] = []
    if env_path.exists() and not env_file_values:
        warnings.append(f"Repo-local .env exists but does not define any provider interpreter variables: {env_path}")

    for sidecar in sidecars:
        env_var = sidecar["env_var"]
        process_env_value = sidecar["process_env_value"]
        env_file_value = sidecar["env_file_value"]
        if env_file_value and not process_env_value:
            warnings.append(
                f"{env_var} is defined in .env but not currently exported in this process. "
                "CLI entrypoints that load .env will see it; direct library calls will not."
            )
        if sidecar["configured_python"] and sidecar["python_path_exists"] is False:
            warnings.append(f"{env_var} points to a missing interpreter: {sidecar['configured_python']}")
        if sidecar.get("configured_python_probe_error"):
            warnings.append(
                f"{env_var} exists but its Python version could not be read cleanly: {sidecar['configured_python_probe_error']}"
            )
        compatibility_status = (sidecar.get("compatibility") or {}).get("status")
        compatibility_note = (sidecar.get("compatibility") or {}).get("note")
        if compatibility_status == "unvalidated" and compatibility_note:
            warnings.append(f"{env_var} uses an unvalidated Python version: {compatibility_note}")
        if not sidecar["script_exists"]:
            warnings.append(f"Provider sidecar script is missing for {sidecar['provider_name']}: {sidecar['script_path']}")
    return warnings


def _collect_recommendations(sidecars: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    for sidecar in sidecars:
        env_var = sidecar["env_var"]
        provider_name = sidecar["provider_name"]
        runtime_probe = sidecar.get("runtime_probe") or {}
        if not sidecar["configured_python"] and not sidecar["runtime_available"]:
            recommendations.append(
                f"Configure {env_var} for {provider_name} or install the provider directly in the current Python environment."
            )
            continue
        if sidecar["configured_python"] and sidecar["python_path_exists"] is False:
            recommendations.append(f"Fix {env_var} so it points to an existing Python interpreter for {provider_name}.")
            continue
        if (sidecar.get("compatibility") or {}).get("status") == "unresolved":
            recommendations.append(
                f"Validate the configured interpreter behind {env_var} before using {provider_name}: "
                f"{(sidecar.get('compatibility') or {}).get('note')}"
            )
            continue
        if (sidecar.get("compatibility") or {}).get("status") == "unvalidated":
            validated_versions = list((sidecar.get("compatibility") or {}).get("validated_python_versions") or [])
            if validated_versions:
                recommendations.append(
                    f"Recreate the {provider_name} sidecar with a validated Python version: "
                    f"{', '.join(validated_versions)}."
                )
            else:
                recommendations.append(
                    f"Treat the {provider_name} sidecar as experimental until this repository records a validated Python version."
                )
            continue
        if sidecar["configured_python"] and not sidecar["runtime_available"]:
            error = runtime_probe.get("error")
            if error:
                recommendations.append(f"Install or repair {provider_name} in {sidecar['configured_python']}: {error}")
            else:
                recommendations.append(f"Validate the configured {provider_name} sidecar interpreter with providers probe.")
            continue
        if sidecar["runtime_available"] and sidecar["configured_source"] == "env_file":
            recommendations.append(
                f"{provider_name} is configured via the repo-local .env. Use the root CLI launcher or `python -m no_watermar.cli` to pick it up automatically."
            )

    return _unique_preserve_order(recommendations)


def _index_provider_descriptors(descriptors: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for group_name in ("mask_providers", "restore_providers"):
        for descriptor in descriptors.get(group_name, []):
            name = descriptor.get("name")
            if isinstance(name, str):
                index[name] = descriptor
    return index


def _build_compatibility_matrix(project_root: Path) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for slot in SIDECAR_SLOTS:
        provider_name = str(slot["provider_name"])
        entry = SIDECAR_COMPATIBILITY.get(provider_name, {})
        matrix[provider_name] = {
            "provider_name": provider_name,
            "env_var": str(slot["env_var"]),
            "kind": str(slot["kind"]),
            "setup_doc": str(project_root / str(slot["setup_doc"])),
            "validated_python_versions": list(entry.get("validated_python_versions") or []),
            "validated_packages": list(entry.get("validated_packages") or []),
            "status": str(entry.get("status") or "unspecified"),
            "notes": list(entry.get("notes") or []),
        }
    return matrix


def _describe_compatibility(provider_name: str, python_info: dict[str, Any] | None) -> dict[str, Any]:
    entry = SIDECAR_COMPATIBILITY.get(provider_name, {})
    validated_versions = list(entry.get("validated_python_versions") or [])
    configured_version = str((python_info or {}).get("python_version") or "")
    configured_major_minor = ".".join(configured_version.split(".")[:2]) if configured_version else None
    status = "unknown"
    note = "No sidecar interpreter is configured."
    if python_info is not None and not python_info.get("ok"):
        status = "unresolved"
        note = str(python_info.get("error") or "Unable to read the configured interpreter version.")
    if configured_major_minor is not None:
        if configured_major_minor in validated_versions:
            status = "validated"
            note = f"Python {configured_major_minor} is in the validated set for {provider_name}."
        else:
            status = "unvalidated"
            note = (
                f"Python {configured_major_minor} is outside the validated set for {provider_name}: "
                f"{', '.join(validated_versions) or 'none documented'}."
            )

    return {
        "status": status,
        "configured_python_version": configured_version or None,
        "validated_python_versions": validated_versions,
        "validated_packages": list(entry.get("validated_packages") or []),
        "note": note,
    }


def _summarize_provider_descriptors(descriptors: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    all_descriptors = list(descriptors.get("mask_providers", [])) + list(descriptors.get("restore_providers", []))
    implemented = [descriptor for descriptor in all_descriptors if descriptor.get("implemented")]
    planned = [descriptor for descriptor in all_descriptors if not descriptor.get("implemented")]
    available = [descriptor for descriptor in implemented if descriptor.get("runtime_available")]
    unavailable = [descriptor for descriptor in implemented if not descriptor.get("runtime_available")]
    return {
        "implemented_total": len(implemented),
        "implemented_available": len(available),
        "implemented_unavailable": len(unavailable),
        "planned_total": len(planned),
    }


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
