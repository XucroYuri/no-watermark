from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import tomllib

from ...config import (
    list_config_template_names,
    resolve_config_path,
    show_project_config,
    init_project_config,
    validate_project_config,
)
from ...io_utils import print_json

CONFIG_COMMAND_ERRORS = (OSError, ValueError, tomllib.TOMLDecodeError)


def configure_config_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("config", help="Inspect and bootstrap local configuration.")
    config_subparsers = parser.add_subparsers(dest="config_command", required=True)

    show_parser = config_subparsers.add_parser("show", help="Show the effective local project config and keyword settings.")
    _add_config_read_args(show_parser)
    show_parser.set_defaults(handler=_handle_show)

    validate_parser = config_subparsers.add_parser("validate", help="Validate the local project config and effective keyword settings.")
    _add_config_read_args(validate_parser)
    validate_parser.set_defaults(handler=_handle_validate)

    init_parser = config_subparsers.add_parser("init", help="Create a local no-watermar.toml file from a built-in template.")
    init_parser.add_argument("--config", type=Path, default=None, help="Optional explicit output path for the generated config file.")
    init_parser.add_argument(
        "--start-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory used for the default config path when --config is not provided.",
    )
    init_parser.add_argument(
        "--template",
        default="default",
        choices=list_config_template_names(),
        help="Built-in config template name.",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite the target config file if it already exists.")
    init_parser.set_defaults(handler=_handle_init)


def _handle_show(args: argparse.Namespace) -> int:
    return _emit_config_command("show", args, lambda: show_project_config(start_dir=args.start_dir, config_path=args.config))


def _handle_validate(args: argparse.Namespace) -> int:
    return _emit_config_command("validate", args, lambda: validate_project_config(start_dir=args.start_dir, config_path=args.config))


def _handle_init(args: argparse.Namespace) -> int:
    return _emit_config_command(
        "init",
        args,
        lambda: init_project_config(
            start_dir=args.start_dir,
            config_path=args.config,
            template_name=args.template,
            force=args.force,
        ),
    )


def _emit_config_command(command: str, args: argparse.Namespace, action: Callable[[], dict[str, object]]) -> int:
    try:
        summary = action()
    except CONFIG_COMMAND_ERRORS as exc:
        print_json(_config_error_payload(command, args.config, args.start_dir, exc))
        return 2

    print_json(summary)
    return 0


def _add_config_read_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Optional explicit path to a no-watermar.toml file.")
    parser.add_argument(
        "--start-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory used for automatic config discovery when --config is not provided.",
    )


def _config_error_payload(command: str, config_path: Path | None, start_dir: Path, error: Exception) -> dict[str, object]:
    resolved_path = resolve_config_path(config_path=config_path, start_dir=start_dir)
    if command == "init" and config_path is not None:
        resolved_path = config_path.expanduser().resolve()
    if command == "init" and config_path is None and resolved_path is None:
        resolved_path = start_dir.resolve() / "no-watermar.toml"
    return {
        "command": f"config {command}",
        "status": "error",
        "config_path": str(resolved_path) if resolved_path else None,
        "error": str(error),
    }
