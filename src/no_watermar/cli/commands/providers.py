from __future__ import annotations

import argparse

from ...benchmark_providers import list_provider_descriptors, probe_provider_runtimes
from ...io_utils import print_json
from ...provider_doctor import build_provider_doctor_report


def configure_providers_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("providers", help="Inspect provider availability and runtime probes.")
    providers_subparsers = parser.add_subparsers(dest="providers_command", required=True)

    list_parser = providers_subparsers.add_parser("list", help="List implemented and planned providers.")
    list_parser.set_defaults(handler=_handle_list)

    probe_parser = providers_subparsers.add_parser("probe", help="Probe configured provider runtimes.")
    probe_parser.set_defaults(handler=_handle_probe)

    doctor_parser = providers_subparsers.add_parser("doctor", help="Diagnose provider interpreters, sidecar scripts, and runtime imports.")
    doctor_parser.set_defaults(handler=_handle_doctor)


def _handle_list(args: argparse.Namespace) -> int:
    del args
    print_json(list_provider_descriptors())
    return 0


def _handle_probe(args: argparse.Namespace) -> int:
    del args
    print_json(probe_provider_runtimes())
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    del args
    print_json(build_provider_doctor_report())
    return 0
