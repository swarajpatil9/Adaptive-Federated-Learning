"""Runtime environment and dependency checks."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List

from packaging.requirements import Requirement


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    platform: str
    has_torch: bool
    has_cuda: bool
    has_mps: bool


class EnvironmentChecker:
    """Check runtime capabilities before experiment execution."""

    @staticmethod
    def check() -> EnvironmentReport:
        has_torch = False
        has_cuda = False
        has_mps = False

        try:
            import torch

            has_torch = True
            has_cuda = bool(torch.cuda.is_available())
            has_mps = bool(torch.backends.mps.is_available())
        except Exception:
            has_torch = False

        return EnvironmentReport(
            python_version=sys.version.split(" ")[0],
            platform=platform.platform(),
            has_torch=has_torch,
            has_cuda=has_cuda,
            has_mps=has_mps,
        )


class DependencyChecker:
    """Verify that installed dependency versions match pinned requirements."""

    @staticmethod
    def validate(requirements_file: str = "requirements.txt") -> List[str]:
        mismatches: List[str] = []
        req_path = Path(requirements_file)
        if not req_path.exists():
            return [f"Requirements file not found: {requirements_file}"]

        lines = req_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("-"):
                continue

            try:
                requirement = Requirement(stripped)
            except Exception:
                mismatches.append(f"Invalid requirement format: {stripped}")
                continue

            package = requirement.name

            try:
                installed = version(package)
            except PackageNotFoundError:
                if str(requirement.specifier):
                    mismatches.append(
                        f"{package} is not installed (expected {requirement.specifier})"
                    )
                else:
                    mismatches.append(f"{package} is not installed")
                continue

            if requirement.specifier and not requirement.specifier.contains(installed):
                mismatches.append(
                    f"{package} version mismatch: installed {installed}, "
                    f"expected {requirement.specifier}"
                )

        return mismatches
