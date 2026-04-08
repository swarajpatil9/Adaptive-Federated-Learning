"""Environment and dependency verification script for AFLF."""

from __future__ import annotations

import sys

from aflf.system.environment import DependencyChecker, EnvironmentChecker


def main() -> int:
    report = EnvironmentChecker.check()

    print("=" * 72)
    print("AFLF Environment Check")
    print("=" * 72)
    print(f"Python version : {report.python_version}")
    print(f"Platform       : {report.platform}")
    print(f"PyTorch        : {'available' if report.has_torch else 'missing'}")
    print(f"CUDA           : {'yes' if report.has_cuda else 'no'}")
    print(f"MPS            : {'yes' if report.has_mps else 'no'}")

    issues = DependencyChecker.validate("requirements.txt")

    if issues:
        print("\nDependency issues:")
        for issue in issues:
            print(f"- {issue}")
        print("\nResult: FAILED")
        return 1

    print("\nResult: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
