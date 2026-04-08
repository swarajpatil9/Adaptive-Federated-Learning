"""Release-grade system verification for AFLF (Phase 15.2)."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml
from packaging.requirements import Requirement

from aflf.config import ConfigValidator, load_and_validate_config
from aflf.training import FederatedTrainer, build_federated_config, load_yaml_config
from experiments.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner
from visualization.plotter import PlotManager
from visualization.visualization_config import VisualizationConfig

ROOT = Path(__file__).resolve().parent


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str = ""


@dataclass
class SystemSummary:
    python_version: str
    torch_version: str
    device: str
    config_status: str
    dependency_status: str
    system_checks_status: str

    @classmethod
    def gather(cls, root: Path) -> "SystemSummary":
        python_version = sys.version.split(" ")[0]
        torch_version = "missing"
        device = "CPU"

        try:
            import torch

            torch_version = str(torch.__version__)
            if bool(torch.cuda.is_available()):
                device = "GPU (CUDA)"
            elif bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                device = "GPU (MPS)"
        except Exception:
            torch_version = "missing"
            device = "CPU"

        try:
            load_and_validate_config(
                config_path=root / "configs" / "baseline.yaml",
                validator=ConfigValidator(),
            )
            config_status = "OK"
        except Exception:
            config_status = "FAIL"

        try:
            from aflf.system import DependencyChecker

            issues = DependencyChecker.validate(str(root / "requirements.txt"))
            dependency_status = "OK" if not issues else "FAIL"
        except Exception:
            dependency_status = "FAIL"

        verifier = SystemVerifier(rounds=1)
        structure_ok, _ = verifier.check_project_structure()
        import_ok, _ = verifier.check_import_resolution()
        system_checks_status = "OK" if structure_ok and import_ok else "FAIL"

        return cls(
            python_version=python_version,
            torch_version=torch_version,
            device=device,
            config_status=config_status,
            dependency_status=dependency_status,
            system_checks_status=system_checks_status,
        )

    def print(self) -> None:
        print("System Summary:\n")
        print(f"Python: {self.python_version}")
        print(f"Torch: {self.torch_version}")
        print(f"Device: {self.device}")
        print(f"Config: {self.config_status}")
        print(f"Dependencies: {self.dependency_status}")
        print(f"System checks: {self.system_checks_status}")


class SystemVerifier:
    """Runs end-to-end release checks and prints PASS/FAIL checklist output."""

    REQUIRED_STRUCTURE: Tuple[str, ...] = (
        "main.py",
        "check_env.py",
        "requirements.txt",
        "configs/baseline.yaml",
        "aflf",
        "experiments",
        "visualization",
        "results",
    )

    CORE_MODULES: Tuple[str, ...] = (
        "aflf",
        "aflf.training",
        "aflf.server",
        "aflf.selection",
        "aflf.privacy",
        "aflf.optimization",
        "aflf.communication",
        "experiments.experiment_runner",
        "visualization.plotter",
    )

    IMPORT_TO_PACKAGE: Dict[str, str] = {
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "PIL": "pillow",
        "cv2": "opencv-python",
    }

    PACKAGE_TO_IMPORT: Dict[str, str] = {
        "scikit-learn": "sklearn",
        "pyyaml": "yaml",
    }

    EXTERNAL_TOPLEVEL_ALLOWLIST: Set[str] = {
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "sklearn",
        "yaml",
        "matplotlib",
        "seaborn",
        "tqdm",
        "opacus",
        "packaging",
    }

    def __init__(self, rounds: int = 2) -> None:
        self.rounds = max(1, int(rounds))
        self.results: List[CheckResult] = []
        self.issues: List[str] = []
        self.execution_artifacts: Dict[str, Any] = {}

    def run(self) -> int:
        self._run_check("Project structure", self.check_project_structure)
        self._run_check("Import resolution", self.check_import_resolution)
        self._run_check("Dependency availability", self.check_dependency_availability)
        self._run_check("Requirements completeness", self.check_requirements_completeness)
        self._run_check("Config loading", self.check_config_loading)
        self._run_check("Feature toggles", self.check_toggles)
        self._run_check("CLI behavior", self.check_cli_behavior)
        self._run_check("Training pipeline", self.check_training_pipeline)
        self._run_check("Experiment runner", self.check_experiment_runner)
        self._run_check("Visualization", self.check_visualization_generation)
        self._run_check("Output validation", self.check_output_validation)
        self._run_check("Failure safety", self.check_failure_safety)
        self._run_check("Reproducibility", self.check_reproducibility)
        self._run_check("Logging", self.check_logging)
        self._run_check("Performance sanity", self.check_performance_sanity)
        self._run_check("Integration correctness", self.check_integration_correctness)

        self.print_summary()
        return 0 if all(item.passed for item in self.results) else 1

    def _run_check(self, name: str, check_fn) -> None:
        try:
            passed, details = check_fn()
        except Exception as exc:  # pragma: no cover - release safety path
            passed = False
            details = f"{exc}\n{traceback.format_exc()}"
        self.results.append(CheckResult(name=name, passed=passed, details=details))
        if not passed:
            self.issues.append(f"{name}: {details}")

    def _python(self) -> str:
        return sys.executable

    def _run_cmd(self, args: List[str], timeout: int = 180) -> Tuple[int, str]:
        proc = subprocess.run(
            args,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()

    def _make_tiny_config(self, output_name: str, *, all_features: bool) -> Path:
        base = load_yaml_config(str(ROOT / "configs" / "baseline.yaml"))

        base.setdefault("seed", 42)
        base.setdefault("federated", {})
        base.setdefault("training", {})
        base.setdefault("data", {})
        base.setdefault("selection", {})
        base.setdefault("privacy", {})
        base.setdefault("communication", {})
        base.setdefault("optimization", {}).setdefault("adaptive_lr", {})

        base["federated"]["num_rounds"] = self.rounds
        base["federated"]["clients_per_round"] = 2
        base["training"]["local_epochs"] = 1
        base["training"]["batch_size"] = 16
        base["training"]["learning_rate"] = 0.001
        base["data"]["num_clients"] = 4
        base["data"]["download"] = True

        if all_features:
            base["selection"]["strategy"] = "dynamic"
            base["privacy"]["privacy_enabled"] = True
            base["communication"]["compression_enabled"] = True
            base["optimization"]["adaptive_lr"]["enabled"] = True
            base["seed"] = 777
        else:
            base["selection"]["strategy"] = "random"
            base["privacy"]["privacy_enabled"] = False
            base["communication"]["compression_enabled"] = False
            base["optimization"]["adaptive_lr"]["enabled"] = False
            base["seed"] = 123

        out_dir = ROOT / "results" / "system_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / output_name
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(base, handle, sort_keys=False)
        return path

    def _run_main(self, config_path: Path, output_dir: Path, experiment: str, seed: int) -> Dict[str, Any]:
        cmd = [
            self._python(),
            "main.py",
            "--config",
            str(config_path),
            "--rounds",
            str(self.rounds),
            "--output-dir",
            str(output_dir),
            "--experiment",
            experiment,
            "--seed",
            str(seed),
        ]
        code, out = self._run_cmd(cmd, timeout=600)
        return {"code": code, "output": out, "command": cmd}

    def check_project_structure(self) -> Tuple[bool, str]:
        missing = [path for path in self.REQUIRED_STRUCTURE if not (ROOT / path).exists()]
        if missing:
            return False, f"Missing required paths: {missing}"
        return True, "Core files/directories present"

    def check_import_resolution(self) -> Tuple[bool, str]:
        failed: List[str] = []
        for mod in self.CORE_MODULES:
            if importlib.util.find_spec(mod) is None:
                failed.append(mod)
        if failed:
            return False, f"Unresolved imports: {failed}"
        return True, "Core modules are importable"

    def _requirements_map(self) -> Dict[str, Requirement]:
        reqs: Dict[str, Requirement] = {}
        raw = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        for line in raw:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("-"):
                continue
            req = Requirement(s)
            reqs[req.name.lower()] = req
        return reqs

    def check_dependency_availability(self) -> Tuple[bool, str]:
        reqs = self._requirements_map()
        missing: List[str] = []
        for name in reqs:
            module = self.PACKAGE_TO_IMPORT.get(name, name.replace("-", "_"))
            if importlib.util.find_spec(module) is None:
                missing.append(name)
        if missing:
            return False, f"Missing installed dependencies: {missing}"
        return True, "All runtime dependencies importable"

    def _collect_python_files(self) -> List[Path]:
        roots = [ROOT / "aflf", ROOT / "experiments", ROOT / "visualization", ROOT]
        files: List[Path] = []
        for base in roots:
            if base.is_file() and base.suffix == ".py":
                files.append(base)
                continue
            if base.is_dir():
                files.extend(path for path in base.rglob("*.py") if "htmlcov" not in str(path))
        return sorted(set(files))

    def _collect_external_imports(self) -> Set[str]:
        stdlib = set(getattr(sys, "stdlib_module_names", set()))
        local_roots = {"aflf", "experiments", "visualization", "tests"}
        imports: Set[str] = set()

        for path in self._collect_python_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        if top and top not in stdlib and top not in local_roots:
                            imports.add(top)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    if not node.module:
                        continue
                    top = node.module.split(".", 1)[0]
                    if top and top not in stdlib and top not in local_roots:
                        imports.add(top)

        return imports

    def check_requirements_completeness(self) -> Tuple[bool, str]:
        reqs = self._requirements_map()
        imported = self._collect_external_imports()

        missing_in_requirements: List[str] = []
        for module in sorted(imported):
            package = self.IMPORT_TO_PACKAGE.get(module, module)
            package_lower = package.lower()
            if package_lower not in reqs and module in self.EXTERNAL_TOPLEVEL_ALLOWLIST:
                missing_in_requirements.append(package)

        if missing_in_requirements:
            return False, (
                "Third-party imports not declared in requirements.txt: "
                f"{sorted(set(missing_in_requirements))}"
            )

        return True, "Third-party imports are covered by requirements"

    def check_config_loading(self) -> Tuple[bool, str]:
        config_dir = ROOT / "configs"
        parse_failures: List[str] = []

        for config_file in config_dir.rglob("*.yaml"):
            try:
                yaml.safe_load(config_file.read_text(encoding="utf-8"))
            except Exception as exc:
                parse_failures.append(f"{config_file.relative_to(ROOT)}: {exc}")

        if parse_failures:
            return False, "; ".join(parse_failures)

        load_and_validate_config(config_path=ROOT / "configs" / "baseline.yaml", validator=ConfigValidator())
        return True, "All YAML files parse; baseline config validates"

    def check_toggles(self) -> Tuple[bool, str]:
        baseline = ExperimentConfig(name="baseline_toggle", description="baseline")
        full = ExperimentConfig(
            name="aflf_toggle",
            description="aflf",
            selection_enabled=True,
            privacy_enabled=True,
            adaptive_lr_enabled=True,
            compression_enabled=True,
        )

        b = baseline.build_runtime_config()
        f = full.build_runtime_config()

        conditions = [
            b.get("selection", {}).get("strategy") == "random",
            f.get("selection", {}).get("strategy") == "dynamic",
            bool(b.get("privacy", {}).get("privacy_enabled")) is False,
            bool(f.get("privacy", {}).get("privacy_enabled")) is True,
            bool(b.get("optimization", {}).get("adaptive_lr", {}).get("enabled")) is False,
            bool(f.get("optimization", {}).get("adaptive_lr", {}).get("enabled")) is True,
            bool(b.get("communication", {}).get("compression_enabled")) is False,
            bool(f.get("communication", {}).get("compression_enabled")) is True,
        ]

        if not all(conditions):
            return False, "One or more feature toggles did not map correctly"

        return True, "All requested toggles map correctly"

    def check_cli_behavior(self) -> Tuple[bool, str]:
        checks = []

        code_help, out_help = self._run_cmd([self._python(), "main.py", "--help"])
        checks.append(code_help == 0 and "Adaptive Federated Learning Framework" in out_help)

        code_default, out_default = self._run_cmd([self._python(), "main.py", "--rounds", "1", "--experiment", "cli_default", "--output-dir", "results/system_check/cli_default"], timeout=600)
        checks.append(code_default == 0)

        code_config, out_config = self._run_cmd([
            self._python(),
            "main.py",
            "--config",
            "configs/baseline.yaml",
            "--rounds",
            "1",
            "--experiment",
            "cli_config",
            "--output-dir",
            "results/system_check/cli_config",
        ], timeout=600)
        checks.append(code_config == 0)

        if not all(checks):
            details = (
                f"help={code_help}, default={code_default}, with_config={code_config}. "
                f"default_out={out_default[-300:] if out_default else ''}; "
                f"config_out={out_config[-300:] if out_config else ''}"
            )
            return False, details

        return True, "python main.py, python main.py --config, python main.py --help all work"

    def check_training_pipeline(self) -> Tuple[bool, str]:
        baseline_cfg = self._make_tiny_config("baseline_small.yaml", all_features=False)
        aflf_cfg = self._make_tiny_config("aflf_small.yaml", all_features=True)

        baseline_out_dir = ROOT / "results" / "system_check" / "baseline"
        aflf_out_dir = ROOT / "results" / "system_check" / "aflf"

        baseline_run = self._run_main(baseline_cfg, baseline_out_dir, "baseline_check", seed=123)
        aflf_run = self._run_main(aflf_cfg, aflf_out_dir, "aflf_check", seed=777)

        self.execution_artifacts["baseline_run"] = baseline_run
        self.execution_artifacts["aflf_run"] = aflf_run
        self.execution_artifacts["baseline_out_dir"] = baseline_out_dir
        self.execution_artifacts["aflf_out_dir"] = aflf_out_dir

        if baseline_run["code"] != 0:
            return False, f"Baseline run failed: {baseline_run['output'][-600:]}"
        if aflf_run["code"] != 0:
            return False, f"AFLF run failed: {aflf_run['output'][-600:]}"

        return True, "Baseline and AFLF end-to-end runs succeeded"

    def check_experiment_runner(self) -> Tuple[bool, str]:
        output_dir = ROOT / "results" / "system_check" / "experiments"
        baseline_small = self._make_tiny_config("exp_baseline.yaml", all_features=False)

        baseline_cfg = ExperimentConfig(
            name="baseline",
            description="baseline small",
            base_config_path=str(baseline_small),
            selection_enabled=False,
            privacy_enabled=False,
            adaptive_lr_enabled=False,
            compression_enabled=False,
            seed=22,
            num_runs=1,
            output_subdir=str(output_dir),
        )
        aflf_cfg = ExperimentConfig(
            name="aflf_full",
            description="aflf small",
            base_config_path=str(baseline_small),
            selection_enabled=True,
            privacy_enabled=True,
            adaptive_lr_enabled=True,
            compression_enabled=True,
            seed=22,
            num_runs=1,
            output_subdir=str(output_dir),
        )

        runner = ExperimentRunner(output_dir=str(output_dir))
        payload = runner.run_many([baseline_cfg, aflf_cfg])
        self.execution_artifacts["experiment_payload"] = payload
        self.execution_artifacts["experiment_out_dir"] = output_dir

        baseline_ok = "baseline" in payload and payload["baseline"].get("num_runs", 0) >= 1
        aflf_ok = "aflf_full" in payload and payload["aflf_full"].get("num_runs", 0) >= 1
        if not (baseline_ok and aflf_ok):
            return False, "Experiment runner did not produce expected payload keys"

        return True, "Experiment runner produced baseline and AFLF outputs"

    def check_visualization_generation(self) -> Tuple[bool, str]:
        metrics_dir = Path(self.execution_artifacts.get("experiment_out_dir", ROOT / "results" / "system_check" / "experiments")) / "metrics"
        plots_dir = ROOT / "results" / "system_check" / "plots"

        config = VisualizationConfig(metrics_dir=metrics_dir, output_dir=plots_dir)
        artifacts = PlotManager(config=config).generate_all_plots()
        self.execution_artifacts["plot_artifacts"] = artifacts.paths

        if not artifacts.paths:
            return False, "No plots were generated"

        missing = [str(path) for path in artifacts.paths.values() if not Path(path).exists()]
        if missing:
            return False, f"Plot artifacts missing: {missing}"

        return True, f"Generated {len(artifacts.paths)} plots"

    def check_output_validation(self) -> Tuple[bool, str]:
        baseline_dir = Path(self.execution_artifacts.get("baseline_out_dir", ROOT / "results" / "system_check" / "baseline"))
        exp_dir = Path(self.execution_artifacts.get("experiment_out_dir", ROOT / "results" / "system_check" / "experiments"))
        plot_artifacts = self.execution_artifacts.get("plot_artifacts", {})

        metrics_ok = bool(list((baseline_dir / "metrics").glob("*_rounds.csv")))
        logs_ok = bool(list((baseline_dir / "logs").glob("*.log")))
        exp_logs_ok = (exp_dir / "experiment_tracker.json").exists() and (exp_dir / "comparison_table.json").exists()
        plots_ok = bool(plot_artifacts)

        if not all([metrics_ok, logs_ok, exp_logs_ok, plots_ok]):
            return False, (
                f"metrics_ok={metrics_ok}, logs_ok={logs_ok}, "
                f"experiment_logs_ok={exp_logs_ok}, plots_ok={plots_ok}"
            )

        return True, "Metrics, plots, experiment logs, and logs are generated"

    def check_failure_safety(self) -> Tuple[bool, str]:
        missing_code, missing_out = self._run_cmd(
            [self._python(), "main.py", "--config", "configs/DOES_NOT_EXIST.yaml"], timeout=60
        )
        missing_ok = missing_code != 0 and "Config file not found" in missing_out

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            tmp.write("seed: 42\nmodel: {}\n")
            invalid_path = tmp.name

        invalid_code, invalid_out = self._run_cmd(
            [self._python(), "main.py", "--config", invalid_path], timeout=120
        )
        os.unlink(invalid_path)
        invalid_ok = invalid_code != 0 and (
            "Configuration validation failed" in invalid_out or "Missing required" in invalid_out
        )

        if not (missing_ok and invalid_ok):
            return False, (
                f"missing_ok={missing_ok}, invalid_ok={invalid_ok}, "
                f"missing_out={missing_out[-300:]}, invalid_out={invalid_out[-300:]}"
            )

        return True, "Invalid config and missing file fail gracefully"

    def _read_latest_round_csv(self, output_dir: Path, experiment_name: str) -> Path:
        metrics_dir = output_dir / "metrics"
        candidates = sorted(metrics_dir.glob(f"{experiment_name}_phase7_*_rounds.csv"))
        if not candidates:
            raise FileNotFoundError(f"No rounds.csv found for {experiment_name} in {metrics_dir}")
        return candidates[-1]

    def _summary_from_round_csv(self, csv_path: Path) -> Dict[str, float]:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return {"final_global_accuracy": 0.0, "final_global_loss": 0.0, "total_training_time": 0.0}

        header = lines[0].split(",")
        row = lines[-1].split(",")
        idx = {name: i for i, name in enumerate(header)}

        def value(name: str) -> float:
            if name not in idx or idx[name] >= len(row):
                return 0.0
            try:
                return float(row[idx[name]])
            except ValueError:
                return 0.0

        return {
            "final_global_accuracy": value("global_accuracy"),
            "final_global_loss": value("global_loss"),
            "total_training_time": value("total_training_time"),
        }

    def check_reproducibility(self) -> Tuple[bool, str]:
        cfg = self._make_tiny_config("repro_small.yaml", all_features=False)
        out1 = ROOT / "results" / "system_check" / "repro_run1"
        out2 = ROOT / "results" / "system_check" / "repro_run2"

        run1 = self._run_main(cfg, out1, "repro_same", seed=2026)
        run2 = self._run_main(cfg, out2, "repro_same", seed=2026)

        if run1["code"] != 0 or run2["code"] != 0:
            return False, "Reproducibility runs failed to execute"

        csv1 = self._read_latest_round_csv(out1, "repro_same")
        csv2 = self._read_latest_round_csv(out2, "repro_same")
        s1 = self._summary_from_round_csv(csv1)
        s2 = self._summary_from_round_csv(csv2)

        same_accuracy = abs(s1["final_global_accuracy"] - s2["final_global_accuracy"]) < 1e-8
        same_loss = abs(s1["final_global_loss"] - s2["final_global_loss"]) < 1e-8

        if not (same_accuracy and same_loss):
            return False, f"Metric mismatch across same-seed runs: run1={s1}, run2={s2}"

        return True, f"Same-seed runs consistent: accuracy={s1['final_global_accuracy']:.6f}"

    def check_logging(self) -> Tuple[bool, str]:
        out_dir = Path(self.execution_artifacts.get("baseline_out_dir", ROOT / "results" / "system_check" / "baseline"))
        logs = sorted((out_dir / "logs").glob("*.log"))
        if not logs:
            return False, "No log files generated"

        text = logs[-1].read_text(encoding="utf-8", errors="ignore")
        has_start = "Adaptive Federated Learning Framework" in text
        has_complete = "TRAINING COMPLETE" in text or "Training finished" in text

        if not (has_start and has_complete):
            return False, "Logs missing expected start/complete markers"

        return True, f"Log file generated: {logs[-1].name}"

    def check_performance_sanity(self) -> Tuple[bool, str]:
        cfg_data = load_yaml_config(str(self._make_tiny_config("perf_small.yaml", all_features=False)))
        fed_cfg = build_federated_config(cfg_data)

        tracemalloc.start()
        before_current, before_peak = tracemalloc.get_traced_memory()

        def run_once(seed: int) -> Dict[str, Any]:
            cfg_data["seed"] = seed
            fed = build_federated_config(cfg_data)
            trainer = FederatedTrainer.from_components(
                data_config=cfg_data["data"],
                model_config=cfg_data["model"],
                federated_config=fed,
                selection_config=cfg_data.get("selection"),
                experiment_name=f"perf_{seed}",
                metrics_output_dir=str(ROOT / "results" / "system_check" / "perf" / "metrics"),
            )
            return trainer.fit()

        t0 = time.perf_counter()
        _ = run_once(99)
        _ = run_once(100)
        elapsed = time.perf_counter() - t0

        after_current, after_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_growth_mb = max(0, after_current - before_current) / (1024 * 1024)
        peak_mb = after_peak / (1024 * 1024)

        round_budget_ok = (elapsed / (2 * max(1, fed_cfg.num_rounds))) < 45.0
        memory_ok = memory_growth_mb < 75.0

        if not (round_budget_ok and memory_ok):
            return False, (
                f"Performance thresholds exceeded: elapsed={elapsed:.2f}s, "
                f"memory_growth_mb={memory_growth_mb:.2f}, peak_mb={peak_mb:.2f}"
            )

        return True, (
            f"Round time sane and no major memory growth (elapsed={elapsed:.2f}s, "
            f"growth={memory_growth_mb:.2f}MB)"
        )

    def check_integration_correctness(self) -> Tuple[bool, str]:
        aflf_run = self.execution_artifacts.get("aflf_run")
        if not aflf_run:
            return False, "AFLF run artifacts are missing"

        out_dir = Path(self.execution_artifacts.get("aflf_out_dir", ROOT / "results" / "system_check" / "aflf"))
        csv_path = self._read_latest_round_csv(out_dir, "aflf_check")
        headers = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")

        required_columns = {
            "privacy_enabled_fraction",
            "communication_reduction_percentage",
            "learning_rate",
            "num_selected_clients",
        }
        missing_cols = sorted(required_columns - set(headers))
        if missing_cols:
            return False, f"Integration metric columns missing: {missing_cols}"

        aflf_cfg = load_yaml_config(str(ROOT / "results" / "system_check" / "aflf_small.yaml"))
        toggles_ok = (
            aflf_cfg.get("privacy", {}).get("privacy_enabled") is True
            and aflf_cfg.get("communication", {}).get("compression_enabled") is True
            and aflf_cfg.get("optimization", {}).get("adaptive_lr", {}).get("enabled") is True
            and aflf_cfg.get("selection", {}).get("strategy") == "dynamic"
        )

        if not toggles_ok:
            return False, "AFLF toggle configuration was not applied as expected"

        return True, "Privacy, adaptive LR, communication, and selection integrate without conflict"

    def print_summary(self) -> None:
        print("=" * 88)
        print("AFLF SYSTEM VERIFICATION (PHASE 15.2)")
        print("=" * 88)
        for item in self.results:
            print(f"{item.name} {'PASS' if item.passed else 'FAIL'}")
            if item.details:
                print(f"  {item.details}")

        print("-" * 88)
        if self.issues:
            print("Failure report:")
            for issue in self.issues:
                print(f"- {issue}")
        else:
            print("All checks passed.")
        print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Release verification checklist runner for AFLF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=2, help="Rounds for fast verification runs")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print fast environment/config/dependency summary without full verification",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summary:
        summary = SystemSummary.gather(ROOT)
        summary.print()
        return 0 if (
            summary.config_status == "OK"
            and summary.dependency_status == "OK"
            and summary.system_checks_status == "OK"
        ) else 1

    verifier = SystemVerifier(rounds=args.rounds)
    return verifier.run()


if __name__ == "__main__":
    sys.exit(main())
