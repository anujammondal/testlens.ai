#!/usr/bin/env python3
"""
Run the full TestLens end-to-end pipeline.

Flow:
1) Fetch test cases from QMetry
2) Create semantic embeddings
3) Generate duplicate report HTML
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_python(user_python: str | None = None) -> str:
    """Resolve which Python interpreter should run the pipeline."""
    if user_python:
        return user_python

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    return sys.executable


def validate_dependencies(python_executable: str) -> bool:
    """Validate that required dependencies are available."""
    required_modules = [
        "requests",
        "dotenv",
        "numpy",
        "sentence_transformers",
    ]
    check_cmd = [
        python_executable,
        "-c",
        "import importlib.util, sys; mods=sys.argv[1:]; "
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
        "print(','.join(missing)); sys.exit(0 if not missing else 1)",
        *required_modules,
    ]
    result = subprocess.run(
        check_cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True

    missing = result.stdout.strip() or "project dependencies"
    print("\n❌ Missing Python dependencies:", missing)
    print("Install them with:")
    print(f"   {python_executable} -m pip install -r requirements.txt")
    print("")
    print("Tip: create and use a virtual environment first:")
    print("   python3 -m venv .venv")
    print("   source .venv/bin/activate")
    print("   pip install -r requirements.txt")
    return False


def run_step(name: str, command: List[str]) -> None:
    """Run a single pipeline step and stop on failure."""
    print(f"\n{'=' * 72}")
    print(f"▶ {name}")
    print(f"{'=' * 72}")
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full TestLens flow: fetch -> embeddings -> duplicate report"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Embedding similarity threshold for duplicate report (default: 0.92)",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated HTML report automatically (macOS)",
    )
    parser.add_argument(
        "--python",
        help="Python interpreter to use (defaults to .venv/bin/python if available)",
    )
    args = parser.parse_args()

    python_executable = resolve_python(args.python)
    if not validate_dependencies(python_executable):
        return 1

    fetch_cmd = [
        python_executable,
        "src/qMetryIntegration/fetchTestCasesWithQParam.py",
    ]
    embed_cmd = [
        python_executable,
        "src/qMetryIntegration/createSemanticEmbeddings.py",
    ]
    report_cmd = [
        python_executable,
        "src/qMetryIntegration/generateDuplicateReport.py",
        "--output",
        "reports/duplicate_testcases_report.html",
        "--threshold",
        str(args.threshold),
    ]

    try:
        run_step("Fetching test cases from QMetry", fetch_cmd)
        run_step("Creating semantic embeddings", embed_cmd)
        run_step("Generating duplicate report", report_cmd)
    except subprocess.CalledProcessError as exc:
        print(f"\n❌ Pipeline failed at step exit code {exc.returncode}")
        return exc.returncode

    report_path = PROJECT_ROOT / "reports" / "duplicate_testcases_report.html"
    print(f"\n✅ End-to-end flow completed.")
    print(f"📄 Report: {report_path}")

    if args.open_report and report_path.exists():
        try:
            subprocess.run(["open", str(report_path)], check=False)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
