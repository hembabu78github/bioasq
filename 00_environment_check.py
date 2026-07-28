from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "stage0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bytes_to_gib(value: int) -> float:
    return round(value / (1024 ** 3), 2)


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": args,
            "error": f"{type(exc).__name__}: {exc}",
        }


def internet_check(host: str = "api.groq.com", port: int = 443) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=10):
            return {"host": host, "port": port, "reachable": True}
    except Exception as exc:
        return {
            "host": host,
            "port": port,
            "reachable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    drive = Path(ROOT.anchor or "C:\\")
    disk = shutil.disk_usage(drive)
    memory = psutil.virtual_memory()

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "is_64_bit": sys.maxsize > 2**32,
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "memory": {
            "total_gib": bytes_to_gib(memory.total),
            "available_gib": bytes_to_gib(memory.available),
            "percent_used": memory.percent,
        },
        "disk": {
            "checked_drive": str(drive),
            "total_gib": bytes_to_gib(disk.total),
            "free_gib": bytes_to_gib(disk.free),
            "used_gib": bytes_to_gib(disk.used),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "network": internet_check(),
        "commands": {
            "python_version": run_command([sys.executable, "--version"]),
            "pip_version": run_command([sys.executable, "-m", "pip", "--version"]),
        },
        "stage0_constraints": {
            "minimum_python": "3.11",
            "expected_ram_gib": 16,
            "expected_free_disk_gib": 50,
            "gpu_required": False,
            "colab_allowed": False,
        },
    }

    report_path = OUTPUT_DIR / "environment_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    freeze_result = run_command([sys.executable, "-m", "pip", "freeze"])
    freeze_path = OUTPUT_DIR / "pip_freeze.txt"
    freeze_path.write_text(
        freeze_result.get("stdout", "") + "\n",
        encoding="utf-8",
    )

    print(f"Environment report: {report_path}")
    print(f"Package freeze:     {freeze_path}")
    print()
    print(f"Python executable:  {sys.executable}")
    print(f"RAM total:          {report['memory']['total_gib']} GiB")
    print(f"Disk free:          {report['disk']['free_gib']} GiB")
    print(f"Groq host reachable:{report['network']['reachable']}")

    warnings: list[str] = []
    if sys.version_info[:2] != (3, 11):
        warnings.append("Use Python 3.11 for this project.")
    if memory.total < 14 * (1024 ** 3):
        warnings.append("Detected RAM is below the expected 16 GB class.")
    if disk.free < 35 * (1024 ** 3):
        warnings.append("Free disk is below the preferred safety margin of 35 GB.")
    if not report["network"]["reachable"]:
        warnings.append("api.groq.com was not reachable during the test.")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
        return 1

    print("\nStage 0 environment check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
