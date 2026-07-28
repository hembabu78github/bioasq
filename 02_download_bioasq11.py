from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "bioasq11"
OUTPUT_DIR = ROOT / "outputs" / "stage1"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORD_ID = "7655130"
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
TARGET_NAME = "training11b.json"
EXPECTED_MD5 = "fc1fe03831b69157c82a746337c00712"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "JMS-Auditable-RAG/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "JMS-Auditable-RAG/1.0"})
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        copied = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            copied += len(block)
            if total_int:
                print(
                    f"\rDownloaded {copied / 1024**2:.1f} MiB "
                    f"({copied * 100 / total_int:.1f}%)",
                    end="",
                )
            else:
                print(f"\rDownloaded {copied / 1024**2:.1f} MiB", end="")
    print()
    partial.replace(destination)


def main() -> int:
    target = RAW_DIR / TARGET_NAME
    manifest_path = OUTPUT_DIR / "download_manifest.json"

    try:
        metadata = fetch_json(RECORD_API)
    except Exception as exc:
        print(f"ERROR: Could not read Zenodo metadata: {exc}")
        return 1

    file_record = next(
        (item for item in metadata.get("files", []) if item.get("key") == TARGET_NAME),
        None,
    )
    if not file_record:
        print(f"ERROR: {TARGET_NAME} was not found in record {RECORD_ID}.")
        return 1

    links = file_record.get("links", {})
    download_url = links.get("content") or links.get("self")
    if not download_url:
        print("ERROR: Zenodo did not provide a download URL.")
        return 1

    if target.exists() and md5_file(target) != EXPECTED_MD5:
        quarantine = target.with_name(f"{target.stem}.checksum_mismatch{target.suffix}")
        target.replace(quarantine)
        print(f"Moved invalid existing file to: {quarantine}")

    if not target.exists():
        download(download_url, target)
    else:
        print(f"Using existing valid file: {target}")

    observed_md5 = md5_file(target)
    checksum_ok = observed_md5 == EXPECTED_MD5

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "Zenodo",
            "record_id": RECORD_ID,
            "record_api": RECORD_API,
            "record_doi": metadata.get("doi"),
            "dataset_title": metadata.get("metadata", {}).get("title"),
            "dataset_version": metadata.get("metadata", {}).get("version"),
            "file_name": TARGET_NAME,
        },
        "local": {
            "relative_path": str(target.relative_to(ROOT)),
            "size_bytes": target.stat().st_size,
            "expected_md5": EXPECTED_MD5,
            "observed_md5": observed_md5,
            "checksum_ok": checksum_ok,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not checksum_ok:
        print(f"ERROR: Expected {EXPECTED_MD5}, observed {observed_md5}")
        return 2

    print(f"Checksum verified: {observed_md5}")
    print(f"Manifest saved: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
