"""Check current working-tree files for common release hazards.

This is not a license audit, a Git-history scan, or a guarantee that GitHub has
removed previously published objects. Review staged content separately.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RULES = ROOT / ".release-check.local"
MAX_GIT_BYTES = 100 * 1024 * 1024
FORBIDDEN_PARTS = {
    ".venv", ".venvs", "__pycache__", "logs_rl", "logs_eval", "checkpoints",
    "artifacts", "external_models", "a3-walk-deploy", "sonic_release",
    "sample_data", "private_data", ".release-check.local",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(rb"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "personal home path": re.compile(
        rb"""(?<![A-Za-z0-9_])/(?:home|Users)/[^/\\\s"'<>\x00]+(?=/|\s|$)"""
    ),
    "Windows user path": re.compile(
        rb"""(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s"'<>\x00]+(?=[\\/]|\s|$)"""
    ),
}

def git_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    )
    return sorted({ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item})

def release_patterns() -> dict[str, re.Pattern[bytes]]:
    """Add machine-local literal rules without publishing the literals."""
    patterns = dict(SECRET_PATTERNS)
    if not LOCAL_RULES.is_file():
        return patterns
    for number, raw in enumerate(LOCAL_RULES.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, separator, literal = line.partition("=")
        if not separator or not label.strip() or not literal:
            raise SystemExit(f"invalid local release rule at line {number}; expected label=literal")
        # Neither a private literal nor a user-supplied label is printed.
        patterns[f"local rule #{number}"] = re.compile(re.escape(literal.encode("utf-8")))
    return patterns

def inspect_file(path: Path, patterns: dict[str, re.Pattern[bytes]]) -> list[str]:
    relative = path.relative_to(ROOT)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return [f"forbidden path: {relative}"]
    if path.is_symlink():
        return [f"symlink requires manual release review: {relative}"]
    if not path.is_file():
        return []
    size = path.stat().st_size
    if size >= MAX_GIT_BYTES:
        return [f"file reaches GitHub 100 MiB limit: {relative} ({size} bytes)"]
    # Scan every accepted size, including the previously skipped 5-100 MiB range.
    data = path.read_bytes()
    return [f"{label} pattern: {relative}" for label, pattern in patterns.items() if pattern.search(data)]

def main() -> None:
    files = git_files()
    patterns = release_patterns()
    errors = [error for path in files for error in inspect_file(path, patterns)]
    if errors:
        raise SystemExit("RELEASE_CHECK_FAILED\n" + "\n".join(sorted(set(errors))))
    print(f"RELEASE_CHECK_OK files={len(files)} scope=working-tree")
    print("Staged content, Git history, remote cached objects, and permissions require separate review.")

if __name__ == "__main__":
    main()
