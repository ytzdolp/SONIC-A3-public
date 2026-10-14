"""Fail on common files and strings that should not enter a GitHub release."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RULES = ROOT / ".release-check.local"
MAX_GIT_BYTES = 100 * 1024 * 1024
FORBIDDEN_PARTS = {
    ".venv",
    ".venvs",
    "__pycache__",
    "logs_rl",
    "logs_eval",
    "checkpoints",
    "artifacts",
    "external_models",
    "a3-walk-deploy",
    "sonic_release",
    "sample_data",
    "private_data",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "personal home path": re.compile(
        rb"(?<![A-Za-z0-9_])/(?:home|Users)/[A-Za-z0-9._-]+(?:/|\b)"
    ),
}


def git_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def release_patterns() -> dict[str, re.Pattern[bytes]]:
    """Return committed checks plus optional, untracked machine-local literals.

    ``.release-check.local`` uses one ``label=literal`` rule per line. The file
    is ignored by Git so a private path or host does not have to be embedded in
    the checker that is meant to detect it.
    """

    patterns = dict(SECRET_PATTERNS)
    if not LOCAL_RULES.is_file():
        return patterns

    for line_number, raw_line in enumerate(
        LOCAL_RULES.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        label, separator, literal = line.partition("=")
        if not separator or not label.strip() or not literal:
            raise SystemExit(
                f"invalid local release rule at {LOCAL_RULES}:{line_number}; "
                "expected label=literal"
            )
        patterns[f"local rule: {label.strip()}"] = re.compile(
            re.escape(literal.encode("utf-8"))
        )
    return patterns


def main() -> None:
    errors: list[str] = []
    files = git_files()
    patterns = release_patterns()
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden path: {relative}")
            continue
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if size >= MAX_GIT_BYTES:
            errors.append(f"file reaches GitHub 100 MiB limit: {relative} ({size} bytes)")
            continue
        if size > 5 * 1024 * 1024:
            continue
        data = path.read_bytes()
        for label, pattern in patterns.items():
            if pattern.search(data):
                errors.append(f"{label} pattern: {relative}")
    if errors:
        raise SystemExit("RELEASE_CHECK_FAILED\n" + "\n".join(sorted(set(errors))))
    print(f"RELEASE_CHECK_OK files={len(files)}")


if __name__ == "__main__":
    main()
