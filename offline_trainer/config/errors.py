"""Config error types and formatting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfigValidationIssue:
    error_path: str
    error_message: str
    hint: str | None = None


class ConfigError(ValueError):
    """Raised when config validation fails with actionable errors."""

    def __init__(self, issues: Iterable[ConfigValidationIssue]) -> None:
        self.issues = list(issues)
        message = _format_issues(self.issues)
        super().__init__(message)


def _format_issues(issues: list[ConfigValidationIssue]) -> str:
    lines = [f"Config validation failed with {len(issues)} error(s):"]
    for issue in issues:
        lines.append(f"- error_path: {issue.error_path}")
        lines.append(f"  error_message: {issue.error_message}")
        if issue.hint:
            lines.append(f"  hint: {issue.hint}")
    return "\n".join(lines)
