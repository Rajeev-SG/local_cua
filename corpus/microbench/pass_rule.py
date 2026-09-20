#!/usr/bin/env python3
"""Declarative pass-rule interpreter for harvested tasks (issue #27).

A harvested task carries its own `verification.pass_rule`, so the consumer needs
**no per-task code** and admission stays deterministic — no model ever decides
pass/fail. This module mirrors the producer's `tools/task_pass_rule.py`
(`Rajeev-SG/codex-session-orchestration-analysis`, `pareto-research-task-definition/v1`,
pinned in `bench-ext/corpus/SOURCE.json`); `tests/test_pass_rule_parity.py` pins the
rule kinds so the two copies cannot drift silently.

Rules
-----
finding_matches_truth
    fields          list-valued facts compared as exact sets (order-insensitive)
    exact_fields    scalar facts compared exactly (after stripping)
    presence_fields facts compared as booleans (present/absent)
    finding_fields  optional map from a truth key to a differently named finding key
    url_fields      subset of exact_fields compared with URL normalisation (a
                    trailing slash on a bare host is not a difference)
    require_any_of  fail closed when the page carries none of these facts: an
                    empty measurement is an invalid run, not a pass
structural
    expected_labels (ordered list), expected_arrows, require_no_overlap
test-runner
    judged by the runner (the command's exit status), never here -> returns None
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

RULE_KINDS = ("finding_matches_truth", "structural", "test-runner")

#: Set True to let a `finding_matches_truth` rule that declares no `require_any_of` accept an
#: all-empty measurement. Off by default — see the degenerate-measurement guard in `evaluate`.
ALLOW_EMPTY_TRUTH = False


def _norm(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _norm_url(value: Any) -> str:
    """Normalise a URL for comparison: a bare host and the same host with a trailing
    slash are the same page. Nothing else is loosened — scheme, host and path are
    still compared exactly, so a different path or host still fails."""
    text = _norm(value)
    if not text:
        return text
    parts = urlsplit(text)
    if not parts.scheme or not parts.netloc:
        return text
    path = "" if parts.path == "/" else parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _as_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {_norm(item) for item in value}
    return {_norm(value)}


def evaluate(rule: dict, truth: Any, finding: Any):
    """Return True/False, or None when the rule is not this module's to judge."""
    if not isinstance(rule, dict):
        return None
    if not isinstance(truth, dict) or not isinstance(finding, dict):
        return False
    kind = rule.get("kind")
    if kind == "test-runner":
        return None

    if kind == "finding_matches_truth":
        alias = rule.get("finding_fields") or {}
        for field in rule.get("fields") or []:
            key = alias.get(field, field)
            if _as_set(truth.get(field)) != _as_set(finding.get(key)):
                return False
        url_fields = set(rule.get("url_fields") or [])
        for field in rule.get("exact_fields") or []:
            key = alias.get(field, field)
            left, right = truth.get(field), finding.get(key)
            if field in url_fields:
                if _norm_url(left) != _norm_url(right):
                    return False
                continue
            if _norm(left) != _norm(right):
                return False
        for field in rule.get("presence_fields") or []:
            key = alias.get(field, field)
            if bool(truth.get(field)) != bool(finding.get(key)):
                return False
        # Fail closed on a degenerate measurement: if the page carries none of the things
        # being audited, the run is invalid, not a pass.
        #
        # When the rule declares `require_any_of`, that list is the audited set. When it does
        # not, the audited set is the rule's own fact fields. This is the producer's own
        # stated doctrine ("fail closed on a degenerate measurement") applied uniformly: a
        # bot-walled page that serves no tag at all must not score a pass just because the
        # agent also reported nothing. `ALLOW_EMPTY_TRUTH` is the escape hatch if a task ever
        # legitimately expects an empty page.
        any_of = rule.get("require_any_of")
        if any_of is None and not ALLOW_EMPTY_TRUTH:
            any_of = (list(rule.get("fields") or []) + list(rule.get("exact_fields") or [])
                      + list(rule.get("presence_fields") or []))
        if any_of and not any(bool(truth.get(field)) for field in any_of):
            return False
        return True

    if kind == "structural":
        labels = [_norm(x) for x in (finding.get("labels") or [])]
        expected = [_norm(x) for x in (rule.get("expected_labels") or [])]
        if expected and labels != expected:
            return False
        nodes = finding.get("nodes") or []
        if expected and len(nodes) != len(expected):
            return False
        if rule.get("expected_arrows") is not None:
            try:
                if int(finding.get("arrows") or 0) != int(rule["expected_arrows"]):
                    return False
            except (TypeError, ValueError):
                return False
        if rule.get("require_no_overlap") and bool(finding.get("overlap")):
            return False
        return True

    return None


def check_from_rule(rule: dict):
    """Build the `benchlib.Task.check` predicate for a declarative rule.

    `None` (the rule is not this module's / the agent did not produce a finding)
    counts as a fail: only an explicit True passes.
    """
    def check(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        return evaluate(rule, parsed.get("truth"), parsed.get("finding")) is True
    return check
