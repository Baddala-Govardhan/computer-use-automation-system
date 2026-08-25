from __future__ import annotations

import fnmatch
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.actions import ActionType
from core.schema import Action


class RiskLevel(str, Enum):
    SAFE = "safe"
    REVIEW_REQUIRED = "review_required"
    IRREVERSIBLE = "irreversible"


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    risk_level: RiskLevel
    requires_human_confirmation: bool
    reason: str


class AllowedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    allowed_routes: list[str] = Field(default_factory=list)


class AllowlistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_targets: list[AllowedTarget]
    allowed_action_types: list[ActionType]


class IntentRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: RiskLevel
    requires_human_confirmation: bool = False
    blocked: bool = False


class RouteRuleExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: list[str] = Field(default_factory=list)
    locator_text: list[str] = Field(default_factory=list)


class RouteRiskRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_pattern: str
    action_types: list[ActionType]
    exclude: RouteRuleExclusion = Field(default_factory=RouteRuleExclusion)
    classification: IntentRisk


class RiskPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: IntentRisk
    intents: dict[str, IntentRisk] = Field(default_factory=dict)
    route_risk_rules: list[RouteRiskRule] = Field(default_factory=list)


class PolicyChecker:
    def __init__(self, allowlist: AllowlistConfig, risk_policy: RiskPolicyConfig):
        self._allowlist = allowlist
        self._risk_policy = risk_policy

    @classmethod
    def from_files(cls, allowlist_path: Path, risk_policy_path: Path) -> "PolicyChecker":
        allowlist = AllowlistConfig.model_validate(yaml.safe_load(allowlist_path.read_text()))
        risk_policy = RiskPolicyConfig.model_validate(yaml.safe_load(risk_policy_path.read_text()))
        return cls(allowlist, risk_policy)

    def check_action(self, action: Action, current_url: str) -> PolicyDecision:
        intent_risk = self._risk_policy.intents.get(action.intent, self._risk_policy.default)
        route_rule = self._match_route_rule(action, current_url)
        effective_risk = route_rule.classification if route_rule is not None else intent_risk

        if action.type not in self._allowlist.allowed_action_types:
            return PolicyDecision(
                allowed=False,
                risk_level=effective_risk.risk,
                requires_human_confirmation=effective_risk.requires_human_confirmation,
                reason=f"action type '{action.type.value}' is not in allowed_action_types",
            )

        target_url = action.url or current_url
        if not self._url_allowed(target_url):
            return PolicyDecision(
                allowed=False,
                risk_level=effective_risk.risk,
                requires_human_confirmation=effective_risk.requires_human_confirmation,
                reason=f"'{target_url}' is outside every allowlisted target/route",
            )

        if effective_risk.blocked:
            source = f"route rule '{route_rule.route_pattern}'" if route_rule is not None else f"intent '{action.intent}'"
            return PolicyDecision(
                allowed=False,
                risk_level=effective_risk.risk,
                requires_human_confirmation=effective_risk.requires_human_confirmation,
                reason=f"{source} is blocked by risk_policy.yaml",
            )

        if route_rule is not None:
            reason = (
                f"route rule '{route_rule.route_pattern}' overrides intent '{action.intent}': "
                f"risk={effective_risk.risk.value}"
            )
        else:
            reason = f"intent '{action.intent}' resolved to risk={effective_risk.risk.value}"

        return PolicyDecision(
            allowed=True,
            risk_level=effective_risk.risk,
            requires_human_confirmation=effective_risk.requires_human_confirmation,
            reason=reason,
        )

    def _match_route_rule(self, action: Action, current_url: str) -> RouteRiskRule | None:
        path = urlparse(current_url).path or "/"
        for rule in self._risk_policy.route_risk_rules:
            if action.type not in rule.action_types:
                continue
            if not fnmatch.fnmatch(path, rule.route_pattern):
                continue
            if self._route_rule_excludes(action, rule.exclude):
                continue
            return rule
        return None

    def _route_rule_excludes(self, action: Action, exclude: RouteRuleExclusion) -> bool:
        if action.intent in exclude.intents:
            return True
        if exclude.locator_text and action.target is not None:
            haystack = " ".join(
                [action.target.description] + [candidate.value for candidate in action.target.candidates]
            ).lower()
            if any(text.lower() in haystack for text in exclude.locator_text):
                return True
        return False

    def _url_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path or "/"
        for target in self._allowlist.allowed_targets:
            target_parsed = urlparse(target.base_url)
            if parsed.netloc and parsed.netloc != target_parsed.netloc:
                continue
            if not target.allowed_routes:
                return True
            if any(fnmatch.fnmatch(path, route) for route in target.allowed_routes):
                return True
        return False
