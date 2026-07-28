"""Phase 3: Rule Engine - replaces Attention + MLP with structural pattern matching.

Rule: No attention. No MLP. No matrix multiplication. No GELU/ReLU.
Only: Pattern matching on memory bindings -> structural transformations.

A rule = (pattern, action, weight)
- pattern: conditions on memory M
- action: add/remove bindings from M  
- weight: learned scalar (not a matrix!)

Forward pass = for each rule, if pattern matches M, apply action to M.
Complexity: O(R x |M|) - linear in both rules and memory size.
"""
import sys, os
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from nova.v2.memory import WorkingMemory, Binding, RoleVocabulary
else:
    from nova.v2.memory import WorkingMemory, Binding, RoleVocabulary


# === PATTERN ===

@dataclass
class Pattern:
    """A condition on working memory.

    `required`: All these (role, value) must exist in memory.
    `forbidden`: None of these (role, value) must exist in memory.
    `value_constraints`: [(role, op, target)] - value must satisfy op(target).

    op: 'eq', 'neq', 'gt', 'lt', 'gte', 'lte', 'in', 'not_in'
    """
    required: List[Binding] = field(default_factory=list)
    forbidden: List[Binding] = field(default_factory=list)
    value_constraints: List[tuple] = field(default_factory=list)

    def matches(self, memory: WorkingMemory) -> bool:
        for role, value in self.required:
            if not memory.has(role, value):
                return False
        for role, value in self.forbidden:
            if memory.has(role, value):
                return False
        for role, op, target in self.value_constraints:
            values = memory.values_for_role(role)
            if not values:
                return False
            for v in values:
                if not self._check(v, op, target):
                    return False
        return True

    def _check(self, value: int, op: str, target) -> bool:
        if op == 'eq': return value == target
        if op == 'neq': return value != target
        if op == 'gt': return value > target
        if op == 'lt': return value < target
        if op == 'gte': return value >= target
        if op == 'lte': return value <= target
        if op == 'in': return value in target
        if op == 'not_in': return value not in target
        return True

    def __repr__(self):
        parts = []
        if self.required: parts.append(f"req={self.required}")
        if self.forbidden: parts.append(f"forbid={self.forbidden}")
        if self.value_constraints: parts.append(f"c={self.value_constraints}")
        return f"Pattern({', '.join(parts)})" if parts else "Pattern(ANY)"


# === ACTION ===

@dataclass
class Action:
    """A transformation on working memory."""
    add: List[Binding] = field(default_factory=list)
    remove_roles: List[int] = field(default_factory=list)
    remove_values: List[int] = field(default_factory=list)

    def apply(self, memory: WorkingMemory):
        for role in self.remove_roles:
            memory.remove_role(role)
        for value in self.remove_values:
            memory.remove_value(value)
        for role, value in self.add:
            memory.add(role, value)


# === RULE ===

@dataclass
class Rule:
    """A single rule: pattern -> action, with learned weight."""
    pattern: Pattern
    action: Action
    weight: float = 0.0
    firing_count: int = 0
    reward_sum: float = 0.0
    description: str = ""

    def fire(self, memory: WorkingMemory, context: dict = None) -> float:
        if self.pattern.matches(memory):
            self.action.apply(memory)
            self.firing_count += 1
            return self.weight
        return 0.0

    def record_reward(self, reward: float):
        self.reward_sum += reward

    @property
    def average_reward(self) -> float:
        return self.reward_sum / max(self.firing_count, 1)

    def __repr__(self):
        desc = f" - {self.description}" if self.description else ""
        return f"Rule(w={self.weight:.2f}, fires={self.firing_count}){desc}"


# === RULE ENGINE ===

class RuleEngine:
    """Replaces Attention + MLP with pattern matching rules."""

    def __init__(self):
        self.rules: List[Rule] = []
        self.total_confidence: float = 0.0

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def forward(self, memory: WorkingMemory, context: dict = None) -> WorkingMemory:
        self.total_confidence = 0.0
        for rule in self.rules:
            self.total_confidence += rule.fire(memory, context)
        return memory

    def forward_with_trace(self, memory: WorkingMemory, context: dict = None):
        trace = []
        self.total_confidence = 0.0
        for rule in self.rules:
            weight = rule.fire(memory, context)
            if weight > 0:
                trace.append((rule, weight))
            self.total_confidence += weight
        return memory, trace

    def update_weights(self, reward_fn: Callable, alpha: float = 0.01, base_rate: float = 0.0):
        for rule in self.rules:
            if rule.firing_count > 0:
                reward = reward_fn(rule)
                rule.record_reward(reward)
                rule.weight += alpha * (reward - base_rate)
                rule.weight = max(-10.0, min(10.0, rule.weight))

    def prune_weak_rules(self, threshold: float = -5.0):
        self.rules = [r for r in self.rules if r.weight > threshold]

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        if not self.rules:
            return "RuleEngine(empty)"
        active = sum(1 for r in self.rules if r.firing_count > 0)
        return f"RuleEngine({len(self.rules)} rules, {active} active)"


# === DEFAULT RULES ===

def create_default_rules(roles) -> List[Rule]:
    """Create starter rules for language processing."""
    r = []
    r.append(Rule(
        pattern=Pattern(value_constraints=[(roles.get_id("TOKEN_TYPE"), 'gte', 0)]),
        action=Action(add=[(roles.get_id("CONTEXT"), 0)]),
        weight=0.1, description="Record token as context",
    ))
    r.append(Rule(
        pattern=Pattern(value_constraints=[(roles.get_id("POSITION"), 'lte', 2)]),
        action=Action(add=[(roles.get_id("CONTEXT"), 1)]),
        weight=0.05, description="Early position -> beginning marker",
    ))
    r.append(Rule(
        pattern=Pattern(value_constraints=[(roles.get_id("TOKEN_TYPE"), 'eq', 1)]),
        action=Action(add=[(roles.get_id("CONTEXT"), 2)]),
        weight=0.05, description="Number token -> numeric context",
    ))
    r.append(Rule(
        pattern=Pattern(value_constraints=[(roles.get_id("TOKEN_TYPE"), 'eq', 2)]),
        action=Action(add=[(roles.get_id("CONTEXT"), 3)]),
        weight=0.05, description="Punctuation -> sentence end",
    ))
    return r


# === DEMO ===

def demo():
    print("=" * 60)
    print("Nova V2 - Phase 3: Rule Engine")
    print("=" * 60)
    print()
    print("CONSTRAINTS:")
    print("  No Attention (QKV)")
    print("  No MLP / FFN")
    print("  No matrix multiplication")
    print("  No GELU / ReLU")
    print("  Only: pattern matching -> structural actions")
    print()

    from nova.v2.memory import WorkingMemory, RoleVocabulary

    roles = RoleVocabulary()
    engine = RuleEngine()
    for rule in create_default_rules(roles):
        engine.add_rule(rule)

    print(f"Rules: {engine}\n")

    memory = WorkingMemory()
    memory.add(roles.get_id("POSITION"), 0)
    memory.add(roles.get_id("TOKEN_HASH"), 534022)
    memory.add(roles.get_id("TOKEN_TYPE"), 0)
    print(f"Memory before: {memory}\n")

    memory, trace = engine.forward_with_trace(memory)

    print("Forward pass results:")
    for rule, weight in trace:
        print(f"  Rule fired: {rule.description} (weight={weight:.2f})")
    print(f"  Total confidence: {engine.total_confidence:.2f}")
    print(f"Memory after: {memory}")

    print()
    print("NO matrix operations used:")
    print("  - Pattern matching: integer comparisons only")
    print("  - Actions: set add/remove only")
    print("  - Weights: scalars, not matrices")

    return engine


if __name__ == '__main__':
    demo()