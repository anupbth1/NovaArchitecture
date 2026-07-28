"""
Phase 2: Working Memory
========================
Hidden state vector ke代替 mein — ek set of typed bindings.

Rule: No hidden state vectors. No d_model. No tensors.
Only: (role, value) pairs in a dynamic set.

role  = integer ID (0 .. ROLE_SPACE-1) — what kind of binding
value = integer — the actual value (hash, id, position, etc.)

Operations: add, remove, match, merge — all O(log N) with indexing.

Memory per binding: 2 × 4 bytes = 8 bytes (vs 2048×4 = 8192 per hidden state)
"""
from typing import List, Tuple, Set, Dict, Optional, Callable
from collections import defaultdict


# Each binding is (role, value) — 8 bytes total
Binding = Tuple[int, int]  # (role_id, value)


class WorkingMemory:
    """
    A set of (role, value) bindings.
    
    This replaces the hidden state vector in Transformers.
    Instead of one big vector (d_model floats), we have
    a variable-sized set of discrete bindings.
    
    Benefits:
    - No fixed size (can grow/shrink as needed)
    - Interpretable (each binding has a semantic role)
    - Sparse (only store what's relevant)
    - Fast (O(log N) operations with indexing)
    """
    
    def __init__(self):
        # Primary storage: list of all bindings
        self.bindings: List[Binding] = []
        
        # Indexes for fast lookup
        self._by_role: Dict[int, List[int]] = defaultdict(list)  # role → indices
        self._by_value: Dict[int, List[int]] = defaultdict(list)  # value → indices
    
    def add(self, role: int, value: int) -> bool:
        """Add a binding. Returns True if new, False if already exists."""
        b = (role, value)
        if b in self:  # Already exists
            return False
        idx = len(self.bindings)
        self.bindings.append(b)
        self._by_role[role].append(idx)
        self._by_value[value].append(idx)
        return True
    
    def remove_role(self, role: int) -> int:
        """Remove all bindings with this role. Returns count removed."""
        indices = sorted(self._by_role.get(role, []), reverse=True)
        count = 0
        for idx in indices:
            self._remove_at(idx)
            count += 1
        return count
    
    def remove_value(self, value: int) -> int:
        """Remove all bindings with this value. Returns count removed."""
        indices = sorted(self._by_value.get(value, []), reverse=True)
        count = 0
        for idx in indices:
            self._remove_at(idx)
            count += 1
        return count
    
    def _remove_at(self, idx: int):
        """Remove binding at index, updating indexes."""
        if idx >= len(self.bindings):
            return
        role, value = self.bindings[idx]
        
        # Remove from indexes
        if role in self._by_role and idx in self._by_role[role]:
            self._by_role[role].remove(idx)
        if value in self._by_value and idx in self._by_value[value]:
            self._by_value[value].remove(idx)
        
        # Remove from list (swap with last for O(1))
        last_idx = len(self.bindings) - 1
        if idx != last_idx:
            # Move last binding to this position
            last_role, last_value = self.bindings[last_idx]
            self.bindings[idx] = (last_role, last_value)
            # Update indexes for moved binding
            self._by_role[last_role].remove(last_idx)
            self._by_role[last_role].append(idx)
            self._by_value[last_value].remove(last_idx)
            self._by_value[last_value].append(idx)
        
        self.bindings.pop()
    
    def match(self, role: Optional[int] = None, value: Optional[int] = None) -> List[Binding]:
        """
        Find bindings matching criteria.
        
        Args:
            role: Filter by role (None = any role)
            value: Filter by value (None = any value)
        
        Returns:
            List of matching (role, value) pairs
        """
        if role is not None and value is not None:
            # Specific (role, value) pair
            b = (role, value)
            return [b] if b in self else []
        
        if role is not None:
            # All bindings with this role
            return [self.bindings[i] for i in self._by_role.get(role, [])]
        
        if value is not None:
            # All bindings with this value
            return [self.bindings[i] for i in self._by_value.get(value, [])]
        
        # All bindings
        return list(self.bindings)
    
    def has(self, role: int, value: int) -> bool:
        """Check if a specific binding exists."""
        for idx in self._by_role.get(role, []):
            if self.bindings[idx][1] == value:
                return True
        return False
    
    def __contains__(self, binding: Binding) -> bool:
        role, value = binding
        for idx in self._by_role.get(role, []):
            if self.bindings[idx][1] == value:
                return True
        return False
    
    def __len__(self) -> int:
        return len(self.bindings)
    
    def __repr__(self) -> str:
        if not self.bindings:
            return "WM(empty)"
        parts = [f"{r}→{v}" for r, v in self.bindings[:10]]
        if len(self.bindings) > 10:
            parts.append(f"...(+{len(self.bindings)-10} more)")
        return f"WM({' | '.join(parts)})"
    
    def clear(self):
        """Clear all bindings."""
        self.bindings.clear()
        self._by_role.clear()
        self._by_value.clear()
    
    def copy(self) -> 'WorkingMemory':
        """Create a copy."""
        wm = WorkingMemory()
        for role, value in self.bindings:
            wm.add(role, value)
        return wm
    
    def merge(self, other: 'WorkingMemory') -> 'WorkingMemory':
        """Merge two memories. Conflicts: other wins."""
        result = self.copy()
        for role, value in other.bindings:
            result.add(role, value)
        return result
    
    def values_for_role(self, role: int) -> List[int]:
        """Get all values for a role."""
        return [self.bindings[i][1] for i in self._by_role.get(role, [])]
    
    def count(self, role: Optional[int] = None) -> int:
        """Count bindings (optionally for a specific role)."""
        if role is None:
            return len(self.bindings)
        return len(self._by_role.get(role, []))


# ============================================================
# ROLE VOCABULARY (learned, discrete)
# ============================================================

class RoleVocabulary:
    """
    Maps role names to integer IDs.
    
    Unlike embeddings, these are HARD ASSIGNMENTS — not vectors.
    Role 42 = ROLE_SUBJECT. Always. No vector representation.
    """
    
    # Built-in roles (used by the system)
    BUILTINS = {
        "POSITION": 0,
        "TOKEN_HASH": 1,
        "TOKEN_TYPE": 2,
        "TOKEN_FREQ": 3,
        "PREDICTED_HASH": 4,
        "CONFIDENCE": 5,
        "SUBJECT": 10,
        "VERB": 11,
        "OBJECT": 12,
        "MODIFIER": 13,
        "TENSE": 14,
        "NUMBER": 15,
        "GENDER": 16,
        "RELATION": 20,
        "CAUSE": 21,
        "EFFECT": 22,
        "CONTEXT": 30,
        "QUESTION": 40,
        "ANSWER": 41,
    }
    
    def __init__(self):
        self._name_to_id = dict(self.BUILTINS)
        self._id_to_name = {v: k for k, v in self.BUILTINS.items()}
        self._next_id = max(self.BUILTINS.values()) + 1
    
    def get_id(self, name: str, create: bool = True) -> int:
        """Get role ID from name. Creates new ID if not found."""
        if name in self._name_to_id:
            return self._name_to_id[name]
        if not create:
            return -1
        idx = self._next_id
        self._next_id += 1
        self._name_to_id[name] = idx
        self._id_to_name[idx] = name
        return idx
    
    def get_name(self, role_id: int) -> str:
        """Get role name from ID."""
        return self._id_to_name.get(role_id, f"role_{role_id}")
    
    @property
    def size(self) -> int:
        return self._next_id


# ============================================================
# DEMO / TEST
# ============================================================

def demo():
    print("=" * 60)
    print("Nova V2 — Phase 2: Working Memory")
    print("=" * 60)
    print()
    print("CONSTRAINTS:")
    print("  ❌ No hidden state vectors")
    print("  ❌ No d_model / hidden_size")
    print("  ❌ No tensors")
    print("  ✅ Only: (role, value) bindings in a set")
    print()
    
    roles = RoleVocabulary()
    wm = WorkingMemory()
    
    print("Adding bindings for 'The cat sat on the mat':")
    wm.add(roles.get_id("POSITION"), 0)
    wm.add(roles.get_id("TOKEN_HASH"), 534022)  # "The"
    wm.add(roles.get_id("SUBJECT"), 550141)      # "cat"
    wm.add(roles.get_id("VERB"), 567565)          # "sat"
    wm.add(roles.get_id("RELATION"), 620802)      # "on"
    wm.add(roles.get_id("OBJECT"), 561031)        # "mat"
    print(f"  {wm}")
    print(f"  Size: {len(wm)} bindings")
    
    print()
    print("Matching (role=SUBJECT, value=ANY):")
    for role, value in wm.match(role=roles.get_id("SUBJECT")):
        role_name = roles.get_name(role)
        print(f"  {role_name} → {value}")
    
    print()
    print("Removing all bindings with role=RELATION:")
    wm.remove_role(roles.get_id("RELATION"))
    print(f"  {wm}")
    
    print()
    print("Memory comparison:")
    wm_bytes = len(wm.bindings) * 8  # 8 bytes per binding
    hs_bytes = 2048 * 4  # typical hidden state: 2048 floats
    print(f"  Working memory: {len(wm.bindings)} bindings × 8 bytes = {wm_bytes} bytes")
    print(f"  Hidden state:   2048 floats × 4 bytes = {hs_bytes} bytes")
    print(f"  Savings: {hs_bytes // max(wm_bytes, 1)}x less memory")
    
    return wm, roles


if __name__ == '__main__':
    demo()