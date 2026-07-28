"""
Reasoning Engine
================
Generates, verifies, and scores hypotheses to answer queries.
Uses iterative refinement (RCV-style) for deep reasoning.
"""
from collections import deque
from .goal import Goal
from .generator import HypothesisGenerator
from .scorer import HypothesisScorer
from .verifier import HypothesisVerifier


class HypothesisReasoner:
    """
    Reasoner that generates hypotheses, verifies them, and scores them.
    Uses iterative refinement to improve answers.
    """
    
    def __init__(self):
        self.generator = HypothesisGenerator()
        self.scorer = HypothesisScorer()
        self.verifier = HypothesisVerifier()
    
    def run(self, query, crystals):
        """Generate, verify, and score hypotheses."""
        hypotheses = self.generator.generate(query, crystals)
        
        ranked = []
        for h in hypotheses:
            h = self.verifier.verify(h)
            score = self.scorer.score(h)
            ranked.append((score, h))
        
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked


class ReasoningEngine:
    """
    Main reasoning engine that processes goals.
    Iteratively refines understanding through multiple reasoning steps.
    """
    
    def __init__(self):
        self.queue = deque()
        self.hypothesis_reasoner = HypothesisReasoner()
    
    def add_goal(self, goal: Goal):
        """Add a goal to the reasoning queue."""
        self.queue.append(goal)
    
    def run(self, brain_state, context):
        """Process all goals in the queue."""
        while self.queue:
            goal = self.queue.popleft()
            
            graph = brain_state.data.get("graph")
            if graph is None:
                continue
            
            concepts = list(graph.nodes.values())
            
            # Retrieve related concepts from memory
            related = []
            for node in concepts:
                neighbors = context.memory.neighbors(node.uid) if hasattr(context.memory, 'neighbors') else []
                related.extend(neighbors)
            
            # Store reasoning trace
            brain_state.data.setdefault("reasoning", []).append({
                "goal": goal.description if hasattr(goal, 'description') else str(goal),
                "concepts": len(concepts),
                "retrieved": len(related),
            })
        
        return brain_state