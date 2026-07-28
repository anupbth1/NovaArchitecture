"""
NovaBrain - The Orchestration Core
===================================
Manages the pipeline of modules: Parser → Reasoner → Memory → Planner → Verifier → Generator
Uses RCV for the actual computation.
"""
from .state import BrainState
from .context import BrainContext
from .task import BrainTask, TaskType
from .scheduler import TaskScheduler
from nova.reasoning.engine import ReasoningEngine
from nova.reasoning.goal import Goal, GoalType
from nova.workers.manager import WorkerManager


class BrainModule:
    """Base class for brain pipeline modules."""
    
    name = "module"
    
    def execute(self, state, context):
        return state


class NovaBrain:
    """
    Orchestrates the entire pipeline.
    
    Pipeline: Parser → Reasoner → Memory → Planner → Verifier → Generator
    
    Each module is executed in order. The pipeline can loop if the planner
    determines more reasoning is needed.
    """
    
    def __init__(self, context: BrainContext = None):
        self.context = context or BrainContext()
        self.pipeline = []
        self.scheduler = TaskScheduler()
        
        # Internal components
        self.workers = WorkerManager()
        self.reasoner = ReasoningEngine()
        
    def register(self, module):
        """Register a pipeline module."""
        self.pipeline.append(module)
    
    def _process_state(self, state, context):
        """Internal state processing (was dangling code)."""
        state = self.workers.execute(state, context)
        
        goal = Goal(GoalType.ANSWER, "Understand user query")
        self.reasoner.add_goal(goal)
        state = self.reasoner.run(state, context)
        
        return state
    
    def think(self, graph):
        """Submit a thinking task to the scheduler."""
        self.scheduler.submit(
            BrainTask.create(TaskType.UNDERSTAND, graph=graph)
        )
        
        tasks = []
        while not self.scheduler.empty():
            task = self.scheduler.next()
            tasks.append(task)
        
        return tasks
    
    def run(self, graph):
        """Run the full pipeline on a semantic graph."""
        state = BrainState()
        state.data["graph"] = graph
        
        # Process initial state
        state = self._process_state(state, self.context)
        
        # Main pipeline loop
        while not state.finished and state.step < self.context.max_steps:
            for module in self.pipeline:
                state = module.execute(state, self.context)
            state.step += 1
        
        return state