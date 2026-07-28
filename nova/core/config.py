from dataclasses import dataclass

@dataclass(slots=True)
class RuntimeConfig:

    device: str = "cuda"

    dtype: str = "float16"

    max_workers: int = 8


@dataclass(slots=True)
class ParserConfig:

    semantic_dim: int = 512

    max_semantic_units: int = 2048


@dataclass(slots=True)
class MemoryConfig:

    graph_size: int = 1_000_000

    cache_size: int = 100_000


@dataclass(slots=True)
class BrainConfig:

    reasoning_steps: int = 16

    planner_steps: int = 8


@dataclass(slots=True)
class NovaConfig:

    runtime = RuntimeConfig()

    parser = ParserConfig()

    memory = MemoryConfig()

    brain = BrainConfig()