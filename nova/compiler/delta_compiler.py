from __future__ import annotations

from typing import List

from nova.compiler.delta_instruction import (
    DeltaInstruction,
    DeltaOp,
)

from nova.compiler.linker import LinkInstruction


class DeltaCompiler:

    def compile(self, graph):

        delta = []

        links = []

        for node in graph.nodes.values():

            delta.append(

                DeltaInstruction(

                    DeltaOp.ACTIVATE,

                    target=node.uid,

                    value=node.confidence,

                )

            )

        for edge in graph.edges:

            links.append(

                LinkInstruction(

                    source=edge.source,

                    relation=edge.relation,

                    target=edge.target,

                    weight=edge.weight,

                )

            )

        return delta, links