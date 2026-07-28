from dataclasses import dataclass, field

from .instruction import Instruction


@dataclass
class Program:

    instructions: list[Instruction] = field(default_factory=list)

    def add(self, opcode, operand=None):

        self.instructions.append(

            Instruction(

                opcode,

                operand,

            )

        )