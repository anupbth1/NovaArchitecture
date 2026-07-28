from .instruction import Opcode


class NovaExecutor:

    def execute(

        self,

        program,

        state,

    ):

        stack = []

        memory = {}

        for ins in program.instructions:

            if ins.opcode == Opcode.LOAD:

                stack.append(ins.operand)

            elif ins.opcode == Opcode.STORE:

                memory[ins.operand] = stack.pop()

            elif ins.opcode == Opcode.FIND:

                stack.append(

                    state.data.get(

                        ins.operand

                    )

                )

            elif ins.opcode == Opcode.COMPARE:

                b = stack.pop()

                a = stack.pop()

                stack.append(a == b)

            elif ins.opcode == Opcode.RETURN:

                state.result = stack.pop()

                return state

        return state