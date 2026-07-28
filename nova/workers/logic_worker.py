from nova.workers.base import Worker


class LogicWorker(Worker):

    name = "logic"

    priority = 80

    def can_handle(self, state):

        return "memory_hits" in state.data

    def execute(self, state, context):

        reasoning = []

        for hit in state.data["memory_hits"]:

            reasoning.append(

                {

                    "concept": hit.concept.value,

                    "confidence": hit.score,

                }

            )

        state.data["reasoning"] = reasoning

        return state