from .goal import Goal, GoalType


class GoalPlanner:

    def build(self, query: str):

        goals = []

        goals.append(

            Goal(

                GoalType.UNDERSTAND,

                "Understand user intent",

                priority=100,

            )

        )

        goals.append(

            Goal(

                GoalType.RETRIEVE,

                "Retrieve relevant concepts",

                priority=90,

            )

        )

        goals.append(

            Goal(

                GoalType.REASON,

                "Reason over retrieved knowledge",

                priority=80,

            )

        )

        goals.append(

            Goal(

                GoalType.VERIFY,

                "Verify reasoning",

                priority=70,

            )

        )

        goals.append(

            Goal(

                GoalType.RESPOND,

                "Generate final answer",

                priority=60,

            )

        )

        return sorted(

            goals,

            key=lambda x: x.priority,

            reverse=True,

        )